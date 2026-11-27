// Question 2 - Tests de performance et validation CUDA
// Utilise les generateurs de la question 1 (vg.h).
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <vector>
#include "vg.h"

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"CUDA %s:%d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)

// Plusieurs blocs peuvent participer au calcul d'un meme prix.
struct Config {
    int nParam, nPaths, threads, pathsPerThread, reduction, rngMode;
    float T, K;
};

constexpr unsigned long long SEED = 20261113ULL;
constexpr float Z95 = 1.96f;

__global__ void init_rng(curandState *states, unsigned long long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        curand_init(seed, (unsigned long long)id, 0, &states[id]);
}

// 0 : reduction en shared ; 1 : shared puis shuffle dans le dernier warp.
__device__ void reduce3(double &a, double &b, double &c,
                        double *sa, double *sb, double *sc, int method) {
    int tid = threadIdx.x;
    sa[tid] = a;
    sb[tid] = b;
    sc[tid] = c;
    __syncthreads();
    if (method == 0) {
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                sa[tid] += sa[tid + stride];
                sb[tid] += sb[tid + stride];
                sc[tid] += sc[tid + stride];
            }
            __syncthreads();
        }
    } else {
        for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
            if (tid < stride) {
                sa[tid] += sa[tid + stride];
                sb[tid] += sb[tid + stride];
                sc[tid] += sc[tid + stride];
            }
            __syncthreads();
        }
        if (tid < 32) {
            a = sa[tid];
            b = sb[tid];
            c = sc[tid];
            for (int delta = 16; delta > 0; delta /= 2) {
                a += __shfl_down_sync(0xffffffff, a, delta);
                b += __shfl_down_sync(0xffffffff, b, delta);
                c += __shfl_down_sync(0xffffffff, c, delta);
            }
            if (tid == 0) {
                sa[0] = a;
                sb[0] = b;
                sc[0] = c;
            }
        }
    }
}

__global__ void simulate_partials(Config cfg,
                                  const float *params, curandState *rngStates,
                                  double *partialPay, double *partialPay2,
                                  double *partialY, int blocksPerParam,
                                  unsigned long long seed) {
    extern __shared__ double smem[];
    double *s1 = smem, *s2 = smem + blockDim.x, *s3 = smem + 2*blockDim.x;
    const int tid = threadIdx.x;
    const int p = blockIdx.x / blocksPerParam;
    const int gid = blockIdx.x * blockDim.x + tid;
    float kappa = params[3*p], theta = params[3*p+1], sigma = params[3*p+2];
    float w = logf(1.0f - theta*kappa - 0.5f*kappa*sigma*sigma) / kappa;
    float times[1] = {cfg.T}, X[1];
    curandState state;
    if (cfg.rngMode == 0) curand_init(seed, (unsigned long long)gid, 0, &state);
    else state = rngStates[gid];

    // Sommes locales avant reduction : on ne stocke pas les trajectoires.
    double acc = 0.0, acc2 = 0.0, accY = 0.0;
    for (int j = 0; j < cfg.pathsPerThread; ++j) {
        vg_path(sigma, theta, kappa, times, 1, X, &state);
        float y = expf(w*cfg.T + X[0]);
        float payoff = fmaxf(y - cfg.K, 0.0f);
        acc += payoff;
        acc2 += (double)payoff * payoff;
        accY += y;
    }
    reduce3(acc, acc2, accY, s1, s2, s3, cfg.reduction);
    if (tid == 0) {
        partialPay[blockIdx.x] = s1[0];
        partialPay2[blockIdx.x] = s2[0];
        partialY[blockIdx.x] = s3[0];
    }
}

// Regroupement des sommes partielles pour chaque prix.
__global__ void finalize(Config cfg, const double *part, const double *part2,
                         const double *partY, int blocksPerParam,
                         double *price, double *stderr, double *meanY) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= cfg.nParam)
        return;
    double sum = 0, sum2 = 0, sumY = 0;
    for (int b = 0; b < blocksPerParam; ++b) {
        int i = p*blocksPerParam + b;
        sum += part[i];
        sum2 += part2[i];
        sumY += partY[i];
    }
    double n = cfg.nPaths;
    double avg = sum / n;
    double variance = (sum2 - sum*avg) / (n-1.0);
    if (variance < 0)
        variance = 0;
    price[p] = avg;
    stderr[p] = sqrt(variance/n);
    meanY[p] = sumY / n;
}

struct Result { double ms, price, se, meanY; };

// Memes parametres pour toutes les configurations testees.
void create_parameters(std::vector<float> &h, int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dk(0.05f, 1.0f);
    std::uniform_real_distribution<float> dt(-0.5f, 0.2f);
    std::uniform_real_distribution<float> ds(0.1f, 0.5f);
    h.resize(3*n);
    for (int i=0; i<n; ++i) {
        h[3*i] = dk(rng);
        h[3*i+1] = dt(rng);
        h[3*i+2] = ds(rng);
    }
}

Result run_case(Config cfg, const std::vector<float> &hParams, int repetitions=5) {
    if (cfg.threads < 32 || cfg.threads > 1024 || (cfg.threads & (cfg.threads-1))) {
        fprintf(stderr,"threads doit etre une puissance de 2 entre 32 et 1024\n"); exit(1);
    }
    int perBlock = cfg.threads * cfg.pathsPerThread;
    if (cfg.nPaths < perBlock || cfg.nPaths % perBlock) {
        fprintf(stderr,"nPaths doit etre divisible par threads * pathsPerThread\n"); exit(1);
    }
    int bpp = cfg.nPaths / perBlock;
    int nBlocks = cfg.nParam * bpp;
    int nStates = nBlocks * cfg.threads;

    float *dParams = nullptr;
    double *dP = nullptr, *dP2 = nullptr, *dY = nullptr;
    double *dPrice = nullptr, *dSE = nullptr, *dMeanY = nullptr;
    curandState *dStates = nullptr;
    CHECK(cudaMalloc((void**)&dParams, 3*cfg.nParam*sizeof(float)));
    CHECK(cudaMemcpy(dParams, hParams.data(), 3*cfg.nParam*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&dP, nBlocks*sizeof(double)));
    CHECK(cudaMalloc((void**)&dP2, nBlocks*sizeof(double)));
    CHECK(cudaMalloc((void**)&dY, nBlocks*sizeof(double)));
    CHECK(cudaMalloc((void**)&dPrice, cfg.nParam*sizeof(double)));
    CHECK(cudaMalloc((void**)&dSE, cfg.nParam*sizeof(double)));
    CHECK(cudaMalloc((void**)&dMeanY, cfg.nParam*sizeof(double)));
    if (cfg.rngMode) {
        CHECK(cudaMalloc((void**)&dStates, (size_t)nStates*sizeof(curandState)));
        init_rng<<<(nStates+255)/256,256>>>(dStates,SEED,nStates);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    }
    auto launch = [&]() {
        simulate_partials<<<nBlocks,cfg.threads,3*cfg.threads*sizeof(double)>>>(
            cfg,dParams,dStates,dP,dP2,dY,bpp,SEED);
        CHECK(cudaGetLastError());
        finalize<<<(cfg.nParam+127)/128,128>>>(cfg,dP,dP2,dY,bpp,dPrice,dSE,dMeanY);
        CHECK(cudaGetLastError());
    };
    launch();
    CHECK(cudaDeviceSynchronize()); // Premier lancement pour echauffer le GPU
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float total=0.0f;
    for (int r=0; r<repetitions; ++r) {
        CHECK(cudaEventRecord(start));
        launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms=0;
        CHECK(cudaEventElapsedTime(&ms,start,stop));
        total += ms;
    }
    std::vector<double> hp(cfg.nParam), hs(cfg.nParam), hy(cfg.nParam);
    CHECK(cudaMemcpy(hp.data(),dPrice,cfg.nParam*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hs.data(),dSE,cfg.nParam*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hy.data(),dMeanY,cfg.nParam*sizeof(double),cudaMemcpyDeviceToHost));
    Result res{total/repetitions,hp[0],hs[0],hy[0]};
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dParams));
    CHECK(cudaFree(dP));
    CHECK(cudaFree(dP2));
    CHECK(cudaFree(dY));
    CHECK(cudaFree(dPrice));
    CHECK(cudaFree(dSE));
    CHECK(cudaFree(dMeanY));
    if (dStates)
        CHECK(cudaFree(dStates));
    return res;
}

void write_row(FILE *csv, const char *test, Config c, Result r) {
    double totalTraj = (double)c.nParam*c.nPaths;
    fprintf(csv,"%s,%d,%d,%d,%d,%d,%d,%.6f,%.8f,%.8f,%.8f,%.3f\n",
            test,c.nParam,c.nPaths,c.threads,c.pathsPerThread,c.reduction,c.rngMode,
            r.ms,r.price,r.se,r.meanY,totalTraj/(r.ms*1000.0));
    fflush(csv);
    printf("%-15s threads=%3d traj/thread=%3d N=%6d  temps=%8.3f ms  prix=%9.6f  SE=%.6f\n",
           test,c.threads,c.pathsPerThread,c.nPaths,r.ms,r.price,r.se);
}

void test_trajectoires(FILE *csv, const std::vector<float>& params, Config base) {
    // Meme nombre de trajectoires, avec une repartition differente.
    for (int p : {1,2,4,8,16,32,64,128}) {
        Config c=base; c.pathsPerThread=p;
        write_row(csv,"trajectoires",c,run_case(c,params));
    }
}
void test_blocs(FILE *csv, const std::vector<float>& params, Config base) {
    for (int threads : {64,128,256,512}) {
        Config c=base; c.threads=threads; c.pathsPerThread=16;
        write_row(csv,"blocs",c,run_case(c,params));
    }
}
void test_reduction(FILE *csv, const std::vector<float>& params, Config base) {
    for (int reduction : {0,1}) {
        Config c=base; c.reduction=reduction;
        write_row(csv,"reduction",c,run_case(c,params));
    }
}
void test_rng(FILE *csv, const std::vector<float>& params, Config base) {
    for (int mode : {0,1}) {
        Config c=base; c.rngMode=mode;
        write_row(csv,"rng",c,run_case(c,params));
    }
}
void test_convergence(FILE *csv, const std::vector<float>& params, Config base) {
    Config c=base; c.nParam=1; c.pathsPerThread=1;
    for (int n : {1024,4096,16384,65536,262144}) {
        c.nPaths=n;
        write_row(csv,"convergence",c,run_case(c,params,3));
    }
}

// Validation du premier prix avec les lois aleatoires de C++.
void test_reference_cpu(const std::vector<float>& params, Config base) {
    double kappa=params[0], theta=params[1], sigma=params[2];
    double correction=log(1-theta*kappa-0.5*kappa*sigma*sigma)/kappa;
    std::mt19937_64 rng(987654321);
    std::gamma_distribution<double> gamma(base.T/kappa,kappa);
    std::normal_distribution<double> normal(0,1);
    int n=300000;
    double sum=0,sum2=0,sumY=0;
    for (int i=0;i<n;i++) {
        double g=gamma(rng), z=normal(rng);
        double y=exp(correction*base.T+theta*g+sigma*sqrt(g)*z);
        double payoff=std::max(y-(double)base.K,0.0);
        sum+=payoff; sum2+=payoff*payoff; sumY+=y;
    }
    double price=sum/n;
    double se=sqrt(std::max(0.0,(sum2-sum*price)/(n-1.0))/n);
    Config gpu=base; gpu.nParam=1;
    Result r=run_case(gpu,params,3);
    printf("\nREFERENCE CPU (independante, N=%d) : prix=%.8f +/- %.8f, E[Y]=%.6f\n",
           n,price,Z95*se,sumY/n);
    printf("CUDA (N=%d)                  : prix=%.8f +/- %.8f, E[Y]=%.6f\n",
           gpu.nPaths,r.price,Z95*r.se,r.meanY);
    printf("Ecart CPU-CUDA=%.8f ; incertitude combinee approx.=%.8f\n\n",
           fabs(r.price-price),Z95*sqrt(se*se+r.se*r.se));
}

void usage(const char *prog) {
    printf("Usage: %s [all|trajectoires|blocs|reduction|rng|convergence|reference]\n",prog);
    printf("Resultats CSV: benchmarks_q2.csv (simulation + reduction finale, hors allocation et init RNG prealable).\n");
}

int main(int argc, char **argv) {
    std::string test=argc>1?argv[1]:"all";
    const std::string names[]={"all","trajectoires","blocs","reduction","rng","convergence","reference"};
    if (std::find(std::begin(names),std::end(names),test)==std::end(names)) {usage(argv[0]);return 1;}
    Config base{64,32768,256,128,0,0,1.0f,1.0f};
    std::vector<float> params;
    create_parameters(params,base.nParam);
    FILE *csv=fopen("benchmarks_q2.csv","w");
    if (!csv) {
        perror("benchmarks_q2.csv");
        return 1;
    }
    fprintf(csv,"test,n_params,n_paths,threads,paths_per_thread,reduction,rng_mode,time_ms,price_0,se_0,mean_y_0,paths_per_sec\n");
    if (test=="all"||test=="trajectoires") test_trajectoires(csv,params,base);
    if (test=="all"||test=="blocs") test_blocs(csv,params,base);
    if (test=="all"||test=="reduction") test_reduction(csv,params,base);
    if (test=="all"||test=="rng") test_rng(csv,params,base);
    if (test=="all"||test=="convergence") test_convergence(csv,params,base);
    if (test=="all"||test=="reference") test_reference_cpu(params,base);
    fclose(csv);
    puts("Resultats ecrits dans benchmarks_q2.csv");
    return 0;
}

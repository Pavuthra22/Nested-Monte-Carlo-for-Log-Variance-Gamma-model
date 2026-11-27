// Question 2 - Simulation Monte Carlo du modele Variance Gamma
// Un bloc traite une ligne de entree.csv.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "vg.h"

#define NB_PARAM 100000
#define THREADS 256
#define PATHS_PER_THREAD 128
#define Z95 1.96

#define CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "Erreur CUDA : %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

__global__ void vg_nested_mc(unsigned long long seed,
                             float *par, float *price, float *ci)
{
    // Parametres communs aux threads du bloc.
    __shared__ float params[5];
    __shared__ double sum[THREADS];
    __shared__ double sum2[THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Chaque bloc lit sa ligne du fichier d'entree.
    if (tid == 0) {
        for (int c = 0; c < 5; c++)
            params[c] = par[5 * blockIdx.x + c];
    }
    __syncthreads();

    float T     = params[0];
    float K     = params[1];
    float kappa = params[2];
    float theta = params[3];
    float sigma = params[4];

    // Correction martingale du prix de l'actif.
    float w = logf(1.0f - theta * kappa - 0.5f * kappa * sigma * sigma) / kappa;

    // Chaque thread simule PATHS_PER_THREAD trajectoires.
    curandState state;
    curand_init(seed ^ 0x9E3779B97F4A7C15ULL, idx, 0, &state);

    // On simule directement a l'echeance T.
    const float times[1] = {T};
    float X[1];
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;

    for (int p = 0; p < PATHS_PER_THREAD; p++) {
        vg_path(sigma, theta, kappa, times, 1, X, &state);
        float Y = expf(w * T + X[0]);
        float payoff = fmaxf(Y - K, 0.0f);

        local_sum += payoff;
        local_sum2 += payoff * payoff;
    }

    // Reduction des deux sommes en memoire partagee.
    sum[tid] = (double)local_sum;
    sum2[tid] = (double)local_sum2;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i) {
            sum[tid] += sum[tid + i];
            sum2[tid] += sum2[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Un thread termine le calcul du prix et de l'intervalle de confiance.
    if (tid == 0) {
        double n = (double)THREADS * PATHS_PER_THREAD;
        double mean = sum[0] / n;
        double var = (sum2[0] / n - mean * mean) * n / (n - 1.0);
        if (var < 0.0)
            var = 0.0;

        price[blockIdx.x] = (float)mean;
        ci[blockIdx.x] = (float)(Z95 * sqrt(var / n));
    }
}

// Lecture des cinq parametres de chaque ligne du CSV.
int lire_csv(const char *nom, float *par, int max)
{

    FILE *f = fopen(nom, "r");
    if (!f){
        return -1;
    }

    char ligne[256];
    fgets(ligne, sizeof(ligne), f); // ignore l'en-tete

    int n = 0;
    while (n < max && fgets(ligne, sizeof(ligne), f)) {
        if (sscanf(ligne, "%*d,%*d,%*d,%f,%f,%f,%f,%f",
                   &par[5 * n], &par[5 * n + 1], &par[5 * n + 2],
                   &par[5 * n + 3], &par[5 * n + 4]) != 5)
            break;
        n++;
    }
    fclose(f);
    return n;
}

int main()
{

    const unsigned long long seed = 1234ULL;

    // Allocation des donnees cote CPU.
    float *h_par = (float *)malloc(5 * NB_PARAM * sizeof(float));
    float *h_price = (float *)malloc(NB_PARAM * sizeof(float));
    float *h_ci = (float *)malloc(NB_PARAM * sizeof(float));
    if (!h_par || !h_price || !h_ci) {
        fprintf(stderr, "Erreur allocation CPU\n");
        free(h_par);
        free(h_price);
        free(h_ci);
        return EXIT_FAILURE;
    }

    // Lecture des parametres.
    int n = lire_csv("../data/entree.csv", h_par, NB_PARAM);
    if (n <= 0) {
        fprintf(stderr, "Impossible de lire entree.csv\n");
        return EXIT_FAILURE;
    }

    // Allocation des tableaux sur le GPU.
    float *d_par, *d_price, *d_ci;
    CUDA_CHECK(cudaMalloc(&d_par, 5 * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_price, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ci, n * sizeof(float)));

    // Copie des entrees vers le GPU.
    CUDA_CHECK(cudaMemcpy(d_par, h_par, 5 * n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vg_nested_mc<<<n, THREADS>>>(seed, d_par, d_price, d_ci);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Recuperation des prix et des demi-largeurs des intervalles.
    CUDA_CHECK(cudaMemcpy(h_price, d_price, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ci, d_ci, n * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *f = fopen("../data/sorties.csv", "w");
    if (!f) {
        fprintf(stderr, "Impossible d'ouvrir sorties.csv\n");
        return EXIT_FAILURE;
    }

    fprintf(f, "T,K,kappa,theta,sigma,price,ci_low,ci_high\n");
    for (int b = 0; b < n; b++) {
        fprintf(f, "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                h_par[5 * b], h_par[5 * b + 1], h_par[5 * b + 2],
                h_par[5 * b + 3], h_par[5 * b + 4],
                h_price[b], h_price[b] - h_ci[b], h_price[b] + h_ci[b]);
    }
    fclose(f);

    printf("%d prix ecrits dans ../data/sorties.csv (%.1f ms)\n", n, ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_par));
    CUDA_CHECK(cudaFree(d_price));
    CUDA_CHECK(cudaFree(d_ci));
    free(h_par);
    free(h_price);
    free(h_ci);
    return 0;
}

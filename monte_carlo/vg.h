#ifndef VG_H
#define VG_H

#include <curand_kernel.h>
#include <math.h>
#include "johnk.h"
#include "best.h"

//fait un pas : dS_i
__device__ inline float vg_increment(float sigma, float theta, float kappa, float dt, curandState *state){
    float a = dt / kappa;
    float G;
    if (a < 1.0f){
        G = gamma_johnk(a, state);
    }else{
        G = gamma_best(a, state);
    }
    float dS = kappa * G;
    float N = curand_normal(state);
    return sigma * N * sqrtf(dS) + theta * dS;
}

//construit la trajectoire X(t_i)
__device__ inline void vg_path(float sigma, float theta, float kappa, const float *t, int n, float *X, curandState *state){

    float x = 0.0f;
    float prev = 0.0f;
    for (int i = 0; i < n; i++){
            x += vg_increment(sigma, theta, kappa, t[i] - prev, state);
            X[i] = x;
            prev = t[i];
    }
}

#endif
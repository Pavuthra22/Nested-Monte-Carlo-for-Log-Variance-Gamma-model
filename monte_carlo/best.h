#ifndef BEST_H
#define BEST_H

#include <curand_kernel.h>
#include <math.h>

// Algorithme 6.8 : générateur de Best pour Gamma(a, 1), a >= 1
__device__ inline float gamma_best(float a, curandState *state)
{
    const float b = a - 1.0f;
    const float c = 3.0f * a - 0.75f;

    // [CORRECTION 2] a = 1 exactement : Gamma(1) = Exp(1).
    // Sans ce cas, b = 0 => b*logf(X/b) = 0*inf = NaN, le test d'acceptation
    // n'est jamais vrai et le noyau boucle indéfiniment.
    if (b == 0.0f)
        return -logf(curand_uniform(state));

    float U, V, W, Y, X, Z;
    while (true)
    {
        U = curand_uniform(state);
        V = curand_uniform(state);
        W = U * (1.0f - U);
        Y = sqrtf(c / W) * (U - 0.5f);
        X = b + Y;
        if (X < 0.0f)
            continue;
        // [CORRECTION 1] Z = 64 W^3 V^2 (et non V^3) : avec V^3 la loi simulée
        // n'est pas une Gamma(a) (variance biaisée, E[Y_T] != 1 dans le modèle VG).
        Z = 64.0f * W * W * W * V * V;
        if (logf(Z) <= 2.0f * (b * logf(X / b) - Y))
            return X;
    }
}
#endif

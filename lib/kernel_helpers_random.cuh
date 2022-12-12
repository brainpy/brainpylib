//
// Created by adadu on 2022/11/30.
//

#ifndef BRAINPYLIB_KERNEL_HELPERS_RANDOM_CUH
#define BRAINPYLIB_KERNEL_HELPERS_RANDOM_CUH

#include <curand_kernel.h>

namespace brainpy_lib{

    __device__ __forceinline__
    double taus88_double(unsigned int seed) {
        /**** VERY IMPORTANT **** :
          The initial seeds s1, s2, s3  MUST be larger than
          1, 7, and 15 respectively.
        */
        /* Generates numbers between 0 and 1. */
        unsigned int s1 = seed + 1, s2 = seed + 7, s3 = seed + 15, b;
        b = (((s1 << 13) ^ s1) >> 19);
        s1 = (((s1 & 4294967294) << 12) ^ b);
        b = (((s2 << 2) ^ s2) >> 25);
        s2 = (((s2 & 4294967288) << 4) ^ b);
        b = (((s3 << 3) ^ s3) >> 11);
        s3 = (((s3 & 4294967280) << 17) ^ b);
        return ((s1 ^ s2 ^ s3) * 2.3283064365386963e-10);
    }

    __device__ __forceinline__
    void taus88_double2(unsigned int seed, double* r1, double* r2) {
        /**** VERY IMPORTANT **** :
          The initial seeds s1, s2, s3  MUST be larger than
          1, 7, and 15 respectively.
        */
        /* Generates numbers between 0 and 1. */
        unsigned int s1 = seed + 1, s2 = seed + 7, s3 = seed + 15, b;
        b = (((s1 << 13) ^ s1) >> 19);
        s1 = (((s1 & 4294967294) << 12) ^ b);
        b = (((s2 << 2) ^ s2) >> 25);
        s2 = (((s2 & 4294967288) << 4) ^ b);
        b = (((s3 << 3) ^ s3) >> 11);
        s3 = (((s3 & 4294967280) << 17) ^ b);
        *r1 = ((s1 ^ s2 ^ s3) * 2.3283064365386963e-10);

        b = (((s1 << 13) ^ s1) >> 19);
        s1 = (((s1 & 4294967294) << 12) ^ b);
        b = (((s2 << 2) ^ s2) >> 25);
        s2 = (((s2 & 4294967288) << 4) ^ b);
        b = (((s3 << 3) ^ s3) >> 11);
        s3 = (((s3 & 4294967280) << 17) ^ b);
        *r2 = ((s1 ^ s2 ^ s3) * 2.3283064365386963e-10);
    }


    __device__ __forceinline__
    double lfsr113_double (unsigned int seed){
        /**** VERY IMPORTANT **** :
           The initial seeds z1, z2, z3, z4  MUST be larger than
           1, 7, 15, and 127 respectively.
        */
        unsigned int z1 = seed + 1, z2 = seed + 7, z3 = seed + 15, z4 = seed + 127, b;
        b  = ((z1 << 6) ^ z1) >> 13;
        z1 = ((z1 & 4294967294UL) << 18) ^ b;
        b  = ((z2 << 2) ^ z2) >> 27;
        z2 = ((z2 & 4294967288UL) << 2) ^ b;
        b  = ((z3 << 13) ^ z3) >> 21;
        z3 = ((z3 & 4294967280UL) << 7) ^ b;
        b  = ((z4 << 3) ^ z4) >> 12;
        z4 = ((z4 & 4294967168UL) << 13) ^ b;
        return (z1 ^ z2 ^ z3 ^ z4) * 2.3283064365386963e-10;
    }

};

#endif //BRAINPYLIB_KERNEL_HELPERS_RANDOM_CUH

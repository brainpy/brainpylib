//
// Created by adadu on 2022/12/11.
//

#ifndef BRAINPYLIB_CPU_JITCONN_MATVEC_H
#define BRAINPYLIB_CPU_JITCONN_MATVEC_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>


namespace brainpy_lib{
    void matvec_prob_homo_float(void **out, const void **in);
    void matvec_prob_homo_double(void **out, const void **in);

    void matvec_prob_uniform_float(void **out, const void **in);
    void matvec_prob_uniform_double(void **out, const void **in);

    void matvec_prob_normal_float(void **out, const void **in);
    void matvec_prob_normal_double(void **out, const void **in);

    
}


#endif //BRAINPYLIB_CPU_JITCONN_MATVEC_H

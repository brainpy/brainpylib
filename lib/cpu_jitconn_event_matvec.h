//
// Created by adadu on 2022/12/11.
//

#ifndef BRAINPYLIB_CPU_JITCONN_EVENT_MATVEC_H
#define BRAINPYLIB_CPU_JITCONN_EVENT_MATVEC_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>


namespace brainpy_lib{
    void event_matvec_prob_homo_float_bool(void **out, const void **in);
    void event_matvec_prob_homo_float_float(void **out, const void **in);
    void event_matvec_prob_homo_double_bool(void **out, const void **in);
    void event_matvec_prob_homo_double_double(void **out, const void **in);
    
    void event_matvec_prob_uniform_float_bool(void **out, const void **in);
    void event_matvec_prob_uniform_float_float(void **out, const void **in);
    void event_matvec_prob_uniform_double_bool(void **out, const void **in);
    void event_matvec_prob_uniform_double_double(void **out, const void **in);
        
    void event_matvec_prob_normal_float_bool(void **out, const void **in);
    void event_matvec_prob_normal_float_float(void **out, const void **in);
    void event_matvec_prob_normal_double_bool(void **out, const void **in);
    void event_matvec_prob_normal_double_double(void **out, const void **in);
    
    
}


#endif //BRAINPYLIB_CPU_JITCONN_EVENT_MATVEC_H

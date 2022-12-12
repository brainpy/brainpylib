//
// Created by adadu on 2022/12/11.
//

#include "cpu_jitconn_vecmat.h"


namespace brainpy_lib {
    namespace {
        template<typename T1>
        void vecmat_prob_homo(void **out, const void **in) {
            const T1 *vector = reinterpret_cast<const T1 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[2]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[3]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[4]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_col);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist {0, 1};
            // keep same random with "matvec_prob_homo"
            for (unsigned int row_i = 0; row_i < num_row; ++row_i) {
                unsigned int col_i = (unsigned int) ceil(log(dist(rng)) / log_p);
                T1 v = vector[row_i];
                while (col_i < num_col) {
                    result[col_i] += v;
                    col_i += (unsigned int) ceil(log(dist(rng)) / log_p);
                }
            }
        }


        template<typename T1>
        void vecmat_prob_uniform(void **out, const void **in) {
            const T1 *vector = reinterpret_cast<const T1 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const T1 w_min = *reinterpret_cast<const T1 *>(in[2]);
            const T1 w_max = *reinterpret_cast<const T1 *>(in[3]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[4]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[5]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[6]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_col);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist1 {0, 1};
            std::uniform_real_distribution<T1> dist2 {w_min, w_max};
            // keep same random with "matvec_prob_uniform"
            for (unsigned int row_i = 0; row_i < num_row; ++row_i) {
                unsigned int col_i = (unsigned int) ceil(log(dist1(rng)) / log_p);
                T1 v = vector[row_i];
                while (col_i < num_col) {
                    result[col_i] += dist2(rng) * v;
                    col_i += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
            }
        }


        template<typename T1>
        void vecmat_prob_normal(void **out, const void **in) {
            const T1 *vector = reinterpret_cast<const T1 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const T1 w_mu = *reinterpret_cast<const T1 *>(in[2]);
            const T1 w_sigma = *reinterpret_cast<const T1 *>(in[3]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[4]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[5]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[6]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_col);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist1 {0, 1};
            std::normal_distribution<T1> dist2 {w_mu, w_sigma};
            // keep same random with "matvec_prob_normal"
            for (unsigned int row_i = 0; row_i < num_row; ++row_i) {
                unsigned int col_i = (unsigned int) ceil(log(dist1(rng)) / log_p);
                T1 v = vector[row_i];
                while (col_i < num_col) {
                    result[col_i] += dist2(rng) * v;
                    col_i += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
            }
        }


    }

    void vecmat_prob_homo_float(void **out, const void **in){vecmat_prob_homo<float>(out, in);}
    void vecmat_prob_homo_double(void **out, const void **in){vecmat_prob_homo<double>(out, in);}

    void vecmat_prob_uniform_float(void **out, const void **in){vecmat_prob_uniform<float>(out, in);}
    void vecmat_prob_uniform_double(void **out, const void **in){vecmat_prob_uniform<double>(out, in);}

    void vecmat_prob_normal_float(void **out, const void **in){vecmat_prob_normal<float>(out, in);}
    void vecmat_prob_normal_double(void **out, const void **in){vecmat_prob_normal<double>(out, in);}


}
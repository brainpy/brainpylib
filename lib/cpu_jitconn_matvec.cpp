//
// Created by adadu on 2022/12/11.
//

#include "cpu_jitconn_matvec.h"


namespace brainpy_lib {
    namespace {
        template<typename T1>
        void matvec_prob_homo(void **out, const void **in) {
            const T1 *vector = reinterpret_cast<const T1 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[2]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[3]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[4]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_row);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist {0, 1};
            for (unsigned int row_i = 0; row_i < num_row; ++row_i) {
                T1 r = 0;
                unsigned int col_i = (unsigned int) ceil(log(dist(rng)) / log_p);
                while (col_i < num_col) {
                    r += vector[col_i];
                    col_i += (unsigned int) ceil(log(dist(rng)) / log_p);
                }
                result[row_i] = r;
            }
        }


        template<typename T1>
        void matvec_prob_uniform(void **out, const void **in) {
            const T1 *vector = reinterpret_cast<const T1 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const T1 w_min = *reinterpret_cast<const T1 *>(in[2]);
            const T1 w_max = *reinterpret_cast<const T1 *>(in[3]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[4]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[5]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[6]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_row);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist1 {0, 1};
            std::uniform_real_distribution<T1> dist2 {w_min, w_max};
            for (unsigned int row_i = 0; row_i < num_row; ++row_i) {
                T1 r = 0;
                unsigned int col_i = (unsigned int) ceil(log(dist1(rng)) / log_p);
                while (col_i < num_col) {
                    r += dist2(rng) * vector[col_i];
                    col_i += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
                result[row_i] = r;
            }
        }


        template<typename T1>
        void matvec_prob_normal(void **out, const void **in) {
            const T1 *vector = reinterpret_cast<const T1 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const T1 w_mu = *reinterpret_cast<const T1 *>(in[2]);
            const T1 w_sigma = *reinterpret_cast<const T1 *>(in[3]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[4]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[5]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[6]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_row);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist1 {0, 1};
            std::normal_distribution<T1> dist2 {w_mu, w_sigma};
            for (unsigned int row_i = 0; row_i < num_row; ++row_i) {
                T1 r = 0;
                unsigned int col_i = (unsigned int) ceil(log(dist1(rng)) / log_p);
                while (col_i < num_col) {
                    r += dist2(rng) * vector[col_i];
                    col_i += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
                result[row_i] = r;
            }
        }


    }

    void matvec_prob_homo_float(void **out, const void **in){matvec_prob_homo<float>(out, in);}
    void matvec_prob_homo_double(void **out, const void **in){matvec_prob_homo<double>(out, in);}

    void matvec_prob_uniform_float(void **out, const void **in){matvec_prob_uniform<float>(out, in);}
    void matvec_prob_uniform_double(void **out, const void **in){matvec_prob_uniform<double>(out, in);}

    void matvec_prob_normal_float(void **out, const void **in){matvec_prob_normal<float>(out, in);}
    void matvec_prob_normal_double(void **out, const void **in){matvec_prob_normal<double>(out, in);}


}
//
// Created by adadu on 2022/12/11.
//

#include "cpu_jitconn_event_matvec.h"


namespace brainpy_lib {
    namespace {
        template<typename T1, typename T2>
        void event_matvec_prob_homo(void **out, const void **in) {
            const T2 *events = reinterpret_cast<const T2 *>(in[0]);
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
                    if (events[col_i]) {
                        r += 1;
                    }
                    col_i += (unsigned int) ceil(log(dist(rng)) / log_p);
                }
                result[row_i] = r;
            }
        }


        template<typename T1, typename T2>
        void event_matvec_prob_uniform(void **out, const void **in) {
            const T2 *events = reinterpret_cast<const T2 *>(in[0]);
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
                    T1 w = dist2(rng);
                    if (events[col_i])
                        r += w;
                    col_i += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
                result[row_i] = r;
            }
        }


        template<typename T1, typename T2>
        void event_matvec_prob_normal(void **out, const void **in) {
            const T2 *events = reinterpret_cast<const T2 *>(in[0]);
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
                    T1 w = dist2(rng);
                    if (events[col_i])
                        r += w;
                    col_i += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
                result[row_i] = r;
            }
        }


    }

    void event_matvec_prob_homo_float_bool(void **out, const void **in){event_matvec_prob_homo<float, bool>(out, in);}
    void event_matvec_prob_homo_double_bool(void **out, const void **in){event_matvec_prob_homo<double, bool>(out, in);}
    void event_matvec_prob_homo_float_float(void **out, const void **in){event_matvec_prob_homo<float, float>(out, in);}
    void event_matvec_prob_homo_double_double(void **out, const void **in){event_matvec_prob_homo<double, double>(out, in);}

    void event_matvec_prob_uniform_float_bool(void **out, const void **in){event_matvec_prob_uniform<float, bool>(out, in);}
    void event_matvec_prob_uniform_double_bool(void **out, const void **in){event_matvec_prob_uniform<double, bool>(out, in);}
    void event_matvec_prob_uniform_float_float(void **out, const void **in){event_matvec_prob_uniform<float, float>(out, in);}
    void event_matvec_prob_uniform_double_double(void **out, const void **in){event_matvec_prob_uniform<double, double>(out, in);}

    void event_matvec_prob_normal_float_bool(void **out, const void **in){event_matvec_prob_normal<float, bool>(out, in);}
    void event_matvec_prob_normal_double_bool(void **out, const void **in){event_matvec_prob_normal<double, bool>(out, in);}
    void event_matvec_prob_normal_float_float(void **out, const void **in){event_matvec_prob_normal<float, float>(out, in);}
    void event_matvec_prob_normal_double_double(void **out, const void **in){event_matvec_prob_normal<double, double>(out, in);}


}
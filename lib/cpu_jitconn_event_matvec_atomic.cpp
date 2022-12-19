//
// Created by adadu on 2022/12/19.
//

#include "cpu_jitconn_event_matvec_atomic.h"


namespace brainpy_lib {
    namespace {
        template<typename T1, typename T2>
        void event_matvec_atomic_prob_homo(void **out, const void **in) {
            const T2 *events = reinterpret_cast<const T2 *>(in[0]);
            const double log_p = *reinterpret_cast<const double *>(in[1]);
            const unsigned int seed = *reinterpret_cast<const unsigned int *>(in[2]);
            const unsigned int num_row = *reinterpret_cast<const unsigned int *>(in[3]);
            const unsigned int num_col = *reinterpret_cast<const unsigned int *>(in[4]);
            T1 *result = reinterpret_cast<T1 *>(out[0]);

            memset(&result[0], 0, sizeof(T1) * num_row);
            std::mt19937 rng {seed};
            std::uniform_real_distribution<double> dist {0, 1};
            for (unsigned int i_col = 0; i_col < num_col; ++i_col) {
                unsigned int i_row = (unsigned int) ceil(log(dist(rng)) / log_p);
                T2 event = events[i_col];
                while (i_row < num_row) {
                    if (event)
                        result[i_row] += 1;
                    i_row += (unsigned int) ceil(log(dist(rng)) / log_p);
                }
            }
        }


        template<typename T1, typename T2>
        void event_matvec_atomic_prob_uniform(void **out, const void **in) {
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
            T1 uniform_weight;
            for (unsigned int i_col = 0; i_col < num_col; ++i_col) {
                unsigned int i_row = (unsigned int) ceil(log(dist1(rng)) / log_p);
                T2 event = events[i_col];
                while (i_row < num_row) {
                    uniform_weight = dist2(rng);
                    if (event)
                        result[i_row] += uniform_weight;
                    i_row += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
            }

        }


        template<typename T1, typename T2>
        void event_matvec_atomic_prob_normal(void **out, const void **in) {
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
            T1 normal_weight;
            for (unsigned int i_col = 0; i_col < num_col; ++i_col) {
                unsigned int i_row = (unsigned int) ceil(log(dist1(rng)) / log_p);
                T2 event = events[i_col];
                while (i_row < num_row) {
                    normal_weight = dist2(rng);
                    if (event)
                        result[i_row] += normal_weight;
                    i_row += (unsigned int) ceil(log(dist1(rng)) / log_p);
                }
            }


        }


    }

    void event_matvec_atomic_prob_homo_float_bool(void **out, const void **in){event_matvec_atomic_prob_homo<float, bool>(out, in);}
    void event_matvec_atomic_prob_homo_double_bool(void **out, const void **in){event_matvec_atomic_prob_homo<double, bool>(out, in);}
    void event_matvec_atomic_prob_homo_float_float(void **out, const void **in){event_matvec_atomic_prob_homo<float, float>(out, in);}
    void event_matvec_atomic_prob_homo_double_double(void **out, const void **in){event_matvec_atomic_prob_homo<double, double>(out, in);}

    void event_matvec_atomic_prob_uniform_float_bool(void **out, const void **in){event_matvec_atomic_prob_uniform<float, bool>(out, in);}
    void event_matvec_atomic_prob_uniform_double_bool(void **out, const void **in){event_matvec_atomic_prob_uniform<double, bool>(out, in);}
    void event_matvec_atomic_prob_uniform_float_float(void **out, const void **in){event_matvec_atomic_prob_uniform<float, float>(out, in);}
    void event_matvec_atomic_prob_uniform_double_double(void **out, const void **in){event_matvec_atomic_prob_uniform<double, double>(out, in);}

    void event_matvec_atomic_prob_normal_float_bool(void **out, const void **in){event_matvec_atomic_prob_normal<float, bool>(out, in);}
    void event_matvec_atomic_prob_normal_double_bool(void **out, const void **in){event_matvec_atomic_prob_normal<double, bool>(out, in);}
    void event_matvec_atomic_prob_normal_float_float(void **out, const void **in){event_matvec_atomic_prob_normal<float, float>(out, in);}
    void event_matvec_atomic_prob_normal_double_double(void **out, const void **in){event_matvec_atomic_prob_normal<double, double>(out, in);}


}
//
// Created by giorgk on 6/25/26.
//

#ifndef NON_LINEAR_H
#define NON_LINEAR_H

#include <deal.II/lac/trilinos_vector.h>
#include <deque>
#include <string>

namespace npsat_flow{
    using namespace dealii;

    struct NonlinearState
    {
        unsigned int nl_iter = 0;

        // For Anderson: store x_k and f_k = G(x_k)-x_k history
        std::deque<dealii::TrilinosWrappers::MPI::Vector> x_hist;
        std::deque<dealii::TrilinosWrappers::MPI::Vector> f_hist;
        std::string anderson_status = "disabled";
        unsigned int anderson_m_used = 0;
        double anderson_max_alpha_seen = 0.0;
        double anderson_step_ratio = 0.0;

        void clear_history()
        {
            x_hist.clear();
            f_hist.clear();
            anderson_status = "history_cleared";
            anderson_m_used = 0;
        }
    };

    struct RelativeKParams
    {
        double r_min     = 0.1;//1e-8;   // residual floor
        double eps       = 0.05;   // smoothing length [L] (e.g. fraction of layer thickness)
        double power_p   = 2.0;    // for PowerSigmoid
    };

    inline double logistic_sigma(const double x)
    {
        if (x > 40.0)  return 1.0;
        if (x < -40.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }
}

#endif //NON_LINEAR_H

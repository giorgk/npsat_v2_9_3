//
// Created by giorgk on 6/25/26.
//

#ifndef NON_LINEAR_H
#define NON_LINEAR_H

#include <deal.II/lac/trilinos_vector.h>

namespace npsat_flow{
    using namespace dealii;

    struct NonlinearState
    {
        unsigned int nl_iter = 0;

        // For Anderson: store x_k and f_k = G(x_k)-x_k history
        std::deque<dealii::TrilinosWrappers::MPI::Vector> x_hist;
        std::deque<dealii::TrilinosWrappers::MPI::Vector> f_hist;

        void clear_history()
        {
            x_hist.clear();
            f_hist.clear();
        }
    };
}

#endif //NON_LINEAR_H

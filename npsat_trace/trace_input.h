//
// Created by giorgk on 6/23/2026.
//

#ifndef NPSAT_V2_TRACE_INPUT_H
#define NPSAT_V2_TRACE_INPUT_H

#include <deal.II/base/mpi.h>

#include "trace_structures.h"

namespace npsat_trace {
    using namespace dealii;

    class InputHandler {
    public:
        InputHandler();
        bool read_ini(int argc, char** argv);
        Trace_options tr_opt;
    private:
        MPI_Comm mpi_communicator;
        ConditionalOStream pcout;
        std::string configFile;
        std::string Version;
    };

    inline InputHandler::InputHandler()
        :
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        Version = "0.0.01";
    }

    inline bool InputHandler::read_ini(int argc, char** argv)
    {
        pcout << "Reading input file" << std::endl;
        return true;
    }
}

#endif //NPSAT_V2_TRACE_INPUT_H

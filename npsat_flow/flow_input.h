//
// Created by giorgk on 6/22/2026.
//

#ifndef NPSAT_V2_FLOW_INPUT_H
#define NPSAT_V2_FLOW_INPUT_H

#include <boost/mpi.hpp>
//#include <boost/program_options.hpp>

#include "flow_structures.h"

namespace npsat_flow {

    using namespace dealii;

    class Input_ini {
    public:
        Input_ini();
        bool read_ini(int argc, char **argv);
        user_options uo;

    private:
        MPI_Comm mpi_communicator;
        ConditionalOStream pcout;
        std::string configFile;
        std::string Version;
    };

    inline Input_ini::Input_ini()
        : mpi_communicator(MPI_COMM_WORLD),
          pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    {
        Version = "0.0.02";
    }

    inline bool Input_ini::read_ini(int argc, char** argv) {
        pcout << "Reading input file" << std::endl;

        return true;

    }


}

#endif //NPSAT_V2_FLOW_INPUT_H

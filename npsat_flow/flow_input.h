//
// Created by giorgk on 6/22/2026.
//

#ifndef NPSAT_V2_FLOW_INPUT_H
#define NPSAT_V2_FLOW_INPUT_H

#include <deal.II/base/mpi.h>
#include <boost/program_options.hpp>

#include "flow_structures.h"

namespace npsat_flow {

    using namespace dealii;
    namespace po = boost::program_options;

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
        po::options_description commandLineOptions("Command line options");
        commandLineOptions.add_options()
            ("version,v", "print version information")
            ("help,h", "Get a list of options in the configuration file")
            ("config,c", po::value<std::string >(), "Set configuration file")
            ;

        po::variables_map vm_cmd;
        po::store(po::parse_command_line(argc, argv, commandLineOptions), vm_cmd);

        if (vm_cmd.empty())
        {
            pcout << " To run NPSAT_v2 specify the configuration file as" << std::endl;
            pcout << "-c config" << std::endl << std::endl;;
            pcout << "Other command line options are:" << std::endl;
            pcout << commandLineOptions << std::endl;
            return false;
        }

        return true;

    }


}

#endif //NPSAT_V2_FLOW_INPUT_H

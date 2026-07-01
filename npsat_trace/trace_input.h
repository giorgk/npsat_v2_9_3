//
// Created by giorgk on 6/23/2026.
//

#ifndef NPSAT_V2_TRACE_INPUT_H
#define NPSAT_V2_TRACE_INPUT_H

#include <deal.II/base/mpi.h>
#include <boost/program_options.hpp>

#include "trace_structures.h"

namespace npsat_trace {
    using namespace dealii;
    namespace po = boost::program_options;

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
            pcout << " To run NPSAT_TRACE specify the configuration file as" << std::endl;
            pcout << "-c config" << std::endl << std::endl;;
            pcout << "Other command line options are:" << std::endl;
            pcout << commandLineOptions << std::endl;
            return false;
        }

        if (vm_cmd.count("version"))
        {
            pcout << "|------------------|" << std::endl;
            pcout << "|    NPSAT_TRACE   |" << std::endl;
            pcout << "| Version : " << Version <<" |" << std::endl;
            pcout << "|     by giork     |" << std::endl;
            pcout << "|------------------|" << std::endl;
            return false;
        }

        // Configuration file options
        po::options_description config_options("Configuration file options");
        config_options.add_options()
        //[Data]
        ("Data.Prefix", po::value<std::string>(), "Main prefix for input files")

        //[Simulation]
        ("Simulation.Delta_time_file", po::value<std::string>(), "Filename with one delta-time value per row")
        ("Simulation.Porosity", po::value<double>()->default_value(0.3), "Effective porosity used by particle tracing")
        ("Simulation.DtEps", po::value<double>()->default_value(0.01), "Time-step completion tolerance")
        ("Simulation.MaxProcessorExchanges", po::value<int>()->default_value(100), "Maximum particle exchanges per time step")
        ("Simulation.MaxStreamlineSteps", po::value<int>()->default_value(10000), "Maximum integration steps per particle streamline")
        ("Simulation.MaxAge", po::value<int>()->default_value(std::numeric_limits<int>::max()), "Maximum total particle travel time")

        //[Output]
        ("Output.Prefix", po::value<std::string>(), "Main prefix for output files")
        ("Output.Print_loaded_tria", po::value<int>()->default_value(0), "Print the loaded triangulation for debug")
        ("Output.Load_tria_exit", po::value<int>()->default_value(0), "Exit after Loading triangulationa")



        ;

        if (vm_cmd.count("help"))
        {
            pcout << " To run NPSAT_TRACE specify the configuration file as" << std::endl;
            pcout << "-c config" << std::endl << std::endl;;
            pcout << "Other command line options are:" << std::endl;
            pcout << commandLineOptions << std::endl;

            pcout << "NPSAT_TRACE configuration file options:" << std::endl;
            pcout << "(All options are case sensitive)" << std::endl;
            pcout << "------------------------------" << std::endl;
            pcout << config_options << std::endl;
            return false;
        }

        po::variables_map vm_cfg;
        if (vm_cmd.count("config")) {
            try
            {
                configFile = vm_cmd["config"].as<std::string>().c_str();
                pcout << "--> Configuration file: " << vm_cmd["config"].as<std::string>().c_str() << std::endl;
                po::store(po::parse_config_file<char>(vm_cmd["config"].as<std::string>().c_str(), config_options), vm_cfg);

                {//Data
                    tr_opt.input_prefix = vm_cfg["Data.Prefix"].as<std::string>();
                }
                {// Output
                    tr_opt.output_prefix = vm_cfg["Output.Prefix"].as<std::string>();
                    tr_opt.write_loaded_tria = vm_cfg["Output.Print_loaded_tria"].as<int>();
                    tr_opt.exit_after_load_tria = vm_cfg["Output.Load_tria_exit"].as<int>() != 0;
                }
                {// Simulation
                    tr_opt.delta_time_file = vm_cfg["Simulation.Delta_time_file"].as<std::string>();
                    tr_opt.sim_opt.porosity = vm_cfg["Simulation.Porosity"].as<double>();
                    tr_opt.sim_opt.dt_eps = vm_cfg["Simulation.DtEps"].as<double>();
                    tr_opt.sim_opt.n_max_proc_exchanges = vm_cfg["Simulation.MaxProcessorExchanges"].as<int>();
                    tr_opt.sim_opt.n_max_streamline_steps = vm_cfg["Simulation.MaxStreamlineSteps"].as<int>();
                    tr_opt.sim_opt.max_age = vm_cfg["Simulation.MaxAge"].as<int>();
                }

            }
            catch (std::exception& E)
            {
                pcout << E.what() << std::endl;
                return false;
            }
        }


        return true;
    }
}

#endif //NPSAT_V2_TRACE_INPUT_H

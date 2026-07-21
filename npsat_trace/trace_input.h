//
// Created by giorgk on 6/23/2026.
//

#ifndef NPSAT_V2_TRACE_INPUT_H
#define NPSAT_V2_TRACE_INPUT_H

#include <limits>

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
        Version = "0.0.02";
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
        ("Data.Prefix", po::value<std::string>(), "Main prefix for input files from flow simulation")
        ("Data.Particle_file", po::value<std::string>(), "Particle file name (include full path)")


        //[Simulation]
        ("Simulation.Delta_time_file", po::value<std::string>(), "Filename with one delta-time value per row")
        ("Simulation.Porosity", po::value<double>()->default_value(0.3), "Effective porosity used by particle tracing")
        ("Simulation.DtEps", po::value<double>()->default_value(0.01), "Time-step completion tolerance")
        ("Simulation.Direction", po::value<double>()->default_value(1.0), "Particle tracking direction: non-negative for forward, negative for backward")
        ("Simulation.StagnantVelocityThreshold", po::value<double>()->default_value(1.0e-8), "Velocity magnitude below which a particle is stagnant for the current time step")
        ("Simulation.WellCaptureDistance", po::value<double>()->default_value(10.0), "Near-well distance that always triggers a well-flow check")
        ("Simulation.WellCaptureCellFraction", po::value<double>()->default_value(0.2), "Near-well check distance as a fraction of cell diameter")
        ("Simulation.WellInfluenceQScale", po::value<double>()->default_value(1.0), "Well influence radius multiplier applied to sqrt(abs(Qe))")
        ("Simulation.WellInfluenceMaxCellFraction", po::value<double>()->default_value(0.45), "Maximum well influence radius as a fraction of cell diameter")
        ("Simulation.MaxProcessorExchanges", po::value<int>()->default_value(100), "Maximum particle exchanges per time step")
        ("Simulation.MaxStreamlineSteps", po::value<int>()->default_value(10000), "Maximum integration steps per particle streamline")
        ("Simulation.MaxNonExpandingSteps", po::value<int>()->default_value(50), "Terminate particles after this many non-expanding trajectory steps")
        ("Simulation.MaxAge", po::value<int>()->default_value(std::numeric_limits<int>::max()), "Maximum total particle travel time")
        ("Simulation.Max_particles_per_iter", po::value<int>()->default_value(20000), "Maximum number of particles per iterations")
        ("Simulation.VelocityInterpolation", po::value<std::string>()->default_value("split_rt0"), "Velocity interpolation scheme: split_rt0 or idw")

        //[IWD]
        ("IWD.Power", po::value<double>()->default_value(2.0), "Inverse-distance weighting power")
        ("IWD.ProximityTolerance", po::value<double>()->default_value(0.01), "Use a sample directly when the anisotropic distance is below this tolerance")
        ("IWD.AnisotropyRatio", po::value<double>()->default_value(0.0), "Vertical distance multiplier; zero estimates it from cell geometry")


        //[Output]
        ("Output.Prefix", po::value<std::string>(), "Main prefix for output files")
        ("Output.Print_loaded_tria", po::value<int>()->default_value(0), "Print the loaded triangulation for debug")
        ("Output.Load_tria_exit", po::value<int>()->default_value(0), "Exit after Loading triangulationa")
        ("Output.Write_bin", po::value<int>()->default_value(0), "Write streamline output in binary format")
        ("Output.Write_ascii", po::value<int>()->default_value(1), "Write streamline output in ASCII format")

        //[Misc]
        ("Misc.Dbg_prefix", po::value<std::string>(), "Main prefix for debug files")
        ("Misc.Init_cell_dbg", po::value<int>()->default_value(0), "Enable debug output for cell initialization")
        ("Misc.Particle_traj_dbg", po::value<int>()->default_value(0), "Enable debug output for particle trajectories")
        ("Misc.Cache_bilinear_coefficients", po::value<int>()->default_value(0), "Cache bilinear map coefficients in initialized cell velocity caches")




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
                    tr_opt.particles_file = vm_cfg["Data.Particle_file"].as<std::string>();
                }
                {// Output
                    tr_opt.output_prefix = vm_cfg["Output.Prefix"].as<std::string>();
                    tr_opt.write_loaded_tria = vm_cfg["Output.Print_loaded_tria"].as<int>();
                    tr_opt.exit_after_load_tria = vm_cfg["Output.Load_tria_exit"].as<int>() != 0;
                    tr_opt.write_bin = vm_cfg["Output.Write_bin"].as<int>() != 0;
                    tr_opt.write_ascii = vm_cfg["Output.Write_ascii"].as<int>() != 0;
                    if (!tr_opt.write_bin && !tr_opt.write_ascii)
                        throw std::runtime_error("At least one of Output.Write_bin or Output.Write_ascii must be enabled.");
                }
                {// Simulation
                    tr_opt.delta_time_file = vm_cfg["Simulation.Delta_time_file"].as<std::string>();
                    tr_opt.sim_opt.porosity = vm_cfg["Simulation.Porosity"].as<double>();
                    tr_opt.sim_opt.dt_eps = vm_cfg["Simulation.DtEps"].as<double>();
                    tr_opt.sim_opt.direction = (vm_cfg["Simulation.Direction"].as<double>() < 0.0) ? -1.0 : 1.0;
                    tr_opt.sim_opt.stagnant_velocity_threshold = std::max(0.0, vm_cfg["Simulation.StagnantVelocityThreshold"].as<double>());
                    tr_opt.sim_opt.well_capture_distance = vm_cfg["Simulation.WellCaptureDistance"].as<double>();
                    tr_opt.sim_opt.well_capture_cell_fraction = vm_cfg["Simulation.WellCaptureCellFraction"].as<double>();
                    tr_opt.sim_opt.well_influence_q_scale = vm_cfg["Simulation.WellInfluenceQScale"].as<double>();
                    tr_opt.sim_opt.well_influence_max_cell_fraction = vm_cfg["Simulation.WellInfluenceMaxCellFraction"].as<double>();
                    tr_opt.sim_opt.n_max_proc_exchanges = vm_cfg["Simulation.MaxProcessorExchanges"].as<int>();
                    tr_opt.sim_opt.n_max_streamline_steps = vm_cfg["Simulation.MaxStreamlineSteps"].as<int>();
                    tr_opt.sim_opt.n_max_nonexpanding_steps = vm_cfg["Simulation.MaxNonExpandingSteps"].as<int>();
                    tr_opt.sim_opt.max_age = vm_cfg["Simulation.MaxAge"].as<int>();
                    tr_opt.n_paticles_parallel = vm_cfg["Simulation.Max_particles_per_iter"].as<int>();
                    const std::string interpolation = vm_cfg["Simulation.VelocityInterpolation"].as<std::string>();
                    if (interpolation == "split_rt0")
                        tr_opt.sim_opt.velocity_interpolation = VelocityInterpolationScheme::split_rt0;
                    else if (interpolation == "idw")
                        tr_opt.sim_opt.velocity_interpolation = VelocityInterpolationScheme::idw;
                    else
                        throw std::runtime_error("Simulation.VelocityInterpolation must be 'split_rt0' or 'idw'.");
                }

                {// IWD
                    tr_opt.idw_opt.power = vm_cfg["IWD.Power"].as<double>();
                    tr_opt.idw_opt.proximity_tolerance = vm_cfg["IWD.ProximityTolerance"].as<double>();
                    tr_opt.idw_opt.anisotropy_ratio = vm_cfg["IWD.AnisotropyRatio"].as<double>();
                    if (!(tr_opt.idw_opt.power > 0.0))
                        throw std::runtime_error("IWD.Power must be greater than zero.");
                    if (!(tr_opt.idw_opt.proximity_tolerance > 0.0))
                        throw std::runtime_error("IWD.ProximityTolerance must be greater than zero.");
                    if (tr_opt.idw_opt.anisotropy_ratio < 0.0)
                        throw std::runtime_error("IWD.AnisotropyRatio must be zero (automatic) or greater than zero.");
                }

                {// Misc
                    tr_opt.misc_opt.dbg_prefix = vm_cfg["Misc.Dbg_prefix"].as<std::string>();
                    tr_opt.misc_opt.init_cell_dbg = vm_cfg["Misc.Init_cell_dbg"].as<int>() != 0;
                    tr_opt.misc_opt.particle_traj_dbg = vm_cfg["Misc.Particle_traj_dbg"].as<int>() != 0;
                    tr_opt.misc_opt.cache_bilinear_coefficients = vm_cfg["Misc.Cache_bilinear_coefficients"].as<int>() != 0;
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

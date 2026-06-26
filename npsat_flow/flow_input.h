//
// Created by giorgk on 6/22/2026.
//

#ifndef NPSAT_V2_FLOW_INPUT_H
#define NPSAT_V2_FLOW_INPUT_H

#include <deal.II/base/mpi.h>
#include <boost/program_options.hpp>

#include "flow_structures.h"
#include "helper_func.h"

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
        Version = "0.0.03";
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

        if (vm_cmd.count("version"))
        {
            pcout << "|------------------|" << std::endl;
            pcout << "|     NPSAT_v2     |" << std::endl;
            pcout << "| Version : " << Version <<" |" << std::endl;
            pcout << "|     by giork     |" << std::endl;
            pcout << "|------------------|" << std::endl;
            return false;
        }

        // Configuration file options
        po::options_description config_options("Configuration file options");
        config_options.add_options()
        //[Paths]
            ("Paths.Main", po::value<std::string>(), "Main simulation path")
            ("Paths.Input", po::value<std::string>(), "Path relative to Main for all input files")
            ("Paths.Output", po::value<std::string>(), "Path relative to Main for all output files")

            //[Geometry]
            ("Geometry.Type", po::value<std::string>(), "BOX or FILE")
            // Box options
            ("Geometry.BoxDims", po::value<std::string>(), "List as X,Y,Z")
            ("Geometry.BoxLLP", po::value<std::string>(), "List as X,Y,Z the coordinates of the left lowest point")
            // File options
            ("Geometry.MeshFileName", po::value<std::string>(), "Name of the mesh input file")
            //Common options
            ("Geometry.Top", po::value<std::string>(), "A number or the name of a 2D function")
            ("Geometry.Bottom", po::value<std::string>(), "A number or the name of a 2D function")

            //[Discretization]
            ("Discretization.Nxyz", po::value<std::string>(), "List as nx,ny,nz. This is used only with box")
            ("Discretization.Vertical", po::value<std::string>(), "A list of numbers 0,0.3,0.7,1 or file")

            //[Refinement]
            ("Refinement.Initial", po::value<int>()->default_value(1), "Initial refinement")
            ("Refinement.Wells", po::value<int>()->default_value(1), "Well refinement")
            ("Refinement.Streams", po::value<int>()->default_value(1), "Stream refinement")
            ("Refinement.Top", po::value<int>()->default_value(1), "Stream refinement")
            ("Refinement.TopDepth", po::value<int>()->default_value(1), "Stream refinement")
            ("Refinement.Dirichlet", po::value<int>()->default_value(1), "Stream refinement")
            ("Refinement.GHB", po::value<int>()->default_value(1), "Stream refinement")
            ("Refinement.Neumann", po::value<int>()->default_value(1), "Stream refinement")


            //[Boundary conditions] dirichlet
            ("BC.Dirichlet", po::value<std::string>(), "A file with Dirichlet boundary conditions")
            ("BC.Neumann", po::value<std::string>(), "A file with Neumann boundary conditions")
            ("BC.GHB", po::value<std::string>(), "A file with General Head boundary conditions")
            ("BC.HalfWidth", po::value<double>(), "Filename with well data")
            ("BC.MinOverlap", po::value<double>(), "Filename with well data")

            //[Properties]
            ("Properties.Kxy", po::value<std::string>(), "Horizontal hydraulic conductivity. A number or the name of a 3D function")
            ("Properties.Kz", po::value<std::string>(), "Vertical hydraulic conductivity. A number or the name of a 3D function")
            ("Properties.Sy", po::value<std::string>(), "Specific yield. A number or the name of a 3D function")
            ("Properties.Ss", po::value<std::string>(), "Specific storage. A number or the name of a 3D function")

            //[Sources]
            ("Sources.Rch_Data", po::value<std::string>()->default_value(""), "Filename with groundwater recharge spatial distribution function")
            ("Sources.Rch_factor", po::value<double>()->default_value(1), "Global Recharge multiplier")
            ("Sources.Well_Data", po::value<std::string>()->default_value(""), "Filename with well data")
            ("Sources.Well_Rates", po::value<std::string>()->default_value(""), "Filename with well pumping rate")
            ("Sources.Well_factor", po::value<double>()->default_value(1), "Global Well Pumping multiplier")
            ("Sources.Stream_Data", po::value<std::string>()->default_value(""), "Filename with stream data")
            ("Sources.Stream_Rates", po::value<std::string>()->default_value(""), "Filename with stream rates")
            ("Sources.Stream_factor", po::value<double>()->default_value(1), "Global Well Pumping multiplier")

            //[IC]
            ("IC.Head", po::value<std::string>(), "Filename with initial head function")

            //[Simulation]
            ("Simulation.Nsteps", po::value<int>()->default_value(24), "Number of time steps")
            ("Simulation.Start_step", po::value<int>()->default_value(0), "Start time step")
            ("Simulation.Delta_time_file", po::value<std::string>(), "Filename with time step data")
            ("Simulation.Confined", po::value<int>()->default_value(0), "Treat aquifer as confined and disable nonlinear unconfined K/Sy behavior")

            //[Solver]
            ("Solver.System_iterations", po::value<int>()->default_value(15000), "Iterations for system solver")
            ("Solver.System_tol", po::value<double>()->default_value(1e-12), "System solver tolerance")

            //[Nonlinear]
            ("Nonlinear.Iterations", po::value<unsigned int>()->default_value(25), "Maximum Picard iterations")
            ("Nonlinear.AbsTolUpdate", po::value<double>()->default_value(1e-2), "Absolute nonlinear update tolerance")
            ("Nonlinear.RelTolUpdate", po::value<double>()->default_value(1e-4), "Relative nonlinear update tolerance")
            ("Nonlinear.DampingOmega", po::value<double>()->default_value(0.7), "Picard damping factor")
            ("Nonlinear.UseAnderson", po::value<int>()->default_value(1), "Enable Anderson acceleration")
            ("Nonlinear.AndersonStart", po::value<unsigned int>()->default_value(2), "Nonlinear iteration to start Anderson acceleration")
            ("Nonlinear.AndersonM", po::value<unsigned int>()->default_value(5), "Anderson acceleration memory depth")
            ("Nonlinear.AndersonReg", po::value<double>()->default_value(1e-11), "Anderson least-squares regularization")
            ("Nonlinear.CarryHistoryAcrossTimesteps", po::value<int>()->default_value(0), "Carry Anderson history across time steps")
            ("Nonlinear.RechargeStabilizationMode", po::value<std::string>()->default_value("effective_top"), "Recharge stabilization mode: hysteresis_only or effective_top")
            ("Nonlinear.UseRechargeHysteresis", po::value<int>()->default_value(1), "Use hysteresis for recharge receiver wet/dry switching")
            ("Nonlinear.RechargeDryingSaturatedFraction", po::value<double>()->default_value(5.0e-3), "Minimum saturated thickness fraction for a previous recharge receiver to remain active")
            ("Nonlinear.RechargeWettingSaturatedFraction", po::value<double>()->default_value(5.0e-2), "Minimum saturated thickness fraction for a new recharge receiver to become active")
            ("Nonlinear.RechargeMinRelativeK", po::value<double>()->default_value(1.0e-4), "Minimum relative conductivity for a cell to receive routed recharge")
            ("Nonlinear.EffectiveTopMode", po::value<std::string>()->default_value("recharge_receivers"), "Effective top mode: off, recharge_receivers, or all_water_table_cells")

            //[Output]
            ("Output.Prefix", po::value<std::string>(), "Main prefix for output files")
            ("Output.Print_initial_mesh", po::value<int>()->default_value(0), "Write the initial mesh as VTK for serial runs")
            ("Output.Print_mesh_with_prop", po::value<int>()->default_value(0), "Write initial mesh with nodal properties and boundary-condition data")
            ("Output.Print_mesh_exit", po::value<int>()->default_value(0), "Write the initial mesh output and exit before setup/solve")
            ("Output.Save_trace_data", po::value<int>()->default_value(0), "Enable printing data for tracing simulation")
            ("Output.Print_vtk", po::value<int>()->default_value(0), "Enable VTK output")
            ("Output.Print_water_table", po::value<int>()->default_value(0), "Write water table DAT output")
            ("Output.Print_q_to_vtu", po::value<int>()->default_value(0), "Write face flux cell data to VTU/VTK outputs")
            ("Output.Print_lambda_to_vtu", po::value<int>()->default_value(0), "Write face lambda cell data to VTU/VTK outputs")
            ("Output.Print_area_to_vtu", po::value<int>()->default_value(0), "Write face area cell data to VTU/VTK outputs")
            ("Output.Print_cell_budget_csv", po::value<int>()->default_value(0), "Write cell budget CSV output")
            ("Output.Print_solution_cell_vtu", po::value<int>()->default_value(0), "Write solution cell VTU output")
            ("Output.Print_cellcenters_vtk", po::value<int>()->default_value(0), "Write cell-center point VTK output")
            ("Output.Print_cell_well_map_csv", po::value<int>()->default_value(0), "Write cell-well map CSV output")
            ("Output.Print_trace_well_maps_csv", po::value<int>()->default_value(0), "Write trace-well coupling map CSV output")
            ("Output.Print_wellbore_segments_csv", po::value<int>()->default_value(0), "Write wellbore segment CSV output")
            ("Output.Print_wellbore_segments_vtu", po::value<int>()->default_value(0), "Write wellbore segment VTU output")
            ("Output.Print_wellbore_segments_legacy_vtk", po::value<int>()->default_value(0), "Write wellbore segment legacy VTK output")
            ("Output.Print_wellboreflow_csv", po::value<int>()->default_value(0), "Write wellbore flow CSV output")
            ("Output.Print_well_resid_csv", po::value<int>()->default_value(0), "Write well residual CSV output")

            //[misc]
            ("Misc.Print_matrices", po::value<int>()->default_value(1), "Print matrices Debug only")
            ("Misc.Verbose_level", po::value<int>()->default_value(0), "How much output you want [0 1 2]")

        ;

        if (vm_cmd.count("help"))
        {
            pcout << " To run NPSAT_v2 specify the configuration file as" << std::endl;
            pcout << "-c config" << std::endl << std::endl;;
            pcout << "Other command line options are:" << std::endl;
            pcout << commandLineOptions << std::endl;

            pcout << "NPSAT_v2 configuration file options:" << std::endl;
            pcout << "(All options are case sensitive)" << std::endl;
            pcout << "------------------------------" << std::endl;
            pcout << config_options << std::endl;
            return false;
        }

        po::variables_map vm_cfg;
        if (vm_cmd.count("config")) {
            try {
                configFile = vm_cmd["config"].as<std::string>().c_str();
                pcout << "--> Configuration file: " << vm_cmd["config"].as<std::string>().c_str() << std::endl;
                po::store(po::parse_config_file<char>(vm_cmd["config"].as<std::string>().c_str(), config_options), vm_cfg);

                { // Paths
                    uo.main_path = vm_cfg["Paths.Main"].as<std::string>();
                    uo.input_path = vm_cfg["Paths.Input"].as<std::string>();
                    uo.output_path = vm_cfg["Paths.Output"].as<std::string>();
                }

                { // Geometry
                    uo.isBox = vm_cfg["Geometry.Type"].as<std::string>() == "BOX";
                    if (uo.isBox)
                    {
                        uo.box_dims = parse_list<double>(vm_cfg["Geometry.BoxDims"].as<std::string>());
                        uo.box_llp = parse_list<double>(vm_cfg["Geometry.BoxLLP"].as<std::string>());
                        uo.box_nxyz = parse_list<unsigned int>(vm_cfg["Discretization.Nxyz"].as<std::string>());
                    }
                    else
                    {
                        uo.mesh_file = vm_cfg["Geometry.MeshFileName"].as<std::string>();
                    }
                    const std::string vertical_discretization =
                        vm_cfg["Discretization.Vertical"].as<std::string>();
                    if (try_parse_list<double>(vertical_discretization, uo.vert_discr))
                    {
                        uo.vert_file.clear();
                    }
                    else
                    {
                        uo.vert_discr.clear();
                        uo.vert_file = vertical_discretization;
                    }
                    uo.top_fnc = vm_cfg["Geometry.Top"].as<std::string>();
                    uo.bot_fnc = vm_cfg["Geometry.Bottom"].as<std::string>();
                }

                {// Refinement
                    uo.ref_opt.initial = vm_cfg["Refinement.Initial"].as<int>();
                    uo.ref_opt.wells = vm_cfg["Refinement.Wells"].as<int>();
                    uo.ref_opt.streams = vm_cfg["Refinement.Streams"].as<int>();
                    uo.ref_opt.top = vm_cfg["Refinement.Top"].as<int>();
                    uo.ref_opt.top_depth = vm_cfg["Refinement.TopDepth"].as<int>();
                    uo.ref_opt.dirichlet = vm_cfg["Refinement.Dirichlet"].as<int>();
                    uo.ref_opt.GHB = vm_cfg["Refinement.GHB"].as<int>();
                    uo.ref_opt.neumann = vm_cfg["Refinement.Neumann"].as<int>();
                }

                {// Boundary conditions
                    uo.bndr_cond.dirichlet_file = vm_cfg["BC.Dirichlet"].as<std::string>();
                    uo.bndr_cond.ghb_file = vm_cfg["BC.GHB"].as<std::string>();
                    uo.bndr_cond.neumann_file = vm_cfg["BC.Neumann"].as<std::string>();
                    uo.bndr_cond.half_with = vm_cfg["BC.HalfWidth"].as<double>();
                    uo.bndr_cond.min_overlap = vm_cfg["BC.MinOverlap"].as<double>();
                }

                {// Properties
                    uo.hgeo.kx_file = vm_cfg["Properties.Kxy"].as<std::string>();
                    uo.hgeo.kz_file = vm_cfg["Properties.Kz"].as<std::string>();
                    uo.hgeo.sy_file = vm_cfg["Properties.Sy"].as<std::string>();
                    uo.hgeo.ss_file = vm_cfg["Properties.Ss"].as<std::string>();
                }

                {// Sources
                    uo.sources.rch_file = vm_cfg["Sources.Rch_Data"].as<std::string>();
                    uo.sources.rch_factor = vm_cfg["Sources.Rch_factor"].as<double>();
                    uo.sources.well_file = vm_cfg["Sources.Well_Data"].as<std::string>();
                    uo.sources.well_factor = vm_cfg["Sources.Well_factor"].as<double>();
                    uo.sources.well_rates_file = vm_cfg["Sources.Well_Rates"].as<std::string>();
                    uo.sources.stream_file = vm_cfg["Sources.Stream_Data"].as<std::string>();
                    uo.sources.stream_factor = vm_cfg["Sources.Stream_factor"].as<double>();
                    uo.sources.stream_rates_file = vm_cfg["Sources.Stream_Rates"].as<std::string>();
                }

                {//IC
                    uo.initial_head_file = vm_cfg["IC.Head"].as<std::string>();
                }

                {//Simulation
                    uo.sim_opt.n_steps = vm_cfg["Simulation.Nsteps"].as<int>();
                    uo.sim_opt.Start_step = vm_cfg["Simulation.Start_step"].as<int>();
                    uo.sim_opt.delta_time_file = vm_cfg["Simulation.Delta_time_file"].as<std::string>();
                    uo.sim_opt.confined = vm_cfg["Simulation.Confined"].as<int>() == 1;
                }

                {//Solver
                    uo.solver_opt.System_iterations = vm_cfg["Solver.System_iterations"].as<int>();
                    uo.solver_opt.System_tol = vm_cfg["Solver.System_tol"].as<double>();
                }

                {//Nonlinear
                    uo.NLC.max_picard_iters = vm_cfg["Nonlinear.Iterations"].as<unsigned int>();
                    uo.NLC.abs_tol_update = vm_cfg["Nonlinear.AbsTolUpdate"].as<double>();
                    uo.NLC.rel_tol_update = vm_cfg["Nonlinear.RelTolUpdate"].as<double>();
                    uo.NLC.damping_omega = vm_cfg["Nonlinear.DampingOmega"].as<double>();
                    uo.NLC.use_anderson = vm_cfg["Nonlinear.UseAnderson"].as<int>() == 1;
                    uo.NLC.anderson_start = vm_cfg["Nonlinear.AndersonStart"].as<unsigned int>();
                    uo.NLC.anderson_m = vm_cfg["Nonlinear.AndersonM"].as<unsigned int>();
                    uo.NLC.anderson_reg = vm_cfg["Nonlinear.AndersonReg"].as<double>();
                    uo.NLC.carry_history_across_timesteps = vm_cfg["Nonlinear.CarryHistoryAcrossTimesteps"].as<int>() == 1;
                    //uo.NLC.recharge_stabilization_mode = parse_recharge_stabilization_mode(vm_cfg["Nonlinear.RechargeStabilizationMode"].as<std::string>());
                    uo.NLC.use_recharge_hysteresis = vm_cfg["Nonlinear.UseRechargeHysteresis"].as<int>() == 1;
                    uo.NLC.recharge_drying_saturated_fraction = vm_cfg["Nonlinear.RechargeDryingSaturatedFraction"].as<double>();
                    uo.NLC.recharge_wetting_saturated_fraction = vm_cfg["Nonlinear.RechargeWettingSaturatedFraction"].as<double>();
                    uo.NLC.recharge_min_relative_k = vm_cfg["Nonlinear.RechargeMinRelativeK"].as<double>();
                    //uo.NLC.effective_top_mode = parse_effective_top_mode(vm_cfg["Nonlinear.EffectiveTopMode"].as<std::string>());
                    if (uo.NLC.recharge_stabilization_mode == NonlinearControls::RechargeStabilizationMode::HysteresisOnly)
                    {
                        uo.NLC.effective_top_mode = NonlinearControls::EffectiveTopMode::Off;
                    }
                }

                {// Output
                    uo.output_prefix = vm_cfg["Output.Prefix"].as<std::string>();
                    uo.print_initial_mesh = vm_cfg["Output.Print_initial_mesh"].as<int>() == 1;
                    uo.print_mesh_with_prop = vm_cfg["Output.Print_mesh_with_prop"].as<int>() == 1;
                    uo.print_mesh_exit = vm_cfg["Output.Print_mesh_exit"].as<int>() == 1;
                    uo.save_trace_data = vm_cfg["Output.Save_trace_data"].as<int>() == 1;
                    uo.print_vtk = vm_cfg["Output.Print_vtk"].as<int>() == 1;
                    uo.print_water_table = vm_cfg["Output.Print_water_table"].as<int>() == 1;
                    uo.print_q_to_vtu = vm_cfg["Output.Print_q_to_vtu"].as<int>() == 1;
                    uo.print_lambda_to_vtu = vm_cfg["Output.Print_lambda_to_vtu"].as<int>() == 1;
                    uo.print_area_to_vtu = vm_cfg["Output.Print_area_to_vtu"].as<int>() == 1;
                    uo.print_cell_budget_csv = vm_cfg["Output.Print_cell_budget_csv"].as<int>() == 1;
                    uo.print_solution_cell_vtu = vm_cfg["Output.Print_solution_cell_vtu"].as<int>() == 1;
                    uo.print_cellcenters_vtk = vm_cfg["Output.Print_cellcenters_vtk"].as<int>() == 1;
                    uo.print_cell_well_map_csv = vm_cfg["Output.Print_cell_well_map_csv"].as<int>() == 1;
                    uo.print_trace_well_maps_csv = vm_cfg["Output.Print_trace_well_maps_csv"].as<int>() == 1;
                    uo.print_wellbore_segments_csv = vm_cfg["Output.Print_wellbore_segments_csv"].as<int>() == 1;
                    uo.print_wellbore_segments_vtu = vm_cfg["Output.Print_wellbore_segments_vtu"].as<int>() == 1;
                    uo.print_wellbore_segments_legacy_vtk = vm_cfg["Output.Print_wellbore_segments_legacy_vtk"].as<int>() == 1;
                    uo.print_wellboreflow_csv = vm_cfg["Output.Print_wellboreflow_csv"].as<int>() == 1;
                    uo.print_well_resid_csv = vm_cfg["Output.Print_well_resid_csv"].as<int>() == 1;
                }

                { //Misc
                    uo.print_matrices = vm_cfg["Misc.Print_matrices"].as<int>() == 1;
                    uo.verbose_level = vm_cfg["Misc.Verbose_level"].as<int>();
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

#endif //NPSAT_V2_FLOW_INPUT_H

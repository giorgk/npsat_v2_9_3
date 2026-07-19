


#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/conditional_ostream.h>
#include  <deal.II/base/timer.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_trace.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/data_out.h>

#include  <deal.II/grid/grid_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include  <deal.II/base/timer.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>

#include "npsat_flow/flow_input.h"
#include "npsat_flow/time_step_tracking.h"
#include "npsat_flow/interpolation/interpolation_function.h"
#include "npsat_flow/interpolation/interp_interface.h"
#include "npsat_flow/hydrogeo_prop.h"
#include "npsat_flow/streams.h"
#include "npsat_flow/mnwells.h"
#include "npsat_flow/BC/dirichlet_bc.h"
#include "npsat_flow/BC/ghb_bc.h"
#include "npsat_flow/mesh_gen.h"
#include "npsat_flow/local_element_data.h"
#include "npsat_flow/dof_ownership.h"
#include "npsat_flow/mpi_helpers.h"
#include "npsat_flow/non_linear.h"

using namespace dealii;


template <int dim>
class NPSAT_FLOW {
public:
    NPSAT_FLOW(const unsigned int degree, const npsat_flow::user_options &uo_in);

    void run();

private:
    enum CheckpointPhase
    {
        checkpoint_phase_spinup = 0,
        checkpoint_phase_simulation = 1,
        checkpoint_phase_finished = 2
    };

    void set_simulation_data();
    void refine_triangulation();
    void initialize_local_cell_slots();
    // Methods related to setup system
    void setup_system();
    void apply_trace_boundary_conditions();
    void rebuild_trace_constraints();
    void setup_local_cell_well_link();
    void setup_well_index_sets_by_segments();
    void update_local_cell_well_link_owners();
    void build_trace_well_coupling_maps();
    void initialize_initial_head();
    void save_checkpoint(const unsigned int completed_spinup_iterations);
    unsigned int load_checkpoint();
    std::string checkpoint_base_path() const;

    // Methods related to assemble system
    void assemble_system();
    void identify_top_active_cells( std::vector<unsigned char> &recharge_receiver,
      std::vector<double> &receiver_recharge_area, std::vector<double> &receiver_effective_z_top);
    void compute_cell_r_and_storage(npsat_flow::CellNonlinearData &out, const typename DoFHandler<dim>::active_cell_iterator &cell,
          const std::vector<types::global_dof_index> &head_dof_indices, const double effective_z_top = std::numeric_limits<double>::quiet_NaN()) const;
    double effective_top_for_cell(const npsat_flow::CellNonlinearData &cell_data, double routed_receiver_effective_z_top) const;
    bool should_receive_gw_recharge(const std::vector<unsigned char> &recharge_receiver,
      const typename DoFHandler<dim>::active_cell_iterator &cell,const unsigned int i_face) const;

    // Methods related to solve
    void solve();
    void build_schur_rhs(TrilinosWrappers::MPI::Vector &schur_rhs);
    void back_substitute_well_heads(const TrilinosWrappers::MPI::Vector &lambda_owned);

    // Post process methods
    void compute_heads();
    void compute_fluxes();
    void compute_spinup_diagnostics(double &head_change_max,
      double &head_change_rms, double &storage_volume_change,
      double &storage_rate, double &storage_throughput_fraction) const;
    void compute_flux_change_diagnostics(
      const TrilinosWrappers::MPI::Vector &previous_flux,
      double &flux_change_max, double &flux_change_rms,
      double &flux_change_relative_l2) const;
    void compute_dirichlet_boundary_fluxes(
      double &dirichlet_inflow, double &dirichlet_outflow,
      double &dirichlet_net_outflow) const;
    void compute_update_norm(const TrilinosWrappers::MPI::Vector &h_prev, const TrilinosWrappers::MPI::Vector &h_next,
      double &update_norm, double &ref_norm, double &full_update_norm) const;
    bool check_nonlinear_convergence(const double update_norm, const double ref_norm) const;
    bool anderson_accelerate(TrilinosWrappers::MPI::Vector &x_accel,
                                       const TrilinosWrappers::MPI::Vector &x_k,
                                       const TrilinosWrappers::MPI::Vector &G_xk,
                                       npsat_flow::NonlinearState &nl_state,
                                       const npsat_flow::NonlinearControls &ctl) const;
    void update_anderson_history( const TrilinosWrappers::MPI::Vector &x_old,
                                  const TrilinosWrappers::MPI::Vector &x_new,
                                  npsat_flow::NonlinearState &nl_state,
                                  const npsat_flow::NonlinearControls &ctl) const;
    void clear_anderson_history(npsat_flow::NonlinearState &nl_state) const;
    void apply_damped_update(TrilinosWrappers::MPI::Vector &h_guess,
                                       const TrilinosWrappers::MPI::Vector &h_candidate,
                                       const double omega) const;

    // Save data for trace app
    void save_triangulation() const;
    void save_velocity_io_mapping_once() const;
    void export_cell_well_map_binary_once(const std::string &prefix) const;
    void save_water_table_per_step(const std::string &prefix) const;
    void save_velocity_per_step(const std::string &prefix) const;
    void build_and_write_vface_rt0_per_step(const std::string &prefix,
          TrilinosWrappers::MPI::Vector &vface_global) const;

    // Methods to print the output data
    void write_well_exchange_identity_csv_mpi(const std::string &prefix) const;
    void compute_wellbore_flows(const std::string &prefix) const;
    void write_wellbore_segments_csv_mpi(const std::string &prefix) const;
    void output_results(const std::string &prefix);

    std::string output_root_path() const;
    std::string output_prefix_path() const;
    void align_time_dependent_data();
    void write_final_mesh_with_properties() const;
    const std::string &cellid_string_from_active_index(const unsigned int aidx) const;

    // Debug only methods
    void print_matrix(const FullMatrix<double> &A, const std::string& name);
    void print_vector(const Vector<double> &A, const std::string& name);

    MPI_Comm mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    const unsigned int degree;

    // Physical units:
    // h [L]           // Head (length, e.g., meters)
    // q [L/T]         // Flux (velocity, e.g., m/day)
    // Λ [L]           // Trace = head on faces (meters)
    // Mathematical function spaces:
    // h ∈ L²(Ω)       // Square-integrable functions
    // q ∈ H(div, Ω)   // Functions with square-integrable divergence
    // Λ ∈ L²(∂K)      // Square-integrable on faces
    const FE_RaviartThomas<dim> fe_flux;    ///< RT element for flux in H(div).
    const FE_DGQ<dim> fe_head;              ///< DG element for cell-centered hydraulic head.
    const FE_FaceQ<dim> fe_trace;           ///< Face element for hybrid trace head lambda.

    DoFHandler<dim> dof_handler_flux; ///< DoF handler for flux unknowns.
    DoFHandler<dim> dof_handler_head; ///< DoF handler for cell head unknowns.
    DoFHandler<dim> dof_handler_trace; ///< DoF handler for trace-head unknowns.

    TrilinosWrappers::MPI::Vector solution_trace; ///< Solved trace-head vector with relevant ghost entries.

    TrilinosWrappers::BlockSparseMatrix     block_system_matrix; ///< Global block matrix for trace and well unknowns.
    TrilinosWrappers::MPI::BlockVector       block_rhs_vector; ///< Global block right-hand side for trace and well rows.
    TrilinosWrappers::MPI::BlockVector       block_solution; ///< Block solution containing trace and well values.
    TrilinosWrappers::MPI::Vector       well_solution; ///< Owned well-head solution values.
    TrilinosWrappers::MPI::Vector well_solution_ghosted; ///< Well-head values with ghost entries for segment calculations.

    IndexSet lambda_locally_owned_dofs; ///< Locally owned trace-head DoFs.
    IndexSet lambda_locally_relevant_dofs; ///< Locally relevant trace-head DoFs including ghosts.
    IndexSet flux_locally_owned_dofs; ///< Locally owned flux DoFs.
    IndexSet flux_locally_relevant_dofs; ///< Locally relevant flux DoFs including ghosts.
    IndexSet head_locally_owned_dofs; ///< Locally owned head DoFs.
    IndexSet head_locally_relevant_dofs; ///< Locally relevant head DoFs including ghosts.
    IndexSet well_locally_owned_dofs; ///< Locally owned global well ids modeled as algebraic DoFs.
    IndexSet well_locally_relevant_dofs; ///< Locally relevant well ids needed by local well-cell links.

    AffineConstraints<double> lambda_constraints; ///< Dirichlet constraints applied to trace-head DoFs.
    npsat_flow::OwnershipManager lambda_ownership; ///< Cached trace-DoF ownership lookup for routing MPI contributions.

    TrilinosWrappers::MPI::Vector h_old; ///< Head vector from the previous time step.
    TrilinosWrappers::MPI::Vector h_new; ///< Recovered head vector from the current nonlinear solve.
    TrilinosWrappers::MPI::Vector h_guess; ///< Current nonlinear head iterate used for coefficient evaluation.
    TrilinosWrappers::MPI::Vector q_new; ///< Recovered flux vector for the current solution.


    const npsat_flow::user_options uo;

    npsat_flow::InterpolationFunction<dim> gw_recharge;
    npsat_flow::HydraulicProperties<dim> hgeo_prop;
    npsat_flow::StreamCollection<dim> streams;
    npsat_flow::MNWellCollection mnwells;
    std::vector<std::pair<unsigned int, std::vector<npsat_flow::CellWellLink>>> local_cell_well_map; ///< Owned-cell to intersecting-well segment links.
    std::vector<std::pair<unsigned int, std::string>> local_cell_id_strings; ///< Stable CellId strings keyed by active cell index.
    std::vector<unsigned int> well_owner_rank; ///< MPI owner rank for each global well id.
    npsat_flow::SortedVectorMap<types::global_dof_index, std::vector<npsat_flow::WellRef>>  trace_to_well_dof; ///< Locally owned trace DoF to coupled well references.
    npsat_flow::SortedVectorMap<unsigned int, std::vector<npsat_flow::TraceRef>> well_to_trace_dof; ///< Locally owned well id to coupled trace references.


    npsat_flow::DirichletBoundary<dim> dirichlet_bc; ///< Parsed Dirichlet boundary condition definitions.
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_map; ///< Boundary id to Dirichlet function lookup.

    npsat_flow::GHBBoundary<dim> ghb_bc; ///< Parsed general-head-boundary condition definitions.
    std::map<types::boundary_id, npsat_flow::GHBFunctionPair<dim>> ghb_boundary_map; ///< Boundary id to GHB head/conductance functions.

    npsat_flow::LocalElementDataRT0DG0 local_element_data_rt_0dg0;

    npsat_flow::NonlinearState    nl_state; ///< Mutable nonlinear iteration counters and Anderson history.
    std::unordered_set<std::uint64_t> previous_recharge_receiver_gids; ///< Recharge receivers accepted in the previous nonlinear assembly.
    npsat_flow::RelativeKParams r_params; ///< Parameters controlling nonlinear relative conductivity and storage smoothing.

    ConditionalOStream pcout;
    TimerOutput computing_timer;
    Timer timer;
    npsat_flow::TimeStepTracker time_tracking;
    unsigned int my_rank; ///< MPI rank of this process.
    unsigned int n_proc; ///< Total number of MPI ranks.
    unsigned int checkpoint_slot = 1; ///< Last committed slot; the next save uses the other slot.
    unsigned int last_linear_iterations = 0;
    double last_recharge_total = 0.0;
    double last_stream_total = 0.0;
    double last_well_total = 0.0;
    double last_requested_pumping_magnitude = 0.0;
    double last_pumping_loss_magnitude = 0.0;
    double last_net_external_total = 0.0;
    unsigned int last_dry_well_count = 0;
    double last_head_min = 0.0;
    double last_head_max = 0.0;
    double last_head_mean = 0.0;
    double last_head_update_l2 = 0.0;
    std::ofstream dry_well_log;
    bool current_spinup_active = false;
    unsigned int current_spinup_solve = 0;


};

template <int dim>
NPSAT_FLOW<dim>::NPSAT_FLOW(const unsigned int degree, const npsat_flow::user_options &uo_in)
    :
    mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement))
    , degree(degree)
    , fe_flux(degree)
    , fe_head(degree)
    , fe_trace(degree)
    , dof_handler_flux(triangulation)
    , dof_handler_head(triangulation)
    , dof_handler_trace(triangulation)
    , uo(uo_in)
    , pcout(std::cout,
             (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                              pcout,
                              TimerOutput::never,
                              TimerOutput::wall_times)
{
    my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
}

#include "npsat_flow/main_class_impl/npsat_flow_prepare.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_setup.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_checkpoint.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_write_trace.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_assemble.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_solve.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_post.impl.h"
#include "npsat_flow/main_class_impl/npsat_flow_write_output.impl.h"

template <int dim>
void NPSAT_FLOW<dim>::run() {
    //std::cout << "I'm rank " << my_rank << " out of " << n_proc << std::endl;
    //std::cout << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;

    set_simulation_data();
    if (uo.print_mesh_exit)
    {
        pcout << "Output.Print_mesh_exit enabled: exiting after initial mesh output." << std::endl;
        return;
    }

    // Select Start_step forcing before setup_system() interpolates the initial
    // inhomogeneous Dirichlet constraint values.
    align_time_dependent_data();
    setup_system();
    unsigned int completed_spinup_iterations = 0;
    if (uo.sim_opt.restart_from_checkpoint)
        completed_spinup_iterations = load_checkpoint();
    else
        initialize_initial_head();

    if (uo.dry_well_log != 0)
    {
        std::ostringstream dry_log_name;
        dry_log_name << output_prefix_path() << "_dry_wells_rank_"
                     << std::setw(4) << std::setfill('0') << my_rank << ".csv";
        bool dry_log_has_content = false;
        if (uo.sim_opt.restart_from_checkpoint)
        {
            std::ifstream existing_dry_log(dry_log_name.str(), std::ios::binary);
            dry_log_has_content =
                existing_dry_log.good() && existing_dry_log.peek() != std::ifstream::traits_type::eof();
        }
        const std::ios_base::openmode mode =
            std::ios::out |
            (uo.sim_opt.restart_from_checkpoint ? std::ios::app : std::ios::trunc);
        dry_well_log.open(dry_log_name.str(), mode);
        AssertThrow(dry_well_log.good(),
                    ExcMessage("Could not open dry-well log: " + dry_log_name.str()));
        if (!dry_log_has_content)
            dry_well_log
                << "Eid,x,y,top_screen,bot_screen,wt,bot_screen_minus_wt,simulation_step,forcing_step,forcing_file_step,"
                   "spinup,spinup_solve,nonlinear_iteration,requested_pumping\n";
        dry_well_log << std::setprecision(16) << std::scientific;
    }

    TrilinosWrappers::MPI::Vector h_accel_owned;
    h_accel_owned.reinit(head_locally_owned_dofs, mpi_communicator);

    TrilinosWrappers::MPI::Vector previous_spinup_flux_owned;
    previous_spinup_flux_owned.reinit(flux_locally_owned_dofs, mpi_communicator);
    bool have_previous_spinup_flux = false;
    bool have_previous_spinup_dry_well_count = false;
    unsigned int previous_spinup_dry_well_count = 0;
    unsigned int stable_spinup_dry_well_solves = 0;
    bool have_previous_spinup_pumping_loss_fraction = false;
    double previous_spinup_pumping_loss_fraction = 0.0;
    unsigned int consecutive_spinup_passes = 0;

    double update_norm;
    double ref_norm;
    double full_update_norm;

    std::ofstream nonlinear_log;
    if (my_rank == 0 && !uo.log_file.empty())
    {
        const std::string log_path = npsat_flow::resolve_relative_path(output_root_path(), uo.log_file);
        nonlinear_log.open(log_path, std::ios::out | std::ios::trunc);
        AssertThrow(nonlinear_log.good(), ExcMessage("Could not open nonlinear log: " + log_path));
        nonlinear_log << "# Detailed nonlinear iteration log\n"
                      << "# ITER step iter update_inf threshold full_inf raw_l2 omega linear_iters "
                         "assemble_s solve_s recover_s convergence_s recharge streams wells "
                         "net_external dry_wells h_min h_max h_mean\n"
                      << "# ACCEPT step iter method history accepted_l2 aa_status m_used max_alpha step_ratio\n";
        nonlinear_log << std::setprecision(16) << std::scientific;
    }

    const bool spinup_enabled = (uo.spin_uo.iterations > 0);
    if (!spinup_enabled)
        pcout << "Spin-up disabled: Spinup.Iterations = 0." << std::endl;

    while (!time_tracking.done()) {
        const bool first_simulation_step = (time_tracking.simulation_step() == 0);
        const bool spinup_active = first_simulation_step && spinup_enabled;
        const unsigned int spinup_iteration_limit =
            (spinup_active ? uo.spin_uo.iterations : 1u);
        const unsigned int spinup_iteration_begin = (time_tracking.simulation_step() == 0 ? completed_spinup_iterations : 0u);

        for (unsigned int spinup_iteration = spinup_iteration_begin; spinup_iteration < spinup_iteration_limit; ++spinup_iteration)
        {
            current_spinup_active = spinup_active;
            current_spinup_solve = spinup_active ? spinup_iteration + 1 : 0;
            pcout << "\n==============================================" << std::endl;
            pcout << "Time step " << time_tracking.simulation_step()
                  << " of " << time_tracking.n_sim_steps() << std::endl;
            if (time_tracking.simulation_step() == 0 && spinup_iteration_limit > 1)
                pcout << "Spin-up solve " << (spinup_iteration + 1)
                      << " of at most " << spinup_iteration_limit << std::endl;

            align_time_dependent_data();
            rebuild_trace_constraints();

            // ------------------------------------------------------------
            // Nonlinear solve for this time step
            // ------------------------------------------------------------
            bool step_converged = uo.sim_opt.confined;
            {
                // Nonlinear iterate uses a head guess; initialize with previous time step head
                // h_old is needed as head of the previous time step to update RHS and calculate storage change
                h_guess = h_old;

                previous_recharge_receiver_gids.clear();

                if (!uo.NLC.carry_history_across_timesteps)
                    nl_state.clear_history();

                for (nl_state.nl_iter = 0; nl_state.nl_iter < uo.NLC.max_picard_iters; ++nl_state.nl_iter) {
                    typedef std::chrono::steady_clock StageClock;
                    StageClock::time_point stage_start;
                    double stage_local_seconds;
                    double assemble_seconds;
                    double solve_seconds;
                    double recover_seconds;
                    double convergence_seconds;

                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                          << " | assembly started..." << std::endl;
                    stage_start = StageClock::now();
                    assemble_system();
                    stage_local_seconds = std::chrono::duration<double>(StageClock::now()-stage_start).count();
                    assemble_seconds = Utilities::MPI::max(stage_local_seconds,mpi_communicator);
                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                          << " | assembly " << std::fixed << std::setprecision(2) << assemble_seconds
                          << " s | solver started..." << std::defaultfloat << std::endl;

                    stage_start = StageClock::now();
                    solve();
                    stage_local_seconds = std::chrono::duration<double>(StageClock::now()-stage_start).count();
                    solve_seconds = Utilities::MPI::max(stage_local_seconds,mpi_communicator);
                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                          << " | solver " << std::fixed << std::setprecision(2) << solve_seconds
                          << " s (" << last_linear_iterations << " iters) | head recovery started..."
                          << std::defaultfloat << std::endl;

                    stage_start = StageClock::now();
                    compute_heads();
                    stage_local_seconds = std::chrono::duration<double>(StageClock::now()-stage_start).count();
                    recover_seconds = Utilities::MPI::max(stage_local_seconds,mpi_communicator);
                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                          << " | head recovery " << std::fixed << std::setprecision(2) << recover_seconds
                          << " s | convergence check started..." << std::defaultfloat << std::endl;

                    if (uo.sim_opt.confined)
                    {
                        h_guess = h_new;
                        step_converged = true;
                        break;
                    }

                    // (4) Compute update norm for convergence checks
                    // compare previous head h_guess and new solution h_new
                    stage_start = StageClock::now();
                    compute_update_norm(h_guess, h_new, update_norm, ref_norm, full_update_norm);
                    stage_local_seconds = std::chrono::duration<double>(StageClock::now()-stage_start).count();
                    convergence_seconds = Utilities::MPI::max(stage_local_seconds,mpi_communicator);
                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                          << " | convergence check " << std::fixed << std::setprecision(2)
                          << convergence_seconds << " s" << std::defaultfloat << std::endl;

                    // Report convergence before preparing or writing the detailed log.
                    // In particular, do not let a slow network-filesystem write make the
                    // nonlinear solve appear stuck after the convergence timing line.
                    const bool nonlinear_converged_now = check_nonlinear_convergence(update_norm, ref_norm);

                    if (my_rank == 0 && nonlinear_log.is_open())
                    {
                        const double threshold = uo.NLC.abs_tol_update + uo.NLC.rel_tol_update * std::max(ref_norm, 1e-30);
                        nonlinear_log << "ITER " << time_tracking.simulation_step() << ' ' << nl_state.nl_iter << ' '
                                      << update_norm << ' ' << threshold << ' ' << full_update_norm << ' '
                                      << last_head_update_l2 << ' ' << uo.NLC.damping_omega << ' '
                                      << last_linear_iterations << ' ' << assemble_seconds << ' '
                                      << solve_seconds << ' ' << recover_seconds << ' '
                                      << convergence_seconds << ' ' << last_recharge_total << ' '
                                      << last_stream_total << ' ' << last_well_total << ' '
                                      << last_net_external_total << ' ' << last_dry_well_count << ' '
                                      << last_head_min << ' ' << last_head_max << ' ' << last_head_mean << '\n';
                    }

                    if (nonlinear_converged_now)
                    {
                        h_guess = h_new;
                        step_converged = true;
                        break;
                    }

                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                        << " | nonlinear update started..." << std::endl;
                    stage_start = StageClock::now();

                    // ----------------------------------------------------
                    // First build the damped Picard iterate.
                    // This is the actual nonlinear fixed-point map H(x).
                    // ----------------------------------------------------
                    TrilinosWrappers::MPI::Vector h_picard = h_guess;
                    apply_damped_update(h_picard, h_new, uo.NLC.damping_omega);

                    // Store the fixed-point pair (x_k, H(x_k)-x_k), not the accepted
                    // accelerated displacement. The current pair must be present before AA.
                    update_anderson_history(h_guess, h_picard, nl_state, uo.NLC);

                    const TrilinosWrappers::MPI::Vector *accepted=&h_picard;

                    // ----------------------------------------------------
                    // Accelerate the damped map instead of the raw Picard map
                    // ----------------------------------------------------
                    bool aa_ok = false;
                    if (uo.NLC.use_anderson)
                    {
                        aa_ok = anderson_accelerate(
                                    h_accel_owned,
                                    /*x_k=*/h_guess,
                                    /*G_xk=*/h_picard,
                                    nl_state,
                                    uo.NLC);

                        if (aa_ok)
                        {
                            accepted=&h_accel_owned;
                            if (uo.verbose_level > 0)
                                pcout << "  Anderson acceleration accepted at NL iter " << nl_state.nl_iter << std::endl;
                        }
                    }

                    // l2_norm() is an MPI collective for Trilinos vectors and must be
                    // called by every rank, even though only rank 0 writes the log file.
                    TrilinosWrappers::MPI::Vector accepted_step(*accepted);
                    accepted_step -= h_guess;
                    const double accepted_step_l2 = accepted_step.l2_norm();

                    if (my_rank == 0 && nonlinear_log.is_open())
                    {
                        nonlinear_log << "ACCEPT " << time_tracking.simulation_step() << ' '
                                      << nl_state.nl_iter << ' ' << (aa_ok ? "anderson" : "picard") << ' '
                                      << nl_state.x_hist.size() << ' ' << accepted_step_l2 << ' '
                                      << (uo.NLC.use_anderson ? nl_state.anderson_status : "disabled") << ' '
                                      << nl_state.anderson_m_used << ' ' << nl_state.anderson_max_alpha_seen << ' '
                                      << nl_state.anderson_step_ratio << '\n';
                    }

                    //apply_damped_update(h_guess, *target, uo.NLC.damping_omega);
                    // Accept the chosen iterate.
                    h_guess = *accepted;
                    h_guess.update_ghost_values();
                    stage_local_seconds = std::chrono::duration<double>(StageClock::now()-stage_start).count();
                    const double update_seconds = Utilities::MPI::max(stage_local_seconds,mpi_communicator);
                    pcout << "STG " << std::setw(3) << nl_state.nl_iter
                        << " | nonlinear update " << std::fixed << std::setprecision(2)
                        << update_seconds << " s | method " << (aa_ok ? "Anderson" : "Picard")
                        << " | AA " << (uo.NLC.use_anderson ? nl_state.anderson_status : "disabled")
                        << " | history " << nl_state.x_hist.size()
                        << std::defaultfloat << std::endl;
                    //break;

                }
            }

            AssertThrow(step_converged,
                        ExcMessage("Nonlinear solve did not converge; time was not advanced and the checkpoint was not updated."));

            // Flux recovery is required for every accepted spin-up solution so
            // that consecutive flow fields can be compared. For ordinary time
            // steps this is the same recovery that was previously done below,
            // immediately before output.
            compute_fluxes();

            double spinup_head_change = 0.0;
            double spinup_head_change_rms = 0.0;
            double spinup_storage_volume_change = 0.0;
            double spinup_storage_rate = 0.0;
            double spinup_storage_throughput_fraction = 0.0;
            double spinup_flux_change_max = 0.0;
            double spinup_flux_change_rms = 0.0;
            double spinup_flux_change_relative_l2 = 0.0;
            double spinup_dirichlet_inflow = 0.0;
            double spinup_dirichlet_outflow = 0.0;
            double spinup_dirichlet_net_outflow = 0.0;
            double spinup_budget_inflow = 0.0;
            double spinup_budget_outflow = 0.0;
            double spinup_budget_residual = 0.0;
            double spinup_budget_percent_discrepancy = 0.0;
            double spinup_pumping_loss_fraction = 0.0;
            double spinup_pumping_loss_fraction_change = 0.0;
            bool spinup_pumping_loss_stable = false;
            bool spinup_metrics_pass = false;
            bool spinup_converged = false;
            if (spinup_active)
            {
                compute_spinup_diagnostics(spinup_head_change,
                                           spinup_head_change_rms,
                                           spinup_storage_volume_change,
                                           spinup_storage_rate,
                                           spinup_storage_throughput_fraction);
                compute_dirichlet_boundary_fluxes(spinup_dirichlet_inflow,
                                                  spinup_dirichlet_outflow,
                                                  spinup_dirichlet_net_outflow);

                // Assemble a complete budget for the currently supported model:
                // recharge, streams, wells, Dirichlet exchange, and storage.
                // Positive source components enter the aquifer; positive storage
                // is accumulation and therefore belongs on the outflow side.
                spinup_budget_inflow =
                    std::max(last_recharge_total, 0.0) +
                    std::max(last_stream_total, 0.0) +
                    std::max(last_well_total, 0.0) +
                    spinup_dirichlet_inflow +
                    std::max(-spinup_storage_rate, 0.0);
                spinup_budget_outflow =
                    std::max(-last_recharge_total, 0.0) +
                    std::max(-last_stream_total, 0.0) +
                    std::max(-last_well_total, 0.0) +
                    spinup_dirichlet_outflow +
                    std::max(spinup_storage_rate, 0.0);
                spinup_budget_residual =
                    spinup_budget_inflow - spinup_budget_outflow;
                const double budget_sum =
                    spinup_budget_inflow + spinup_budget_outflow;
                spinup_budget_percent_discrepancy =
                    budget_sum > 0.0
                        ? 200.0 * spinup_budget_residual / budget_sum
                        : 0.0;
                if (have_previous_spinup_flux)
                    compute_flux_change_diagnostics(previous_spinup_flux_owned,
                                                    spinup_flux_change_max,
                                                    spinup_flux_change_rms,
                                                    spinup_flux_change_relative_l2);

                if (!have_previous_spinup_dry_well_count ||
                    last_dry_well_count != previous_spinup_dry_well_count)
                    stable_spinup_dry_well_solves = 1;
                else
                    ++stable_spinup_dry_well_solves;
                previous_spinup_dry_well_count = last_dry_well_count;
                have_previous_spinup_dry_well_count = true;

                spinup_pumping_loss_fraction =
                    last_requested_pumping_magnitude > 0.0
                        ? last_pumping_loss_magnitude / last_requested_pumping_magnitude
                        : 0.0;
                if (have_previous_spinup_pumping_loss_fraction)
                {
                    spinup_pumping_loss_fraction_change =
                        std::abs(spinup_pumping_loss_fraction -
                                 previous_spinup_pumping_loss_fraction);
                    spinup_pumping_loss_stable =
                        spinup_pumping_loss_fraction_change <=
                        uo.spin_uo.pumping_loss_stability_tolerance;
                }
                previous_spinup_pumping_loss_fraction =
                    spinup_pumping_loss_fraction;
                have_previous_spinup_pumping_loss_fraction = true;

                const unsigned int completed_solves = spinup_iteration + 1;
                spinup_metrics_pass =
                    have_previous_spinup_flux &&
                    completed_solves >= uo.spin_uo.minimum_solves &&
                    std::isfinite(spinup_head_change) &&
                    std::isfinite(spinup_head_change_rms) &&
                    std::isfinite(spinup_flux_change_relative_l2) &&
                    spinup_head_change <= uo.spin_uo.tolerance &&
                    spinup_head_change_rms <= uo.spin_uo.rms_head_tolerance &&
                    spinup_flux_change_relative_l2 <=
                        uo.spin_uo.flux_relative_l2_tolerance &&
                    spinup_pumping_loss_fraction <=
                        uo.spin_uo.pumping_loss_fraction_tolerance &&
                    spinup_pumping_loss_stable;

                if (spinup_metrics_pass)
                    ++consecutive_spinup_passes;
                else
                    consecutive_spinup_passes = 0;
                spinup_converged =
                    consecutive_spinup_passes >= uo.spin_uo.consecutive_passes;
            }
            const bool spinup_limit_reached =
                spinup_active && (spinup_iteration + 1 == spinup_iteration_limit);
            const bool final_spinup_iteration =
                !spinup_active || spinup_converged || spinup_limit_reached;

            if (spinup_active)
            {
                const double prescribed_throughput =
                    std::abs(last_recharge_total) + std::abs(last_stream_total) +
                    std::abs(last_well_total);
                pcout << "Spin-up diagnostics:" << std::scientific
                      << "\n  max |delta h| = " << spinup_head_change
                      << " m; guard = " << uo.spin_uo.tolerance << " m"
                      << "\n  RMS delta h = " << spinup_head_change_rms
                      << " m; tolerance = " << uo.spin_uo.rms_head_tolerance << " m"
                      << "\n  storage volume change = " << spinup_storage_volume_change
                      << " volume units"
                      << "\n  storage rate = " << spinup_storage_rate
                      << " volume/time"
                      << "\n  |storage rate| / prescribed-source throughput = "
                      << spinup_storage_throughput_fraction
                      << " (throughput = " << prescribed_throughput << " volume/time)"
                      << "\n  Dirichlet inflow = " << spinup_dirichlet_inflow
                      << " volume/time"
                      << "\n  Dirichlet outflow = " << spinup_dirichlet_outflow
                      << " volume/time"
                      << "\n  Dirichlet net outflow = "
                      << spinup_dirichlet_net_outflow << " volume/time"
                      << "\n  complete budget inflow = " << spinup_budget_inflow
                      << " volume/time"
                      << "\n  complete budget outflow = " << spinup_budget_outflow
                      << " volume/time"
                      << "\n  complete budget residual (in-out) = "
                      << spinup_budget_residual << " volume/time"
                      << "\n  complete budget percent discrepancy = "
                      << spinup_budget_percent_discrepancy << " %";
                if (have_previous_spinup_flux)
                    pcout << "\n  max |delta q coefficient| = "
                          << spinup_flux_change_max
                          << "\n  RMS delta q coefficient = "
                          << spinup_flux_change_rms
                          << "\n  relative flux L2 change = "
                          << spinup_flux_change_relative_l2
                          << "; tolerance = "
                          << uo.spin_uo.flux_relative_l2_tolerance;
                else
                    pcout << "\n  flux change = unavailable (first recovered spin-up field)";
                pcout << "\n  dry wells = " << last_dry_well_count
                      << "; unchanged-count solves = "
                      << stable_spinup_dry_well_solves
                      << " (diagnostic only)"
                      << "\n  requested pumping = "
                      << last_requested_pumping_magnitude << " volume/time"
                      << "\n  pumping loss from dry wells = "
                      << last_pumping_loss_magnitude << " volume/time"
                      << "\n  pumping-loss fraction = "
                      << spinup_pumping_loss_fraction
                      << "; tolerance = "
                      << uo.spin_uo.pumping_loss_fraction_tolerance
                      << "\n  pumping-loss fraction change = ";
                if (have_previous_spinup_pumping_loss_fraction && spinup_iteration > 0)
                    pcout << spinup_pumping_loss_fraction_change
                          << "; stability tolerance = "
                          << uo.spin_uo.pumping_loss_stability_tolerance
                          << " (" << (spinup_pumping_loss_stable ? "PASS" : "not yet") << ")";
                else
                    pcout << "unavailable (first spin-up solve)";
                pcout
                      << "\n  convergence metrics = "
                      << (spinup_metrics_pass ? "PASS" : "not yet")
                      << "; consecutive passes = "
                      << consecutive_spinup_passes << " of "
                      << uo.spin_uo.consecutive_passes
                      << std::defaultfloat << std::endl;

                for (auto dof = flux_locally_owned_dofs.begin();
                     dof != flux_locally_owned_dofs.end(); ++dof)
                    previous_spinup_flux_owned[*dof] = q_new[*dof];
                previous_spinup_flux_owned.compress(VectorOperation::insert);
                have_previous_spinup_flux = true;
            }

            if (!final_spinup_iteration)
            {
                h_old = h_new;
                save_checkpoint(spinup_iteration + 1);
                continue;
            }

            if (spinup_converged)
                pcout << "Spin-up converged after " << (spinup_iteration + 1)
                      << " solves." << std::endl;
            else if (spinup_active)
                pcout << "Spin-up iteration limit reached after "
                      << spinup_iteration_limit << " solves." << std::endl;

            //Printing output
            const std::string out_prefix = output_prefix_path();
            write_well_exchange_identity_csv_mpi(out_prefix);
            MPI_Barrier(mpi_communicator);
            compute_wellbore_flows(out_prefix);
            MPI_Barrier(mpi_communicator);
            write_wellbore_segments_csv_mpi(out_prefix);
            MPI_Barrier(mpi_communicator);
            output_results(out_prefix);
            MPI_Barrier(mpi_communicator);

            // Save data for trace
            export_cell_well_map_binary_once(out_prefix);
            MPI_Barrier(mpi_communicator);
            save_water_table_per_step(out_prefix);
            MPI_Barrier(mpi_communicator);
            save_velocity_per_step(out_prefix);
            MPI_Barrier(mpi_communicator);

            h_old = h_new;
            time_tracking.advance();
            save_checkpoint(0);
            // Re-enter the outer loop so TimeStepTracker::done() is checked after
            // advancing. Otherwise a converged spin-up exits step 0 but continues
            // through the stale spin-up for-loop limit.
            break;
        }
        completed_spinup_iterations = 0;
    }
    MPI_Barrier(mpi_communicator);
    pcout << "Simulation Finished" << std::endl;
}





int main(int argc, char **argv) {
    try {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,1);
        npsat_flow::Input_ini prm_ini;
        if (!prm_ini.read_ini(argc, argv))
            return 0;
        NPSAT_FLOW<3> npsat_flow(0, prm_ini.uo);
        npsat_flow.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}

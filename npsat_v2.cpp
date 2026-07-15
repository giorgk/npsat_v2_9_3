


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
    void set_simulation_data();
    void refine_triangulation();
    void initialize_local_cell_slots();
    // Methods related to setup system
    void setup_system();
    void apply_trace_boundary_conditions();
    void setup_local_cell_well_link();
    void setup_well_index_sets_by_segments();
    void update_local_cell_well_link_owners();
    void build_trace_well_coupling_maps();
    void initialize_initial_head();
    void save_checkpoint(const unsigned int completed_initial_repeats);
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
    double last_net_external_total = 0.0;
    unsigned int last_dry_well_count = 0;
    double last_head_min = 0.0;
    double last_head_max = 0.0;
    double last_head_mean = 0.0;
    double last_head_update_l2 = 0.0;


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

    setup_system();
    unsigned int completed_initial_repeats = 0;
    if (uo.sim_opt.restart_from_checkpoint)
        completed_initial_repeats = load_checkpoint();
    else
        initialize_initial_head();

    TrilinosWrappers::MPI::Vector h_accel_owned;
    h_accel_owned.reinit(head_locally_owned_dofs, mpi_communicator);

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

    while (!time_tracking.done()) {
        const unsigned int repeat_count = (time_tracking.simulation_step() == 0 ? uo.sim_opt.initial_step_repeats : 1u);
        const unsigned int repeat_begin = (time_tracking.simulation_step() == 0 ? completed_initial_repeats : 0u);

        for (unsigned int initial_repeat = repeat_begin; initial_repeat < repeat_count; ++initial_repeat)
        {
            pcout << "\n==============================================" << std::endl;
            pcout << "Time step " << time_tracking.simulation_step()
                  << " of " << time_tracking.n_sim_steps() << std::endl;
            if (time_tracking.simulation_step() == 0 && repeat_count > 1)
                pcout << "Initial stabilization solve " << (initial_repeat + 1)
                      << " of " << repeat_count << std::endl;

            align_time_dependent_data();

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

            const bool final_repeat = (initial_repeat + 1 == repeat_count);
            if (!final_repeat)
            {
                h_old = h_new;
                save_checkpoint(initial_repeat + 1);
                continue;
            }

            compute_fluxes();

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
            //break;
        }
        completed_initial_repeats = 0;
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

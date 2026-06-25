


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
#include "npsat_flow//mpi_helpers.h"

using namespace dealii;


template <int dim>
class NPSAT_FLOW {
public:
  NPSAT_FLOW(const unsigned int degree,
             const npsat_flow::user_options &uo_in);

  void run();

private:
  void set_simulation_data();
  void refine_triangulation();
  void initialize_local_cell_slots();

  void setup_system();
  void apply_trace_boundary_conditions();
  void setup_local_cell_well_link();
  void setup_well_index_sets_by_segments();
  void update_local_cell_well_link_owners();
  void build_trace_well_coupling_maps();

  std::string output_root_path() const;
  std::string output_prefix_path() const;
  void write_final_mesh_with_properties() const;

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

  ConditionalOStream pcout;
  TimerOutput computing_timer;
  npsat_flow::TimeStepTracker time_tracking;
  unsigned int my_rank; ///< MPI rank of this process.
  unsigned int n_proc; ///< Total number of MPI ranks.


};

template <int dim>
std::string NPSAT_FLOW<dim>::output_root_path() const
{
  return npsat_flow::resolve_relative_path(uo.main_path, uo.output_path);
}

template <int dim>
std::string NPSAT_FLOW<dim>::output_prefix_path() const
{
  return npsat_flow::join_paths(output_root_path(), uo.output_prefix);
}


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

template <int dim>
void NPSAT_FLOW<dim>::run() {
  std::cout << "I'm rank " << my_rank << " out of " << n_proc << std::endl;
  std::cout << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;

  set_simulation_data();
  if (uo.print_mesh_exit)
  {
    pcout << "Output.Print_mesh_exit enabled: exiting after initial mesh output." << std::endl;
    return;
  }

  setup_system();

}












int main(int argc, char **argv)
{
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

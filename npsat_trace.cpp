
#include <iostream>
#include <unordered_map>

#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/conditional_ostream.h>


#include "npsat_trace/trace_structures.h"
#include "npsat_trace/trace_help_func.h"
#include "npsat_trace/trace_input.h"
#include "npsat_trace/rt0_face_map.h"
#include "npsat_trace/cached_velocity.h"
#include "npsat_trace/particle_reader.h"



// namespace for general use:
using namespace dealii;

template <int dim>
class NPSAT_TRACE {
public:
  NPSAT_TRACE(const npsat_trace::Trace_options &topt_in);
  void run();
private:
  void load_triangulation();
  void read_parallel_coarse_tria_from_files();
  void setup_triangulation_helpers();
  void setup_system();
  void setup_particles();
  void load_velocity_io_mapping();
  void read_cell_well_map_binary_once();


  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;

  const FE_RaviartThomas<dim> fe_flux;
  DoFHandler<dim> dof_handler_flux;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  MappingQ1<dim> mapping;
  Particles::ParticleHandler<dim> particle_handler;

  TrilinosWrappers::MPI::Vector vface;

  npsat_trace::RT0FaceMap<dim> rt0_map;
  std::unordered_map<std::string, unsigned int> cellid_to_slot;
  std::vector<CellId> slot_cellid;
  std::vector<npsat_trace::CellVelocityCacheRT0Split3D<dim>> all_cells_cache;
  std::vector<bool> all_cells_cache_valid;
  std::vector<std::vector<npsat_trace::CellWellLink>> slot_cell_well_links;
  std::vector<double> slot_water_table_elevation;

  npsat_trace::Trace_options topt;
  ConditionalOStream pcout;
  unsigned int my_rank;
  unsigned int n_proc;

  std::vector<std::vector<BoundingBox<dim>>> global_bounding_boxes;
};

template <int dim>
NPSAT_TRACE<dim>::NPSAT_TRACE(const npsat_trace::Trace_options &topt_in)
  :
  mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement))
  , fe_flux(0)
  , dof_handler_flux(triangulation)
  , mapping()
  , particle_handler()
  , topt(topt_in)
  , pcout(std::cout,
             (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))


{
  my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
}

#include "npsat_trace/main_class_impl/npsat_trace_load.impl.h"


template <int dim>
void NPSAT_TRACE<dim>::run() {
  std::cout << "I'm rank " << my_rank << " out of " << n_proc << std::endl;

  load_triangulation();
  if (topt.exit_after_load_tria)
    return;

  MPI_Barrier(mpi_communicator);
  setup_triangulation_helpers();
  MPI_Barrier(mpi_communicator);
  setup_system();
  MPI_Barrier(mpi_communicator);
  setup_particles();

  const std::vector<double> delta_time_values = npsat_trace::read_delta_time_file(topt.delta_time_file);
  const unsigned int N_time_steps = static_cast<unsigned int>(delta_time_values.size());


}

template <int dim>
void NPSAT_TRACE<dim>::setup_particles() {
  particle_handler.clear();

  // Attach particle handler to the current triangulation and mapping.
  // This sets up internal data structures used for locating/moving particles.
  particle_handler.initialize(triangulation, mapping, npsat_trace::n_particle_props);

  // Optional but useful diagnostics
  pcout << "ParticleHandler initialized. Locally owned active cells: "
        << triangulation.n_locally_owned_active_cells() << std::endl;

  npsat_trace::ParticleReader reader(topt.n_paticles_parallel);
  if (my_rank == 0)
    reader.open(topt.particles_file);
  int iter = 0;

}


int main(int argc, char **argv)
{
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,1);
    npsat_trace::InputHandler input_handler;
    if (!input_handler.read_ini(argc, argv))
      return 0;

    NPSAT_TRACE<3> npsat_trace(input_handler.tr_opt);
    npsat_trace.run();
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

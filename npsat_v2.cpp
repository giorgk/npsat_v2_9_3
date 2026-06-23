


#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include  <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/conditional_ostream.h>
#include  <deal.II/base/timer.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>

#include "npsat_flow/flow_input.h"

using namespace dealii;


template <int dim>
class NPSAT_FLOW {
public:
  NPSAT_FLOW(const unsigned int degree,
             const npsat_flow::user_options &uo_in);

  void run();

private:
  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  const unsigned int degree;

  const npsat_flow::user_options uo;

  ConditionalOStream pcout;
  TimerOutput computing_timer;
  unsigned int my_rank; ///< MPI rank of this process.
  unsigned int n_proc; ///< Total number of MPI ranks.
};


template <int dim>
NPSAT_FLOW<dim>::NPSAT_FLOW(const unsigned int degree, const npsat_flow::user_options &uo_in)
  :
  mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement))
  , degree(degree)
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

template <int dim>
void NPSAT_FLOW<dim>::run() {
  std::cout << "I'm rank " << my_rank << " out of " << n_proc << std::endl;
  std::cout << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;
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

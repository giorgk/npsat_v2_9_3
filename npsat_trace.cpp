

#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/base/conditional_ostream.h>

#include <iostream>


// namespace for general use:
using namespace dealii;

template <int dim>
class NPSAT_TRACE {
public:
  NPSAT_TRACE();
  void run();
private:
  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;

  ConditionalOStream pcout;
  unsigned int my_rank;
  unsigned int n_proc;
};

template <int dim>
NPSAT_TRACE<dim>::NPSAT_TRACE()
  :
  mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement))
  , pcout(std::cout,
             (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))


{
  my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
}

template <int dim>
void NPSAT_TRACE<dim>::run() {
  std::cout << "I'm rank " << my_rank << " out of " << n_proc << std::endl;
  std::cout << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;
}


int main(int argc, char **argv)
{
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,1);
    NPSAT_TRACE<3> npsat_trace;
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

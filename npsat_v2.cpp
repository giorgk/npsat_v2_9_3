

// @sect3{Include files}

// The most fundamental class in the library is the Triangulation class, which
// is declared here:
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
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>

// The final step in importing deal.II is this: All deal.II functions and
// classes are in a namespace <code>dealii</code>, to make sure they don't
// clash with symbols from other libraries you may want to use in conjunction
// with deal.II. One could use these functions and classes by prefixing every
// use of these names by <code>dealii::</code>, but that would quickly become
// cumbersome and annoying. Rather, we simply import the entire deal.II
// namespace for general use:
using namespace dealii;


template <int dim>
class NPSAT_FLOW {
public:
  NPSAT_FLOW(const unsigned int degree);

  void run();

private:
  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  const unsigned int degree;

  ConditionalOStream pcout;
  TimerOutput computing_timer;
  unsigned int my_rank; ///< MPI rank of this process.
  unsigned int n_proc; ///< Total number of MPI ranks.
};


template <int dim>
NPSAT_FLOW<dim>::NPSAT_FLOW(const unsigned int degree)
  :
  mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement))
  , degree(degree)
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
    NPSAT_FLOW<3> npsat_flow(0);
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

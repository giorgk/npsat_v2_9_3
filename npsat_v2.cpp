


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
#include "npsat_flow/time_step_tracking.h"
#include "npsat_flow/interpolation/interpolation_function.h"
#include "npsat_flow/interpolation/interp_interface.h"
#include "npsat_flow/hydrogeo_prop.h"
#include "npsat_flow/streams.h"
#include "npsat_flow/mnwells.h"

using namespace dealii;


template <int dim>
class NPSAT_FLOW {
public:
  NPSAT_FLOW(const unsigned int degree,
             const npsat_flow::user_options &uo_in);

  void run();

private:
  void set_simulation_data();

  MPI_Comm mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  const unsigned int degree;

  const npsat_flow::user_options uo;

  npsat_flow::InterpolationFunction<dim> gw_recharge;
  npsat_flow::HydraulicProperties<dim> hgeo_prop;
  npsat_flow::StreamCollection<dim> streams;
  npsat_flow::MNWellCollection mnwells;

  ConditionalOStream pcout;
  TimerOutput computing_timer;
  npsat_flow::TimeStepTracker time_tracking;
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

  set_simulation_data();

}

template<int dim>
void NPSAT_FLOW<dim>::set_simulation_data() {
  const std::string input_root = npsat_flow::join_paths(uo.main_path, uo.input_path);
  const std::string output_root = npsat_flow::resolve_relative_path(uo.main_path, uo.output_path);

  AssertThrow(!output_root.empty(),
               ExcMessage("Paths.Output must resolve to a non-empty output directory."));
  AssertThrow(npsat_flow::path_exists(output_root),
              ExcMessage("Output directory does not exist: " + output_root));
  AssertThrow(npsat_flow::path_is_directory(output_root),
              ExcMessage("Output path is not a directory: " + output_root));

  {// Time step
    time_tracking.read_delta_time_file(npsat_flow::resolve_relative_path(input_root, uo.sim_opt.delta_time_file));
    time_tracking.initialize(uo.sim_opt.n_steps,uo.sim_opt.Start_step);
  }

  {// Set up recharge
    auto rch_interp = std::make_shared<npsat_flow::InterpInterface<dim>>();
    if (!uo.sources.rch_file.empty())
    {
      rch_interp->read_master_file(uo.sources.rch_file,
                                   uo.sources.rch_factor,
                                   mpi_communicator,
                                   input_root);
    }
    else
    {
      pcout << "Recharge data disabled: Sources.Rch_Data is empty." << std::endl;
    }
    gw_recharge.set_interpolant(rch_interp);
  }

  {// Set up hydrogeology data
    hgeo_prop.read(uo, mpi_communicator);
  }

  {
    std::cout << "Recharge: " << gw_recharge.value(Point<dim>(2500,2500,0)) << std::endl;
    std::cout << "Kx: " << hgeo_prop.conductivity(Point<dim>(2500,2500,0)) << std::endl;
    std::cout << "Ss: " << hgeo_prop.specific_storage(Point<dim>(2500,2500,0)) << std::endl;
    std::cout << "Sy: " << hgeo_prop.specific_yield(Point<dim>(2500,2500,0)) << std::endl;

  }

  {// Streams
    streams.stream_multiplier = uo.sources.stream_factor;
    if (!uo.sources.stream_file.empty() && !uo.sources.stream_rates_file.empty())
    {
      streams.read_streams(npsat_flow::resolve_relative_path(input_root, uo.sources.stream_file),
                           npsat_flow::resolve_relative_path(input_root, uo.sources.stream_rates_file),
                           mpi_communicator);
    }
    else
    {
      streams.clear();
      pcout << "Stream data disabled: Sources.Stream_Data or Sources.Stream_Rates is empty." << std::endl;
    }
  }

  {// Wells
    if (!uo.sources.well_file.empty())
    {
      npsat_flow::rank0_read_wells_distributes(npsat_flow::resolve_relative_path(input_root, uo.sources.well_file),
                                             mnwells,
                                             mpi_communicator);
      if (!uo.sources.well_rates_file.empty())
      {
        mnwells.Q_ts.read_data(npsat_flow::resolve_relative_path(input_root, uo.sources.well_rates_file),
                               mpi_communicator);
      }
      else
      {
        std::vector<double> zero_rates(mnwells.wells.size(), 0.0);
        mnwells.Q_ts.set_data_owned(std::move(zero_rates),
                                    static_cast<std::int64_t>(mnwells.wells.size()),
                                    1);
        pcout << "Well pumping disabled: Sources.Well_Rates is empty." << std::endl;
      }
    }
    else
    {
      mnwells.wells.clear();
      mnwells.well_rtree.clear();
      mnwells.Q_ts.set_data_owned(std::vector<double>(), 0, 0);
      pcout << "Wells disabled: Sources.Well_Data is empty." << std::endl;
    }
    pcout << "wells loaded: " << mnwells.wells.size() << std::endl;
    std::cout << "Rank " << my_rank << " has " << mnwells.wells.size() << std::endl;
  }







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

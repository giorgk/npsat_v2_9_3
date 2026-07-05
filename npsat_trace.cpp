
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
#include "npsat_trace/reader_helpers.h"
#include "npsat_trace/streamline_merge.h"



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
  void distribute_particles(const std::vector<npsat_trace::ParticleSeed> &seeds0);
  void load_data_step(const std::string & file_prefix, unsigned int step);
  void load_vface_rt0_values_step(const std::string &prefix, const unsigned int step_no);
  void read_particle_well_flows_for_step(const std::string &prefix, unsigned int step);
  void read_water_table_for_step(const std::string &prefix, unsigned int step);
  npsat_trace::CellVelocityCacheRT0Split3D<dim> & get_or_build_cell_cache(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    std::ofstream &dbg_cell_list);
  npsat_trace::WellBoreTraceResults<dim> well_bore_flow_trace(const typename DoFHandler<dim>::active_cell_iterator &current_cell, const Point<dim> &x_in) const;


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
  std::unordered_map<npsat_trace::FlowKey, npsat_trace::WellFlowRecord, npsat_trace::FlowKeyHash> flows_by_cell_well;


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
#include "npsat_trace/main_class_impl/npsat_trace_main.impl.h"


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

  npsat_trace::ParticleReader reader(topt.n_paticles_parallel);
  if (my_rank == 0)
    reader.open(topt.particles_file);
  int iter = 0;

  while (true) {
    // ---------------------------
    // (A) READ NEXT CHUNK (rank 0) + DISTRIBUTE
    // ---------------------------
    std::vector<npsat_trace::ParticleSeed> seeds0;
    if (my_rank == 0)
      seeds0 = reader.read_next_chunk();

    // global stop condition (rank 0 reached EOF and returned empty)
    int have = (my_rank == 0 && !seeds0.empty()) ? 1 : 0;
    have = Utilities::MPI::max(have, mpi_communicator);
    if (!have) break;

    distribute_particles(seeds0);


    // ------------------------------------------------------------
    // Prepare per-rank output file for this iter
    // ------------------------------------------------------------
    const std::string rank_str = Utilities::int_to_string(my_rank, 4);
    const std::string iter_str = Utilities::int_to_string(iter, 4);
    const std::string output_base = topt.output_prefix + "_streamlines_rank_" + rank_str + "_iter_" + iter_str;
    npsat_trace::StreamlineOutput sl_out(output_base, topt.write_ascii, topt.write_bin);

    // ---------------------------
    // (B) LOOP OVER TIME STEPS
    // ---------------------------
    unsigned int step = (topt.sim_opt.direction >= 0.0) ? 0u : (N_time_steps - 1u);
    while (true) {
      load_data_step(topt.input_prefix, step);
      // read the size of this timestep
      const double time_step_size = delta_time_values[step];

      // Initialize per-step remaining time for particles that are alive at step start
      // Set for all particles the time step equal to the size of this time step.
      // All the particles at this point should be active. If they have been terminated
      // they should be removed from the particle_handler
      for (auto p = particle_handler.begin(); p != particle_handler.end(); ++p)
      {
        auto props = p->get_properties();
        props[npsat_trace::pState] = 1.0;
        props[npsat_trace::pDtRemaining] = time_step_size;
      }

      // ------------------------------------------------------------
      // Particle loop within this time step
      // Repeat: trace locally -> exchange ghost -> until step done globally
      // ------------------------------------------------------------
      unsigned int exchange_iter = 0;

      while (true) {
        // --------------------------------------------------------
        // 1) TRACE LOCALLY UNTIL (step done) OR (hit ghost) OR (terminate)
        //    Collect ghost-hit particles into send buffers, remove locally.
        // --------------------------------------------------------
        // Count particles that still have dt remaining on this rank.
        // (This is what controls step convergence.)
        unsigned int not_done_local = 0;

        // --------------------------------------------------------
        // 1) TRACE LOCALLY
        //    - Trace only when current cell is locally owned
        //    - Stop tracing when moved to non-owned cell (ghost/foreign)
        //    - Remove particle if it leaves domain (safe erase pattern)
        // --------------------------------------------------------
        auto particle = particle_handler.begin();
        while (particle != particle_handler.end()) {
          auto props = particle->get_properties();
          // Skip the particles that have done with this step
          if (props[npsat_trace::pState] == 2.0 || props[npsat_trace::pState] == -1.0) {
            ++particle;
            continue;
          }
          // Locate surrounding cell
          const typename Triangulation<3, 3>::cell_iterator tria_cell = particle->get_surrounding_cell(triangulation);
          //auto current_cell = tria_cell->as_dof_handler_iterator(dof_handler_flux);
          typename DoFHandler<dim>::active_cell_iterator current_cell(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_flux);

          // This should be highly unlikely
          // If not locally owned, do not trace on this rank.
          // It will be migrated by sort_particles_into_subdomains_and_cells().
          if (!current_cell->is_locally_owned()) {
            // This particle is not done for the step (it still has dt_remaining),
            // but we cannot progress it here.
            if (props[npsat_trace::pDtRemaining] > topt.sim_opt.dt_eps)
              ++not_done_local;

            ++particle;
            continue;
          }

          double &dt_remaining = props[npsat_trace::pDtRemaining];
          if (dt_remaining <= topt.sim_opt.dt_eps){
            // This particle already finished this step on this rank
            props[npsat_trace::pState] = 2.0;
            ++particle;
            continue;
          }

          std::ofstream dbg_out;
          std::ofstream dbg_cell_list;
          if (topt.misc_opt.particle_traj_dbg){
            const std::string f_dbg_name = topt.misc_opt.dbg_prefix + "_dbg_particles_rank_" + rank_str + ".dat";
            dbg_out.open(f_dbg_name, std::ios::trunc);

            const std::string f_dbg_cell_list_name = topt.misc_opt.dbg_prefix + "_cell_list_rank_" + rank_str + ".dat";
            dbg_cell_list.open(f_dbg_cell_list_name, std::ios::trunc);
          }

          // -----------------------------
          // Inner tracing loop
          // -----------------------------
          // This particle still has to walk this step

          const double Eid = props[npsat_trace::pEid];
          const double Sid = props[npsat_trace::pSid];
          //std::cout << "Particle Eid: " << Eid << ", Sid: " << Sid << std::endl;

          double &streamline_steps = props[npsat_trace::pStreamlineSteps];
          double &particle_age = props[npsat_trace::pAge];
          const int max_nonexpanding_steps = topt.sim_opt.n_max_nonexpanding_steps;
          const double stagnant_velocity_threshold = topt.sim_opt.stagnant_velocity_threshold;

          Point<dim> x_ref; // reference point
          double vmag = 0.0;

          // Trace this particle until walks for time_step_size or hits ghost or dead end
          while (dt_remaining > topt.sim_opt.dt_eps) {

            if (streamline_steps >= topt.sim_opt.n_max_streamline_steps) {
              npsat_trace::write_termination(sl_out,props[npsat_trace::pPid], Eid, Sid, npsat_trace::MAX_ITER);
              props[npsat_trace::pState] = -1.0;
              dt_remaining = 0.0;
              break;
            }

            if (topt.sim_opt.max_age >= 0.0) {
              if (particle_age >= topt.sim_opt.max_age) {
                            npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, npsat_trace::MAX_AGE);
                            props[npsat_trace::pState] = -1.0;
                            dt_remaining = 0.0;
                            break;
              }
            }

            streamline_steps += 1.0;

            auto &cached_cell = get_or_build_cell_cache(current_cell, dbg_cell_list);

            const Point<dim> x = particle->get_location();

            // --- WELL-BORE ROUTING (before velocity eval) ---
            {
              const auto wb = this->well_bore_flow_trace(current_cell, x);
              if (wb.terminate) {
                // Keep wb.new_pos even if outside (per your rule) and terminate tracking
                particle->set_location(wb.new_pos);
                npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, wb.end_reason);


                props[npsat_trace::pState] = -1.0;   // mark for deletion
                dt_remaining = 0.0;
                break;
              }

              // If the well moved the particle (position and/or cell), apply it and restart loop iteration
              // so caches/velocity are computed in the correct cell at the new position.
              if (wb.new_cell != current_cell || wb.new_pos != x) {
                particle->set_location(wb.new_pos);
                current_cell = wb.new_cell;

                // If we jumped into a ghost cell, stop local tracing (migration will handle it)
                if (!current_cell->is_locally_owned()){
                  if (dt_remaining < topt.sim_opt.dt_eps)
                    props[npsat_trace::pState] = 2.0;
                  break;
                }
                // Continue to next inner-iteration with updated cell/position.
                // (Avoid using cached_cell from the old cell.)
                continue;
              }
            }

            Tensor<1,dim> u;

            // when we arrive here we have make sure that the point is inside this cell.
            // so we can safely use the following method to get reference coordinates.
            cached_cell.get_clamped_ref_coords(x, x_ref);

            // Calculate the velocity
            cached_cell.compute_velocity_at_particle(x_ref,u,vmag);

            if (topt.misc_opt.particle_traj_dbg)
              dbg_out << x[0] << " " << x[1] << " " << x[2] << " " << u[0] << " " << u[1] << " " << u[2] << std::endl;

            u = u/topt.sim_opt.porosity;
            vmag = vmag/topt.sim_opt.porosity;

            // Write this step
            props[npsat_trace::pVmag] = vmag;
            // Write current location ONLY if velocity is computed
            npsat_trace::write_sample(sl_out, props[npsat_trace::pPid], Eid, Sid, x, vmag);

            // A particle can sit in a stagnant/near-stagnant cell for one
            // transient step and move again later when the flow field changes.
            // Treat this as completed for the current time step; the
            // non-expanding trajectory guard below handles persistent stagnation.
            if (vmag <= stagnant_velocity_threshold) {
              npsat_trace::increment_streamline_nonexpansion(props);
              if (max_nonexpanding_steps > 0 && props[npsat_trace::pNoExpandCount] >= static_cast<double>(max_nonexpanding_steps))
              {
                npsat_trace::write_termination(sl_out,props[npsat_trace::pPid], Eid, Sid, npsat_trace::er_nonexpanding);
                props[npsat_trace::pState] = -1.0;
              }
              else
                props[npsat_trace::pState] = 2.0;
              particle_age += dt_remaining;
              dt_remaining = 0.0;
              break;
            }

            // Calculate the step size using the current velocity
            bool is_stuck = false;
            // find the step size
            npsat_trace::TimeStepControl tsc; //TODO make this user parameter
            const double ds = npsat_trace::compute_step_size_ds<dim>(current_cell->diameter(),
                                                        vmag, dt_remaining, time_step_size, tsc,is_stuck);

            if (is_stuck) {
              npsat_trace::increment_streamline_nonexpansion(props);
              if (max_nonexpanding_steps > 0 &&
                  props[npsat_trace::pNoExpandCount] >= static_cast<double>(max_nonexpanding_steps))
              {
                npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, npsat_trace::er_nonexpanding);
                props[npsat_trace::pState] = -1.0;
              }
              else
                props[npsat_trace::pState] = 2.0;
              particle_age += dt_remaining;
              dt_remaining = 0.0;
              break;
            }

            // move the particle
            const Tensor<1,dim> dir = u / vmag;
            const Point<dim> x_proposed = x + ds * dir * topt.sim_opt.direction;
            const double dt_move = ds / vmag;

            if (topt.sim_opt.max_age >= 0.0 && particle_age + dt_move > topt.sim_opt.max_age) {
              const double alpha_age = npsat_trace::clamp_((topt.sim_opt.max_age - particle_age) / dt_move, 0.0, 1.0);
              const Point<dim> x_age = x + alpha_age * (x_proposed - x);
              particle->set_location(x_age);
              npsat_trace::update_streamline_bbox<dim>(props, x_age);
              particle_age = topt.sim_opt.max_age;
              npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, npsat_trace::MAX_AGE);
              props[npsat_trace::pState] = -1.0;
              dt_remaining = 0.0;
              break;
            }

            {
              const unsigned int slot = static_cast<unsigned int>(current_cell->user_index());
              if (slot < slot_water_table_elevation.size()) {
                const double wt_z = slot_water_table_elevation[slot];
                if (std::isfinite(wt_z)) {
                  constexpr double wt_eps = 1e-10;
                  const double z0 = x[dim-1] - wt_z;
                  const double z1 = x_proposed[dim-1] - wt_z;

                  if (z0 > wt_eps || (z0 <= wt_eps && z1 >= -wt_eps && z1 != z0)) {
                    double alpha = 0.0;
                    if (std::abs(z1 - z0) > wt_eps)
                      alpha = npsat_trace::clamp_(-z0 / (z1 - z0), 0.0, 1.0);

                    const Point<dim> x_wt = x + alpha * (x_proposed - x);
                    particle->set_location(x_wt);
                    npsat_trace::update_streamline_bbox<dim>(props, x_wt);
                    particle_age += alpha * dt_move;
                    npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, npsat_trace::er_water_table);
                    props[npsat_trace::pState] = -1.0;
                    dt_remaining = 0.0;
                    break;
                  }
                }
              }
            }

            // Set the new location for the particle
            particle->set_location(x_proposed);
            npsat_trace::update_streamline_bbox<dim>(props, x_proposed);
            // In the last step for this time step this should be configured so as to return dt_move ~0
            dt_remaining -= dt_move;
            particle_age += dt_move;
            props[npsat_trace::pPid] += 1.0;

            // This checks if the x_proposed is in the current cell.
            // If it's not in search and returns the cell where the particle is currently
            // However this cell can be locally owned or ghosted
            bool cell_found = npsat_trace::check_cell_point<dim>(current_cell, x_proposed);

            if (cell_found) {
              // If it ended up in a ghost (or non-owned) cell, stop tracing here.
              // Migration happens after we finish this local loop.
              if (!current_cell->is_locally_owned()) {
                // There is no need to update any status here.
                // Once all particles are completed within this inner loop
                // the particle will tranfsred to the correct processor where
                // it wont be ghosted. In addition the dt_remaining will be carried over
                if (dt_remaining < topt.sim_opt.dt_eps) {
                  // If the last step entered a ghost element
                  // we should call the walk completed.
                  // we need to transfered this though but deal will take care of that
                  props[npsat_trace::pState] = 2.0;
                }
                break;
              }
            }
            else {
              // If the cell has not been found then most likely it is outside of the domain.
              // The only way this can be inside the domain is if it has skipped at
              // least one lever of elements. In that case we should decrease the step size.
              // We will search to find out which face has exited.
              npsat_trace::FindHexExitResult er = npsat_trace::find_hex_exit(x, x_proposed,
                cached_cell.get_xv(), cached_cell.get_yv(), cached_cell.get_zb(), cached_cell.get_zt());
              if (er.found) {
                int end_reason = npsat_trace::er_exited_domain;
                const int exit_face = er.value.face_index;
                if (exit_face >= 0 && exit_face <= 3) {
                  end_reason = npsat_trace::er_lateral;
                }
                else if (exit_face == 4 && current_cell->at_boundary(4)) {
                  end_reason = npsat_trace::er_bottom;
                }
                else if (exit_face == 5 && current_cell->at_boundary(5)) {
                  end_reason = npsat_trace::er_water_table;
                }
                npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, end_reason);
              }
              else {
                npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, -9); //TODO fix all the key values
              }
              props[npsat_trace::pState] = -1.0;   // mark for deletion
              dt_remaining = 0.0;
              break;
            }

            if (max_nonexpanding_steps > 0 && props[npsat_trace::pNoExpandCount] >= static_cast<double>(max_nonexpanding_steps))
            {
              npsat_trace::write_termination(sl_out, props[npsat_trace::pPid], Eid, Sid, npsat_trace::er_nonexpanding);
              props[npsat_trace::pState] = -1.0;
              dt_remaining = 0.0;
              break;
            }

            if (dt_remaining < topt.sim_opt.dt_eps)
            {
              //since this is days, a step of less than a day is too much should be treated as zero
              props[npsat_trace::pState] = 2.0; // The particle completed its walk for this time step
              break;
            }
          }// while loop tracing a particle within a time step
          if (props[npsat_trace::pState] == 1.0 &&
              props[npsat_trace::pDtRemaining] > topt.sim_opt.dt_eps)
            ++not_done_local;
          ++particle;
        }// While loop iterating particles

        // Clean up removed particles
        bool removed_any = true;
        while (removed_any) {
          removed_any = false;

          for (auto p = particle_handler.begin(); p != particle_handler.end(); ++p) {
            auto props = p->get_properties();
            if (props[npsat_trace::pState] == -1.0) {
              particle_handler.remove_particle(p);
              removed_any = true;
              break; // restart loop because iterators/accessors may be invalid now
            }
          }
        }

        // --------------------------------------------------------
        // 2) Global check: is THIS step finished for all particles?
        // --------------------------------------------------------
        const unsigned int not_done_global = Utilities::MPI::sum(not_done_local, mpi_communicator);
        if (not_done_global == 0)
        {
          break;
        }

        // --------------------------------------------------------
        // 3) Exchange/migrate particles that moved out of subdomain
        // --------------------------------------------------------
        particle_handler.sort_particles_into_subdomains_and_cells();

        AssertThrow(++exchange_iter < static_cast<unsigned int>(topt.sim_opt.n_max_proc_exchanges),
            ExcMessage("Exceeded max exchange iterations in step " +
                       std::to_string(step) + "."));

      }// While loop until all particles terminate within this step

      // ------------------------------------------------------------
      // Optional: If no particles remain globally, stop time cycling
      // ------------------------------------------------------------
      const unsigned int n_local  = particle_handler.n_locally_owned_particles();
      const unsigned int n_global = Utilities::MPI::sum(n_local, mpi_communicator);
      if (n_global == 0) {
        break;
      }
      // Advance time step cyclically.
      if (topt.sim_opt.direction >= 0.0)
        step = (step + 1) % N_time_steps;
      else
        step = (step + N_time_steps - 1) % N_time_steps;
    }// while loop over time steps
    sl_out.close();
    npsat_trace::merge_streamline_iteration(topt.output_prefix,
                                            static_cast<unsigned int>(iter),
                                            n_proc,
                                            my_rank,
                                            topt.write_ascii,
                                            topt.write_bin,
                                            mpi_communicator);
    iter++;
  } // End of main while loop
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

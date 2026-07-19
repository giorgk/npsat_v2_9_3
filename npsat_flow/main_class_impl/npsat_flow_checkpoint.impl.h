#ifndef NPSAT_FLOW_CHECKPOINT_IMPL_H
#define NPSAT_FLOW_CHECKPOINT_IMPL_H

template <int dim>
std::string NPSAT_FLOW<dim>::checkpoint_base_path() const
{
  const std::string checkpoint_root =
      npsat_flow::trim(uo.checkpoint_folder).empty()
          ? output_root_path()
          : npsat_flow::resolve_relative_path(uo.main_path,
                                               uo.checkpoint_folder);
  return npsat_flow::resolve_relative_path(checkpoint_root,
                                           uo.sim_opt.checkpoint_file);
}

template <int dim>
void NPSAT_FLOW<dim>::save_checkpoint(const unsigned int completed_spinup_iterations)
{
  const std::uint64_t magic = static_cast<std::uint64_t>(0x4e5053415443484bULL); // "NPSATCHK"
  const std::uint32_t format_version = 2;
  const unsigned int new_slot = 1u - checkpoint_slot;
  const std::string base = checkpoint_base_path();
  const CheckpointPhase phase =
      (completed_spinup_iterations > 0 ? checkpoint_phase_spinup :
       (time_tracking.done() ? checkpoint_phase_finished : checkpoint_phase_simulation));

  std::ostringstream rank_name;
  rank_name << base << ".slot" << new_slot << ".rank"
            << std::setw(6) << std::setfill('0') << my_rank;
  const std::string final_name = rank_name.str();
  const std::string temp_name = final_name + ".tmp";

  int local_ok = 1;
  {
    std::ofstream out(temp_name.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.good())
      local_ok = 0;
    else
    {
      const std::uint32_t dim_value = dim;
      const std::uint32_t degree_value = degree;
      const std::uint32_t nproc_value = n_proc;
      const std::uint32_t rank_value = my_rank;
      const std::uint32_t phase_value = static_cast<std::uint32_t>(phase);
      const std::uint64_t global_dofs = dof_handler_head.n_dofs();
      const std::uint64_t local_dofs = head_locally_owned_dofs.n_elements();
      const std::uint32_t run_step = time_tracking.simulation_step();
      const std::uint32_t repeat_step = completed_spinup_iterations;

      out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
      out.write(reinterpret_cast<const char *>(&format_version), sizeof(format_version));
      out.write(reinterpret_cast<const char *>(&dim_value), sizeof(dim_value));
      out.write(reinterpret_cast<const char *>(&degree_value), sizeof(degree_value));
      out.write(reinterpret_cast<const char *>(&nproc_value), sizeof(nproc_value));
      out.write(reinterpret_cast<const char *>(&rank_value), sizeof(rank_value));
      out.write(reinterpret_cast<const char *>(&phase_value), sizeof(phase_value));
      out.write(reinterpret_cast<const char *>(&global_dofs), sizeof(global_dofs));
      out.write(reinterpret_cast<const char *>(&local_dofs), sizeof(local_dofs));
      out.write(reinterpret_cast<const char *>(&run_step), sizeof(run_step));
      out.write(reinterpret_cast<const char *>(&repeat_step), sizeof(repeat_step));

      for (auto it = head_locally_owned_dofs.begin();
           it != head_locally_owned_dofs.end(); ++it)
      {
        const std::uint64_t gid = *it;
        const double value = h_old[*it];
        out.write(reinterpret_cast<const char *>(&gid), sizeof(gid));
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
      out.close();
      if (!out.good())
        local_ok = 0;
    }
  }

  int all_ok = 0;
  MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_communicator);
  AssertThrow(all_ok == 1, ExcMessage("Failed to write checkpoint rank file: " + temp_name));

  std::remove(final_name.c_str());
  local_ok = (std::rename(temp_name.c_str(), final_name.c_str()) == 0) ? 1 : 0;
  MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_communicator);
  AssertThrow(all_ok == 1, ExcMessage("Failed to commit checkpoint rank file: " + final_name));
  MPI_Barrier(mpi_communicator);

  if (my_rank == 0)
  {
    const std::string meta_name = base + ".meta";
    const std::string temp_meta = meta_name + ".tmp";
    const std::string backup_meta = meta_name + ".bak";
    std::ofstream meta(temp_meta.c_str(), std::ios::out | std::ios::trunc);
    if (meta.good())
    {
      meta << "NPSAT_CHECKPOINT 2\n"
           << new_slot << ' ' << static_cast<unsigned int>(phase) << ' '
           << n_proc << ' ' << dof_handler_head.n_dofs() << ' '
           << dim << ' ' << degree << ' ' << uo.sim_opt.Start_step << ' '
           << uo.sim_opt.n_steps << ' ' << time_tracking.simulation_step() << ' '
           << completed_spinup_iterations << '\n';
      meta.close();
    }
    local_ok = meta.good() ? 1 : 0;
    if (local_ok)
    {
      std::remove(backup_meta.c_str());
      std::rename(meta_name.c_str(), backup_meta.c_str());
      if (std::rename(temp_meta.c_str(), meta_name.c_str()) != 0)
      {
        std::rename(backup_meta.c_str(), meta_name.c_str());
        local_ok = 0;
      }
      else
        std::remove(backup_meta.c_str());
    }
  }
  MPI_Bcast(&local_ok, 1, MPI_INT, 0, mpi_communicator);
  AssertThrow(local_ok == 1, ExcMessage("Failed to commit checkpoint metadata: " + base + ".meta"));
  checkpoint_slot = new_slot;
  if (phase == checkpoint_phase_spinup)
    pcout << "Spin-up checkpoint saved after iteration "
          << completed_spinup_iterations << std::endl;
  else if (phase == checkpoint_phase_finished)
    pcout << "Finished-simulation checkpoint saved at step "
          << time_tracking.simulation_step() << std::endl;
  else
    pcout << "Simulation checkpoint saved for next step "
          << time_tracking.simulation_step() << std::endl;
}

template <int dim>
unsigned int NPSAT_FLOW<dim>::load_checkpoint()
{
  const std::uint64_t expected_magic = static_cast<std::uint64_t>(0x4e5053415443484bULL);
  const std::string base = checkpoint_base_path();
  unsigned int slot = 0, saved_phase = 0, saved_nproc = 0, saved_dim = 0, saved_degree = 0;
  unsigned int saved_start = 0, saved_nsteps = 0, saved_run_step = 0, saved_repeat = 0;
  std::uint64_t saved_global_dofs = 0;
  int meta_ok = 1;
  int used_backup_metadata = 0;

  if (my_rank == 0)
  {
    std::ifstream meta((base + ".meta").c_str());
    if (!meta.good())
    {
      meta.clear();
      meta.open((base + ".meta.bak").c_str());
      used_backup_metadata = 1;
    }
    std::string label;
    unsigned int version = 0;
    if (!(meta >> label >> version) || label != "NPSAT_CHECKPOINT" || version != 2)
      meta_ok = 0;
    else
    {
      if (!(meta >> slot >> saved_phase >> saved_nproc >> saved_global_dofs
                 >> saved_dim >> saved_degree >> saved_start >> saved_nsteps
                 >> saved_run_step >> saved_repeat))
        meta_ok = 0;
    }
  }
  MPI_Bcast(&meta_ok, 1, MPI_INT, 0, mpi_communicator);
  AssertThrow(meta_ok == 1, ExcMessage("Cannot read a valid checkpoint metadata file: " + base + ".meta"));
  MPI_Bcast(&slot, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_phase, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_nproc, 1, MPI_UNSIGNED, 0, mpi_communicator);
  unsigned long long broadcast_global_dofs = static_cast<unsigned long long>(saved_global_dofs);
  MPI_Bcast(&broadcast_global_dofs, 1, MPI_UNSIGNED_LONG_LONG, 0, mpi_communicator);
  saved_global_dofs = static_cast<std::uint64_t>(broadcast_global_dofs);
  MPI_Bcast(&saved_dim, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_degree, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_start, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_nsteps, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_run_step, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&saved_repeat, 1, MPI_UNSIGNED, 0, mpi_communicator);
  MPI_Bcast(&used_backup_metadata, 1, MPI_INT, 0, mpi_communicator);

  const char *phase_name =
      (saved_phase == static_cast<unsigned int>(checkpoint_phase_spinup)
           ? "spin-up"
           : (saved_phase == static_cast<unsigned int>(checkpoint_phase_finished)
                  ? "finished"
                  : "simulation"));
  pcout << "Checkpoint metadata selected committed slot " << slot
        << " from " << base
        << (used_backup_metadata ? ".meta.bak" : ".meta")
        << "\n  phase = " << phase_name
        << ", simulation counter = " << saved_run_step
        << ", completed spin-up solves = " << saved_repeat
        << ", MPI ranks = " << saved_nproc << std::endl;

  AssertThrow(saved_nproc == n_proc, ExcMessage("Checkpoint requires the same MPI process count."));
  AssertThrow(saved_global_dofs == dof_handler_head.n_dofs(), ExcMessage("Checkpoint head DoF count does not match the current mesh."));
  AssertThrow(saved_dim == dim && saved_degree == degree, ExcMessage("Checkpoint dimension or finite-element degree does not match."));
  AssertThrow(saved_start == static_cast<unsigned int>(uo.sim_opt.Start_step), ExcMessage("Checkpoint Simulation.Start_step does not match."));
  AssertThrow(saved_phase <= static_cast<unsigned int>(checkpoint_phase_finished),
              ExcMessage("Checkpoint contains an invalid execution phase."));
  AssertThrow(saved_run_step <= static_cast<unsigned int>(uo.sim_opt.n_steps),
              ExcMessage("Checkpoint next simulation step exceeds the configured Simulation.Nsteps."));
  if (saved_phase == static_cast<unsigned int>(checkpoint_phase_spinup))
  {
    AssertThrow(uo.spin_uo.iterations > 0,
                ExcMessage("Cannot resume a spin-up checkpoint when Spinup.Iterations is zero."));
    AssertThrow(saved_run_step == 0 && saved_repeat > 0,
                ExcMessage("Spin-up checkpoint contains inconsistent counters."));
    AssertThrow(saved_repeat < uo.spin_uo.iterations,
                ExcMessage("Spinup.Iterations must be greater than the completed spin-up iteration stored in the checkpoint."));
  }
  else
  {
    AssertThrow(saved_repeat == 0,
                ExcMessage("Simulation checkpoint contains an unexpected spin-up iteration count."));
  }

  std::ostringstream rank_name;
  rank_name << base << ".slot" << slot << ".rank"
            << std::setw(6) << std::setfill('0') << my_rank;
  std::ifstream in(rank_name.str().c_str(), std::ios::in | std::ios::binary);
  AssertThrow(in.good(), ExcMessage("Cannot open checkpoint rank file: " + rank_name.str()));

  std::uint64_t magic = 0, global_dofs = 0, local_dofs = 0;
  std::uint32_t format_version = 0, file_dim = 0, file_degree = 0, file_phase = 0;
  std::uint32_t file_nproc = 0, file_rank = 0, file_run_step = 0, file_repeat = 0;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char *>(&format_version), sizeof(format_version));
  in.read(reinterpret_cast<char *>(&file_dim), sizeof(file_dim));
  in.read(reinterpret_cast<char *>(&file_degree), sizeof(file_degree));
  in.read(reinterpret_cast<char *>(&file_nproc), sizeof(file_nproc));
  in.read(reinterpret_cast<char *>(&file_rank), sizeof(file_rank));
  in.read(reinterpret_cast<char *>(&file_phase), sizeof(file_phase));
  in.read(reinterpret_cast<char *>(&global_dofs), sizeof(global_dofs));
  in.read(reinterpret_cast<char *>(&local_dofs), sizeof(local_dofs));
  in.read(reinterpret_cast<char *>(&file_run_step), sizeof(file_run_step));
  in.read(reinterpret_cast<char *>(&file_repeat), sizeof(file_repeat));
  AssertThrow(in.good() && magic == expected_magic && format_version == 2,
              ExcMessage("Invalid checkpoint rank-file header."));
  AssertThrow(file_dim == dim && file_degree == degree && file_nproc == n_proc &&
              file_rank == my_rank && global_dofs == dof_handler_head.n_dofs() &&
              local_dofs == head_locally_owned_dofs.n_elements() &&
              file_phase == saved_phase && file_run_step == saved_run_step &&
              file_repeat == saved_repeat,
              ExcMessage("Checkpoint rank-file metadata does not match."));

  h_old = 0.0;
  for (std::uint64_t i = 0; i < local_dofs; ++i)
  {
    std::uint64_t gid = 0;
    double value = 0.0;
    in.read(reinterpret_cast<char *>(&gid), sizeof(gid));
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    AssertThrow(in.good() && head_locally_owned_dofs.is_element(gid),
                ExcMessage("Checkpoint contains an invalid locally owned head DoF."));
    h_old[gid] = value;
  }
  h_old.compress(VectorOperation::insert);
  time_tracking.restore_simulation_step(saved_run_step);
  checkpoint_slot = slot;
  if (saved_phase == static_cast<unsigned int>(checkpoint_phase_spinup))
    pcout << "Spin-up checkpoint loaded; resuming at spin-up iteration "
          << (saved_repeat + 1) << " of at most "
          << uo.spin_uo.iterations << std::endl;
  else
    pcout << "Simulation checkpoint loaded; next simulation step "
          << saved_run_step << " of " << uo.sim_opt.n_steps
          << " (checkpoint was written with Nsteps=" << saved_nsteps << ")"
          << std::endl;
  return (saved_phase == static_cast<unsigned int>(checkpoint_phase_spinup) ?
          saved_repeat : 0u);
}

#endif

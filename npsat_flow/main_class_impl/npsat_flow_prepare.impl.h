//
// Created by giorgk on 6/24/26.
//

#ifndef NPSAT_FLOW_PREPARE_IMPL_H
#define NPSAT_FLOW_PREPARE_IMPL_H

//using namespace dealii;

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

  {// Dirichlet boundary conditions
    if (!uo.bndr_cond.dirichlet_file.empty()) {
      dirichlet_bc.set_lateral_matching_tolerances(uo.bndr_cond.half_with, uo.bndr_cond.min_overlap);

      dirichlet_bc.read_data(uo.bndr_cond.dirichlet_file,
                             1.0,
                             mpi_communicator,
                             17,
                             input_root);

      pcout << "Dirichlet boundary data loaded: "
            << dirichlet_bc.get_parts().size()
            << " entries"
            << std::endl;

    }
  }

  {// GHB boundary conditions
    if (!uo.bndr_cond.ghb_file.empty())
    {
      ghb_bc.set_lateral_matching_tolerances(uo.bndr_cond.half_with, uo.bndr_cond.min_overlap);

      ghb_bc.read_data(uo.bndr_cond.ghb_file,
                       1.0,  // head factor
                       1.0,  // conductance factor
                       mpi_communicator,
                       1000,
                       input_root);

      ghb_boundary_map = ghb_bc.get_function_map();

      pcout << "GHB boundary data loaded: "
            << ghb_bc.get_parts().size()
            << " entries"
            << std::endl;
    }
  }

  {//Create initial triangulation
    npsat_flow::GridBuilder<dim>::build(triangulation, uo, mpi_communicator);
  }

  if (uo.print_initial_mesh) {
    if (my_rank == 0)
    {
      const std::string filename = output_prefix_path() + "_initial_mesh.vtk";
      std::ofstream outfile(filename.c_str());
      AssertThrow(outfile.good(), ExcMessage("Could not open initial mesh VTK: " + filename));

      GridOut grid_out;
      grid_out.write_vtk(triangulation, outfile);
    }
  }

  refine_triangulation();

  initialize_local_cell_slots();

  {
    dirichlet_bc.assign_to_triangulation(triangulation, dirichlet_boundary_map, 1000);
    dirichlet_boundary_map = dirichlet_bc.get_function_map();

    ghb_bc.assign_to_triangulation(triangulation);
    ghb_boundary_map = ghb_bc.get_function_map();
  }

  if (uo.print_mesh_with_prop)
    write_final_mesh_with_properties();
}

template <int dim>
void NPSAT_FLOW<dim>::refine_triangulation()
{
  typename npsat_flow::GridBuilder<dim>::RefinementTargets targets;
  targets.mnwells = &mnwells;
  targets.streams = &streams;

  if (!dirichlet_bc.get_parts().empty())
  {
    targets.dirichlet = &dirichlet_bc;
  }

  if (!ghb_bc.get_parts().empty())
  {
    targets.ghb = &ghb_bc;
  }

  // Same pattern later for Neumann:
  //
  // if (!neumann_bc.get_parts().empty())
  // {
  //     targets.neumann_face =
  //         [this](const typename npsat_v2::GridBuilder<dim>::CellIterator &cell,
  //                const unsigned int face)
  //         {
  //             return neumann_bc.face_is_neumann(cell, face);
  //         };
  // }

  npsat_flow::GridBuilder<dim>::refine_triangulation(triangulation, uo, targets, mpi_communicator);
}

template <int dim>
void NPSAT_FLOW<dim>::initialize_local_cell_slots()
{
  triangulation.clear_user_data();

  const unsigned int n_local = triangulation.n_locally_owned_active_cells();

  unsigned int slot = 0;
  for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
  {
    if (!cell->is_locally_owned())
      continue;

    cell->set_user_index(slot);
    ++slot;
  }

  AssertThrow(slot == n_local, ExcInternalError());
  local_element_data_rt_0dg0.initialize(n_local);
}

template <int dim>
void NPSAT_FLOW<dim>::write_final_mesh_with_properties() const {
  FE_Q<dim> fe_nodal(1);
  DoFHandler<dim> dof_handler_nodal(triangulation);
  dof_handler_nodal.distribute_dofs(fe_nodal);

  const unsigned int n_dofs = dof_handler_nodal.n_dofs();

  Vector<double> recharge(n_dofs);
  Vector<double> kx(n_dofs);
  Vector<double> ky(n_dofs);
  Vector<double> kz(n_dofs);
  Vector<double> sy(n_dofs);
  Vector<double> ss(n_dofs);
  Vector<double> boundary_id(n_dofs);
  Vector<double> boundary_kind(n_dofs);
  Vector<double> is_boundary(n_dofs);
  Vector<double> is_recharge(n_dofs);
  Vector<double> is_dirichlet(n_dofs);
  Vector<double> dirichlet_head(n_dofs);
  Vector<double> is_ghb(n_dofs);
  Vector<double> ghb_head(n_dofs);
  Vector<double> ghb_conductance(n_dofs);
  Vector<double> proc_id(n_dofs);
  proc_id = -1.0;

  const double recharge_default_value = 0.0;
  const double recharge_default_tolerance = 1.0e-14;

  boundary_id = -1.0;
  boundary_kind = -1.0;

  std::vector<unsigned char> vertex_evaluated(n_dofs, 0);
  std::unordered_map<unsigned int, types::global_dof_index> vertex_to_dof;
  vertex_to_dof.reserve(n_dofs);

  auto set_boundary_kind = [&](const types::global_dof_index dof,
                                 const types::boundary_id bid,
                                 const double kind,
                                 const double priority) {
    is_boundary[dof] = 1.0;
    if (priority >= boundary_kind[dof])
    {
      boundary_kind[dof] = kind;
      boundary_id[dof] = static_cast<double>(bid);
    }
  };

  for (const auto &cell : dof_handler_nodal.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const types::global_dof_index dof = cell->vertex_dof_index(v, 0);
            vertex_to_dof[cell->vertex_index(v)] = dof;
            proc_id[dof] = static_cast<double>(my_rank);

            if (vertex_evaluated[dof] != 0)
                continue;

            const Point<dim> &p = cell->vertex(v);
            const Tensor<2, dim> K = hgeo_prop.conductivity(p);

            kx[dof] = K[0][0];
            ky[dof] = (dim >= 2 ? K[1][1] : K[0][0]);
            kz[dof] = (dim >= 3 ? K[2][2] : 0.0);
            sy[dof] = hgeo_prop.specific_yield(p);
            ss[dof] = hgeo_prop.specific_storage(p);

            vertex_evaluated[dof] = 1;
        }

        for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
        {
            if (!cell->face(face_no)->at_boundary())
                continue;

            const types::boundary_id bid = cell->face(face_no)->boundary_id();
            const bool face_can_receive_recharge = (face_no == 5);
            const auto dirichlet_it = dirichlet_boundary_map.find(bid);
            const auto ghb_it = ghb_boundary_map.find(bid);

            for (unsigned int fv = 0; fv < GeometryInfo<dim>::vertices_per_face; ++fv)
            {
                const Point<dim> vertex = cell->face(face_no)->vertex(fv);
                const auto dof_it = vertex_to_dof.find(cell->face(face_no)->vertex_index(fv));
                AssertThrow(dof_it != vertex_to_dof.end(),
                            ExcMessage("Could not map boundary vertex to nodal output DoF."));

                const types::global_dof_index dof = dof_it->second;

                set_boundary_kind(dof, bid, 0.0, 0.0);

                if (face_can_receive_recharge)
                {
                    const double recharge_value = gw_recharge.value(vertex);
                    if (std::abs(recharge_value - recharge_default_value) >
                        recharge_default_tolerance)
                    {
                        is_recharge[dof] = 1.0;
                        recharge[dof] = recharge_value;
                        set_boundary_kind(dof, bid, 1.0, 1.0);
                    }
                }

                if (dirichlet_it != dirichlet_boundary_map.end() && dirichlet_it->second != nullptr)
                {
                    is_dirichlet[dof] = 1.0;
                    dirichlet_head[dof] = dirichlet_it->second->value(vertex);
                    set_boundary_kind(dof, bid, 2.0, 2.0);
                }

                if (ghb_it != ghb_boundary_map.end())
                {
                    const auto &ghb_functions = ghb_it->second;
                    is_ghb[dof] = 1.0;

                    if (ghb_functions.head != nullptr)
                        ghb_head[dof] = ghb_functions.head->value(vertex);

                    if (ghb_functions.conductance != nullptr)
                        ghb_conductance[dof] = ghb_functions.conductance->value(vertex);

                    set_boundary_kind(dof, bid, 3.0, 3.0);
                }
            }
        }
    }

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_nodal);

  data_out.add_data_vector(recharge, "recharge");
  data_out.add_data_vector(kx, "kx");
  data_out.add_data_vector(ky, "ky");
  data_out.add_data_vector(kz, "kz");
  data_out.add_data_vector(sy, "sy");
  data_out.add_data_vector(ss, "ss");
  data_out.add_data_vector(boundary_id, "boundary_id");
  data_out.add_data_vector(boundary_kind, "boundary_kind");
  data_out.add_data_vector(is_boundary, "is_boundary");
  data_out.add_data_vector(is_recharge, "is_recharge_boundary");
  data_out.add_data_vector(is_dirichlet, "is_dirichlet_boundary");
  data_out.add_data_vector(dirichlet_head, "dirichlet_head");
  data_out.add_data_vector(is_ghb, "is_ghb_boundary");
  data_out.add_data_vector(ghb_head, "ghb_head");
  data_out.add_data_vector(ghb_conductance, "ghb_conductance");
  data_out.add_data_vector(proc_id, "proc_id");

  data_out.build_patches(1);

  const unsigned int digits = Utilities::needed_digits(n_proc);

  const std::string base_name = output_prefix_path() + "_mesh_properties";

  const std::string piece_name = base_name + "." + Utilities::int_to_string(my_rank, digits) + ".vtu";

  {
    std::ofstream out(piece_name.c_str());
    AssertThrow(out.good(),
                ExcMessage("Could not open local mesh property VTU: " + piece_name));
    data_out.write_vtu(out);
  }

  if (my_rank == 0)
  {
    std::vector<std::string> piece_names;
    piece_names.reserve(n_proc);

    for (unsigned int p = 0; p < n_proc; ++p)
    {
      piece_names.push_back(
          base_name + "." +
          Utilities::int_to_string(p, digits) +
          ".vtu");
    }

    const std::string master_name = base_name + ".pvtu";
    std::ofstream master(master_name.c_str());
    AssertThrow(master.good(),
                ExcMessage("Could not open master mesh property PVTU: " + master_name));

    data_out.write_pvtu_record(master, piece_names);
  }


}

#endif //NPSAT_FLOW_PREPARE_IMPL_H

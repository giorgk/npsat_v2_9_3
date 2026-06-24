//
// Created by giorgk on 6/24/26.
//

#ifndef NPSAT_FLOW_SETUP_IMPL_H
#define NPSAT_FLOW_SETUP_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::setup_system() {

    pcout << "Distributing DoFs..." << std::endl;

    {
        const unsigned int n_local = triangulation.n_locally_owned_active_cells();
        std::vector<unsigned char> seen(n_local, 0);
        for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        {
            if (!cell->is_locally_owned())
                continue;

            const unsigned int slot = cell->user_index();
            AssertThrow(slot < n_local,
                        ExcMessage("Cell user_index slots are not initialized after refinement."));
            AssertThrow(seen[slot] == 0,
                        ExcMessage("Duplicate cell user_index slot detected."));
            seen[slot] = 1;
        }
    }

    // 1. Distribute DoFs for each space
    dof_handler_flux.distribute_dofs(fe_flux);
    dof_handler_head.distribute_dofs(fe_head);
    dof_handler_trace.distribute_dofs(fe_trace);

    pcout << "Number of flux DoFs: " << dof_handler_flux.n_dofs() << std::endl;
    pcout << "Number of head DoFs: " << dof_handler_head.n_dofs() << std::endl;
    pcout << "Number of trace DoFs: " << dof_handler_trace.n_dofs() << std::endl;

    // 2. Extract DoF ownership information
    lambda_locally_owned_dofs = dof_handler_trace.locally_owned_dofs();
    flux_locally_owned_dofs = dof_handler_flux.locally_owned_dofs();
    head_locally_owned_dofs = dof_handler_head.locally_owned_dofs();

    DoFTools::extract_locally_relevant_dofs(dof_handler_trace,lambda_locally_relevant_dofs);
    DoFTools::extract_locally_relevant_dofs(dof_handler_head,head_locally_relevant_dofs);
    DoFTools::extract_locally_relevant_dofs(dof_handler_flux,flux_locally_relevant_dofs);

    // 4. Build constraints for the trace (unchanged)
    lambda_constraints.clear();
    lambda_constraints.reinit(lambda_locally_relevant_dofs);

    //Apply hanging node constraints
    DoFTools::make_hanging_node_constraints(dof_handler_trace,lambda_constraints);

}

#endif //NPSAT_FLOW_SETUP_IMPL_H

//
// Created by giorgk on 6/25/26.
//

#ifndef NPSAT_FLOW_SOLVE_IMPL_H
#define NPSAT_FLOW_SOLVE_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::solve(){
    pcout << "Solving global system ..." << std::endl;
    timer.restart();


    // 1. Build Schur RHS
    // Note: Using lambda_locally_owned_dofs for the schur_rhs
    TrilinosWrappers::MPI::Vector schur_rhs(lambda_locally_owned_dofs, mpi_communicator);
    //pcout << "schur_rhs has ghosts? " << schur_rhs.has_ghost_elements() << std::endl;
    build_schur_rhs(schur_rhs);

    //std::cout << "step 1" << std::endl;
    // 2. Preconditioner for S (AMG)
    // AMG is ideal for the face-based system in HMFEM
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    data.elliptic = true;
    preconditioner.initialize(block_system_matrix.block(0, 0), data);

    // 3. Create Schur complement operator
    npsat_flow::SchurComplement<dim> schur_op(block_system_matrix,
                                  lambda_locally_owned_dofs,
                                  well_locally_owned_dofs,
                                  lambda_constraints,
                                  mpi_communicator);

    // 4. Solve Schur complement system
    SolverControl solver_control(uo.solver_opt.System_iterations,
                              uo.solver_opt.System_tol * schur_rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::MPI::Vector lambda_owned(lambda_locally_owned_dofs, mpi_communicator);

    solver.solve(schur_op, lambda_owned,
                 /*block_solution.block(0),*/
                 schur_rhs,
                 preconditioner);

    lambda_constraints.distribute(lambda_owned /*block_solution.block(0)*/);

    // Copy into block_solution so downstream code keeps working
    block_solution.block(0) = lambda_owned;
    block_solution.block(0).update_ghost_values();

    pcout << "   Schur system (Lambda) converged in " << solver_control.last_step() << " iterations." << std::endl;

    solution_trace.reinit(lambda_locally_owned_dofs,
                      lambda_locally_relevant_dofs,
                      mpi_communicator);
    solution_trace = lambda_owned; //block_solution.block(0);
    solution_trace.update_ghost_values();

    back_substitute_well_heads(lambda_owned);
}

template <int dim>
void NPSAT_FLOW<dim>::build_schur_rhs(TrilinosWrappers::MPI::Vector& schur_rhs_owned) {
    AssertThrow(!schur_rhs_owned.has_ghost_elements(), ExcMessage("schur_rhs_owned must be owned-only (no ghosts)"));

    // (1) schur_rhs_owned = b_lambda (by owned dofs)
    pcout << "build_schur_rhs" << std::endl;

    // b in the lambda space
    schur_rhs_owned = block_rhs_vector.block(0);

    // Prepare W^{-1} b_w
    TrilinosWrappers::MPI::Vector invW_bw(well_locally_owned_dofs, mpi_communicator);
    const auto &W = block_system_matrix.block(1, 1);

    for (auto it = well_locally_owned_dofs.begin(); it != well_locally_owned_dofs.end(); ++it)
    {
        const auto w = *it;
        const double diag = W.diag_element(w);
        invW_bw(w) = (std::abs(diag) > 1e-16) ? (block_rhs_vector.block(1)(w) / diag) : 0.0;
    }

    // Compute R * inv(W) * b_w
    TrilinosWrappers::MPI::Vector RinvWbw(lambda_locally_owned_dofs, mpi_communicator);
    block_system_matrix.block(0, 1).vmult(RinvWbw, invW_bw);

    // Final Schur RHS
    schur_rhs_owned.add(-1.0, RinvWbw);
}

template <int dim>
void NPSAT_FLOW<dim>::back_substitute_well_heads(const TrilinosWrappers::MPI::Vector &lambda_owned) {
    TrilinosWrappers::MPI::Vector RTLambda(well_locally_owned_dofs, mpi_communicator);

    // Compute R^T * Lambda
    block_system_matrix.block(0, 1).Tvmult(RTLambda, lambda_owned /*block_solution.block(0)*/);

    // Solve well heads
    const auto &W = block_system_matrix.block(1, 1);
    for (auto it = well_locally_owned_dofs.begin(); it != well_locally_owned_dofs.end(); ++it) {
        const auto i = *it;

        const double bw = block_rhs_vector.block(1)(i);      // includes CWC_e * H_known and -Q
        const double rhs = bw - RTLambda(i);                 // b_w - R^T Λ
        const double hwi = rhs / W.diag_element(i);          // divide by W
        // pcout << "Well head:" << i << " "
        //       << bw << " "
        //       << rhs << " "
        //       << hwi << std::endl;
        well_solution(i) = hwi;
    }
    well_solution.compress(VectorOperation::insert);

    well_solution_ghosted = well_solution;          // copies owned entries
    well_solution_ghosted.update_ghost_values();

    block_solution.block(1) = well_solution;
    //block_solution.block(1).update_ghost_values();
}

#endif //NPSAT_FLOW_SOLVE_IMPL_H

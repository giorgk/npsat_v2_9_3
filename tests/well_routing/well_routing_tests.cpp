#include <deal.II/grid/grid_generator.h>
#include "tracer_under_test.cpp"

void require(bool condition, const char *message) {
    AssertThrow(condition, dealii::ExcMessage(message));
}

void outlet_tests() {
    using npsat_trace::select_well_route;
    using npsat_trace::WellRoute;
    for (unsigned int i = 0; i < 1000; ++i) {
        const double u = (i + 0.5) / 1000.;
        require(select_well_route(-20, 100, 120, u) == WellRoute::up, "extracting segment");
        require(select_well_route(0, 100, 100, u) == WellRoute::up, "zero exchange throughflow");
        require(select_well_route(0, -100, -100, u) == WellRoute::down, "reverse throughflow");
        require(select_well_route(100, 100, 0, u) == WellRoute::aquifer, "complete release");
        require(select_well_route(100, 40, -60, u) == WellRoute::aquifer, "converging axial flow");
        require(select_well_route(20, 100, 80, u) ==
                (i < 800 ? WellRoute::up : WellRoute::aquifer), "partial discharge");
        require(select_well_route(-100, -40, 60, u) ==
                (i < 600 ? WellRoute::up : WellRoute::down), "diverging axial flow");
    }
    require(select_well_route(0, 0, 0, .5) == WellRoute::stagnant, "stagnant segment");
    require(select_well_route(-10, 10, -10, .5) == WellRoute::invalid, "unbalanced flows");
}

// Four vertically connected cells. Exercise the actual routing function and
// ParticleHandler migration with one or two ranks and no external flow files.
void column_test(bool backward, bool pumping) {
    npsat_trace::Trace_options options;
    options.sim_opt.well_capture_cell_fraction = 10;
    NPSAT_TRACE<3> tracer(options);
    GridGenerator::subdivided_hyper_rectangle(tracer.triangulation,
        std::vector<unsigned int>{1, 1, 4}, Point<3>(0, 0, 0), Point<3>(1, 1, 4));
    tracer.dof_handler_flux.distribute_dofs(tracer.fe_flux);
    const double exchange[4] = {-100, 0, 20, 80};
    const double bottom[4] = {0, 100, 100, 80};
    const double top[4] = {100, 100, 80, 0};
    const double sign = backward ? -1 : 1;
    for (const auto &cell : tracer.dof_handler_flux.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        const unsigned int segment = static_cast<unsigned int>(cell->center()[2]);
        const unsigned int slot = tracer.slot_cell_well_links.size();
        tracer.cellid_to_slot[cell->id().to_string()] = slot;
        npsat_trace::CellWellLink link;
        link.wx = link.wy = .5;
        link.wbot = 0;
        link.wtop = 4;
        link.w_zbot = segment;
        link.sl = 1;
        link.n_segments = 4;
        tracer.slot_cell_well_links.push_back({link});
        npsat_trace::WellFlowRecord flow;
        flow.Qe = sign * (pumping ? -25 : exchange[segment]);
        flow.Qwbf_bot = sign * (pumping ? 25 * segment : bottom[segment]);
        flow.Qwbf_top = sign * (pumping ? 25 * (segment + 1) : top[segment]);
        tracer.flows_by_cell_well[{cell->active_cell_index(), 0}] = flow;
    }
    tracer.setup_triangulation_helpers();
    tracer.setup_particles();
    std::vector<npsat_trace::ParticleSeed> seeds;
    if (tracer.my_rank == 0)
        for (unsigned int i = 0; i < 1000; ++i) {
            npsat_trace::ParticleSeed seed{};
            seed.Eid = 1;
            seed.Sid = i;
            seed.x = .1 + .8 * (i + .5) / 1000.;
            seed.y = .3;
            seed.z = backward ? 3.5 : .5;
            seeds.push_back(seed);
        }
    tracer.distribute_particles(seeds);
    unsigned int released_lower = 0, released_upper = 0, captured = 0, handoffs = 0;
    for (unsigned int pass = 0; pass < 10; ++pass) {
        unsigned int pending = 0;
        for (auto p = tracer.particle_handler.begin(); p != tracer.particle_handler.end(); ++p) {
            auto props = p->get_properties();
            if (props[npsat_trace::pState] == 2) continue;
            const auto tria_cell = p->get_surrounding_cell(tracer.triangulation);
            DoFHandler<3>::active_cell_iterator cell(&tracer.triangulation,
                tria_cell->level(), tria_cell->index(), &tracer.dof_handler_flux);
            const auto result = tracer.well_bore_flow_trace(cell, p->get_location(),
                                                           props[npsat_trace::pWellId]);
            require(!result.stagnant, "unexpected stagnant route");
            props[npsat_trace::pWellId] = result.well_id;
            if (result.terminate) {
                require(pumping && !backward && result.end_reason == npsat_trace::er_well_captured,
                        "unexpected well termination");
                ++captured;
            } else {
                p->set_location(result.new_pos);
                if (result.well_id >= 0) {
                    ++pending;
                    ++handoffs;
                    continue;
                }
                if (backward && !pumping)
                    require(result.new_pos[2] < 1, "reverse crossflow must reach bottom aquifer");
                if (!backward && !pumping)
                    require(result.new_pos[2] > 2, "must pass zero-exchange segment");
                if (result.new_pos[2] < 3) ++released_lower;
                else ++released_upper;
            }
            require(props[npsat_trace::pAge] == 0, "well routing must be instantaneous");
            props[npsat_trace::pState] = 2;
        }
        if (Utilities::MPI::sum(pending, MPI_COMM_WORLD) == 0) break;
        tracer.particle_handler.sort_particles_into_subdomains_and_cells();
        require(pass < 9, "routing did not finish");
    }
    const unsigned int lower = Utilities::MPI::sum(released_lower, MPI_COMM_WORLD);
    const unsigned int upper = Utilities::MPI::sum(released_upper, MPI_COMM_WORLD);
    const unsigned int terminal = Utilities::MPI::sum(captured, MPI_COMM_WORLD);
    const unsigned int migrations = Utilities::MPI::sum(handoffs, MPI_COMM_WORLD);
    require(lower + upper + terminal == 1000, "lost particles");
    if (!backward && !pumping)
        require(lower > 150 && lower < 250 && upper == 1000 - lower,
                "partial-release ensemble should be approximately 20/80");
    if (pumping && !backward) require(terminal == 1000, "pump must capture all particles");
    if (tracer.n_proc > 1 && !(pumping && backward))
        require(migrations > 0, "MPI case must exercise captured-well handoff");
    tracer.pcout << "column backward=" << backward << " pumping=" << pumping
                 << " lower=" << lower << " upper=" << upper << " captured=" << terminal
                 << " handoffs=" << migrations << std::endl;
}

int main(int argc, char **argv) {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
    outlet_tests();
    column_test(false, false);
    column_test(true, false);
    column_test(false, true);
    column_test(true, true);
}

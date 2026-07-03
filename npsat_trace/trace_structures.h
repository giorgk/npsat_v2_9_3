//
// Created by giorgk on 6/23/2026.
//

#ifndef NPSAT_V2_TRACE_STRUCTURES_H
#define NPSAT_V2_TRACE_STRUCTURES_H


namespace npsat_trace {

    struct Sim_opt {
        int maxSteps = 1000;
        double porosity = 0.3;
        int N_time_steps = 5;
        double dt_eps = 0.01;
        int n_max_proc_exchanges = 100;
        int n_max_streamline_steps = 10000;
        int max_age = -1;
    };

    struct Trace_options {
        std::string input_prefix; // The main prefix for output files from NSPAT flow
        std::string output_prefix;
        std::string particles_file;
        std::string delta_time_file;
        Sim_opt sim_opt;
        int n_paticles_parallel = 20000;

        int write_loaded_tria = 0;
        int exit_after_load_tria = 0;
    };

    enum Prop : unsigned int
    {
        pPid = 0,
        pEid = 1,
        pSid = 2,
        //pRt  = 3,
        //pRf  = 3,
        //pAge = 4,        // double
        pDtRemaining = 4,// double
        pVmag = 5,       // double (last velocity magnitude)
        pState = 6,    //TODO define states optional: 0 dormant, 1 active, 2 exited
        pStreamlineSteps = 7,
        pAge = 8,
    };

    static constexpr unsigned int n_particle_props = 9;

    enum EndReason : int
    {
        er_reached_dt_end   = 1,  // finished this step on this rank (not a true termination)
        er_entered_ghost    = 2,
        er_exited_domain    = 3,
        er_stuck            = 4,
        er_bad_mapping      = 5,
        er_zero_velocity    = 6,
        er_water_table      = 7,
        MAX_ITER            = 8,
        MAX_AGE             = 9,
        er_bottom           = 10,
        er_lateral          = 11,
    };

    struct CellWellLink{
        std::uint32_t well_global_index = 0;
        std::uint16_t well_owner_rank   = 0;

        // Stable well metadata
        std::int32_t eid  = 0;
        double wx         = 0.0;
        double wy         = 0.0;
        double wtop       = 0.0;
        double wbot       = 0.0;
        std::int32_t q_row = 0;

        // Link geometry fields used by tracer
        double ze      = 0.0;
        double sl      = 0.0;
        double w_zbot  = 0.0;
    };

    enum class InputKind : std::uint8_t { Unknown=0, Direct=1, Wells=2 };

    using id32_t = std::uint32_t;

    struct WellRow
    {
        id32_t Eid = 0;
        double x=0, y=0, ztop=0, zbot=0;
        int rt=0, rf=0;
        int nlay=1, n_per_layer=1;
    };

    struct ParticleSeed
    {
        id32_t Eid = 0, Sid = 0;
        double x=0, y=0, z=0;
        id_t rt=0, rf=0;
    };

    struct FlowKey
    {
        std::uint64_t cell_id;
        std::uint32_t well_global_index;

        bool operator==(const FlowKey &o) const noexcept
        {
            return cell_id == o.cell_id && well_global_index == o.well_global_index;
        }
    };

    struct FlowKeyHash
    {
        std::size_t operator()(const FlowKey &k) const noexcept
        {
            // simple hash combine
            std::size_t h1 = std::hash<std::uint64_t>{}(k.cell_id);
            std::size_t h2 = std::hash<std::uint32_t>{}(k.well_global_index);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    struct WellFlowRecord
    {
        // Must match the POD layout written by npsat_v2.
        // We keep cell_id as uint64_t; if your writer uses uint32_t, it still reads safely
        // only if it wrote uint64_t. If it wrote uint32_t, change this to uint32_t here.
        std::uint32_t cell_id = 0;
        std::uint32_t well_global_index = 0;
        double ze = 0.0;        // if written
        double Qe        = 0.0;
        double Qwbf_bot  = 0.0;
        double Qwbf_top  = 0.0;
    };

    struct TimeStepControl
    {
        double max_step = 100000;
        unsigned int n_steps_per_cell = 5;
        double max_step_time = 100;
        unsigned int n_steps_per_time = 3;
    };

    struct ExitResult {
        int face_index; // deal.II index: 0-3 lateral, 4 bottom, 5 top
        dealii::Point<3> intersection_point;
    };

    struct FindHexExitResult
    {
        bool found;
        ExitResult value;

        FindHexExitResult()
            : found(false), value()
        {}

        FindHexExitResult(const ExitResult &v)
            : found(true), value(v)
        {}
    };

}

#endif //NPSAT_V2_TRACE_STRUCTURES_H

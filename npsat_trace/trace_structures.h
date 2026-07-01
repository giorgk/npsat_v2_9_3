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
        pState = 6,    // optional: 0 dormant, 1 active, 2 exited
        pStreamlineSteps = 7,
        pAge = 8,
    };

    static constexpr unsigned int n_particle_props = 9;

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

}

#endif //NPSAT_V2_TRACE_STRUCTURES_H

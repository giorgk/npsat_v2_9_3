//
// Created by giorgk on 6/23/2026.
//

#ifndef NPSAT_V2_TRACE_STRUCTURES_H
#define NPSAT_V2_TRACE_STRUCTURES_H


namespace npsat_trace {
    using namespace dealii;

    enum class TransportMethod : unsigned char {
        point,
        trajectory_atlas
    };

    enum class VelocityInterpolationScheme : unsigned char {
        split_rt0,
        coarse_rt,
        idw,
        cell_idw
    };

    template <int dim>
    struct CellVelocitySample {
        Point<dim> position;
        Tensor<1,dim> velocity;
    };

    struct IDW_opt {
        double power = 2.0;
        double proximity_tolerance = 0.01;
        // A positive value overrides the per-cell horizontal/vertical ratio.
        // Zero selects the automatic geometry-based estimate.
        double anisotropy_ratio = 0.0;
    };

    struct TimeStepControl
    {
        double max_step = 100000;
        unsigned int n_steps_per_cell = 5;
        double max_step_time = 100;
        unsigned int n_steps_per_time = 3;
    };

    struct Point_opt {
        TimeStepControl time_step_control;
    };

    struct Atlas_opt {
        unsigned int quadrature_points_per_direction = 2;
        unsigned int stored_samples_per_trajectory = 8;
        unsigned int maximum_query_samples = 32;
        unsigned int maximum_local_steps = 200;
        unsigned int maximum_branches_per_packet = 16;
        double kernel_power = 2.0;
        double kernel_epsilon = 1.0e-6;
        double minimum_packet_weight = 0.01;
        double minimum_split_weight = 0.01;
        double flow_tolerance = 1.0e-12;
        double balance_relative_tolerance = 1.0e-5;
    };

    struct Sim_opt {
        int maxSteps = 1000;
        double porosity = 0.3;
        int N_time_steps = 5;
        double dt_eps = 0.01;
        double direction = 1.0;
        double stagnant_velocity_threshold = 1.0e-8;
        double well_capture_distance = 10.0;
        double well_capture_cell_fraction = 0.2;
        double well_influence_q_scale = 1.0;
        double well_influence_max_cell_fraction = 0.45;
        int n_max_proc_exchanges = 100;
        int n_max_streamline_steps = 10000;
        int n_max_nonexpanding_steps = 50;
        int max_age = -1;
        TransportMethod transport_method = TransportMethod::point;
        VelocityInterpolationScheme velocity_interpolation = VelocityInterpolationScheme::split_rt0;
    };

    struct Misc_opt {
        std::string dbg_prefix;
        bool init_cell_dbg = false;
        bool particle_traj_dbg = false;
        bool cache_bilinear_coefficients = false;
    };

    struct BilinearMapCoefficients
    {
        double a0, a1, a2, a3;
        double b0, b1, b2, b3;

        BilinearMapCoefficients()
            : a0(0.0), a1(0.0), a2(0.0), a3(0.0),
              b0(0.0), b1(0.0), b2(0.0), b3(0.0)
        {}
    };

    struct Trace_options {
        std::string input_prefix; // The main prefix for output files from NSPAT flow
        std::string output_prefix;
        std::string particles_file;
        std::string delta_time_file;
        Sim_opt sim_opt;
        IDW_opt idw_opt;
        Point_opt point_opt;
        Atlas_opt atlas_opt;
        Misc_opt misc_opt;
        int n_paticles_parallel = 20000;

        int write_loaded_tria = 0;
        int exit_after_load_tria = 0;
        bool write_bin = false;
        bool write_ascii = true;
    };

    enum Prop : unsigned int
    {
        pPid = 0,
        pEid,
        pSid,
        //pRt,
        //pRf,
        pDtRemaining,// double
        pVmag,       // double (last velocity magnitude)
        pState,    //TODO define states optional: 0 dormant, 1 active, 2 exited
        pStreamlineSteps,
        pAge,
        pBBoxMinX,
        pBBoxMinY,
        pBBoxMinZ,
        pBBoxMaxX,
        pBBoxMaxY,
        pBBoxMaxZ,
        pNoExpandCount,
        n_particle_props
    };

    //static constexpr unsigned int n_particle_props = 9;

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
        er_well_captured    = 12,
        er_well_mass_balance = 13,
        er_nonexpanding     = 14,
        er_max_proc_exchanges = 15,
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
        std::int32_t n_segments = 0;

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
        double radius = 0.0;
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
        // Must match the fields read from npsat_v2 particle-well-flow v1 records.
        std::uint32_t cell_id = 0;
        std::uint32_t well_global_index = 0;
        double ze = 0.0;        // if written
        double Qe        = 0.0;
        double Qwbf_bot  = 0.0;
        double Qwbf_top  = 0.0;
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

    template<int dim>
    struct WellBoreTraceResults
    {
        bool terminate;
        int end_reason;
        typename dealii::DoFHandler<dim>::active_cell_iterator new_cell;
        dealii::Point<dim> new_pos;
    };

    enum class NewtonFailure
    {
        None,
        SingularJacobian,
        NanIterate,
        MaxIterations
    };

    struct NewtonDebugInfo
    {
        NewtonFailure failure;
        int iterations;
        double u;
        double v;
        double detJ;
        double residual;
        double curX;
        double curY;

        NewtonDebugInfo()
            : failure(NewtonFailure::None),
              iterations(0),
              u(0.0),
              v(0.0),
              detJ(0.0),
              residual(0.0),
              curX(0.0),
              curY(0.0)
        {}
    };

}

#endif //NPSAT_V2_TRACE_STRUCTURES_H

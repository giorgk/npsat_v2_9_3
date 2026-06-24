//
// Created by giorgk on 6/22/2026.
//
#include <string>
#include <vector>

#ifndef NPSAT_V2_FLOW_STRUCTURES_H
#define NPSAT_V2_FLOW_STRUCTURES_H

namespace npsat_flow {

    static const std::vector<unsigned int> face_order = {0, 1, 3, 2};

    struct Refinement_options
    {
        int initial = 0;
        int wells = 0;
        int streams = 0;
        int top = 0;
        int top_depth = 0;
        int dirichlet = 0;
        int GHB = 0;
        int neumann = 0;
    };

    struct Sources_uo
    {
        std::string rch_file;
        double rch_factor = 1.0;

        std::string stream_file;
        std::string stream_rates_file;
        double stream_factor = 1.0;

        std::string well_file;
        std::string well_rates_file;
        double well_factor = 1.0;
    };

    struct Hydrogeology_options {
        std::string kx_file;
        std::string kz_file;
        std::string ss_file;
        std::string sy_file;
    };

    struct Time_step_uo
    {
        //std::string time_file;
        unsigned int n_sim_steps;
        unsigned int start_step;
    };

    struct BC_uo
    {
        std::string dirichlet_file;
        std::string ghb_file;
        std::string neumann_file;
        double half_with;
        double min_overlap;
    };

    struct Sim_uo {
        int n_steps = 1;
        int Start_step = 0;
        std::string delta_time_file;
        bool confined = false;
    };

    struct Solver_uo {
        int System_iterations = 15000;
        double System_tol = 1e-12;
    };

    struct NonlinearControls
    {
        enum class RechargeStabilizationMode
        {
            HysteresisOnly,
            EffectiveTop
        };
        enum class EffectiveTopMode
        {
            Off,
            RechargeReceivers,
            AllWaterTableCells
        };

        unsigned int max_picard_iters = 25;
        double       abs_tol_update   = 1e-2;   // e.g. meters
        double       rel_tol_update   = 1e-4;
        double       damping_omega    = 0.7;    // Picard damping
        bool         use_anderson     = true;
        unsigned int anderson_start  = 2;      // after 1–2 Picard steps
        unsigned int anderson_m      = 5;      // memory depth <= 5
        double       anderson_reg    = 1e-11;  // small regularization for LS
        bool         carry_history_across_timesteps = false;
        bool         use_recharge_hysteresis = true;
        double       recharge_drying_saturated_fraction = 5.0e-3;
        double       recharge_wetting_saturated_fraction = 5.0e-2;
        double       recharge_min_relative_k = 1.0e-4;
        RechargeStabilizationMode recharge_stabilization_mode =
            RechargeStabilizationMode::EffectiveTop;
        EffectiveTopMode effective_top_mode = EffectiveTopMode::RechargeReceivers;
    };


    struct user_options
    {
        std::string main_path;
        std::string output_path;
        std::string input_path;
        std::string output_prefix;
        std::string initial_head_file;

        bool isBox = false;
        std::vector<double> box_dims;
        std::vector<double> box_llp;
        std::vector<unsigned int> box_nxyz;
        std::vector<double> vert_discr;
        std::string vert_file;
        std::string mesh_file;
        std::string top_fnc;
        std::string bot_fnc;

        bool print_matrices;
        bool print_initial_mesh = false;
        bool print_mesh_with_prop = false;
        bool print_mesh_exit = false;
        bool save_trace_data = false;
        bool print_vtk = false;
        bool print_water_table = false;
        bool print_q_to_vtu = false;
        bool print_lambda_to_vtu = false;
        bool print_area_to_vtu = false;
        bool print_cell_budget_csv = false;
        bool print_solution_cell_vtu = false;
        bool print_cellcenters_vtk = false;
        bool print_cell_well_map_csv = false;
        bool print_trace_well_maps_csv = false;
        bool print_wellbore_segments_csv = false;
        bool print_wellbore_segments_vtu = false;
        bool print_wellbore_segments_legacy_vtk = false;
        bool print_wellboreflow_csv = false;
        bool print_well_resid_csv = false;

        int n_steps = 1;
        int start_step = 0;

        Refinement_options ref_opt;
        Hydrogeology_options hgeo;
        Sources_uo sources;

        Time_step_uo time_tracking;

        BC_uo bndr_cond;
        Sim_uo sim_opt;
        Solver_uo solver_opt;
        NonlinearControls NLC;
    };

    template <typename T>
    struct MatrixView
    {
        static_assert(std::is_arithmetic<T>::value, "MatrixView<T>: T must be arithmetic.");
        T *data = nullptr;
        std::int64_t nrows = 0;
        std::int64_t ncols = 0;

        inline T &operator()(std::int64_t i, std::int64_t j)
        {
            return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) +
                        static_cast<std::size_t>(j)];
        }

        inline const T &operator()(std::int64_t i, std::int64_t j) const
        {
            return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) +
                        static_cast<std::size_t>(j)];
        }
        inline bool empty() const { return (data == nullptr) || (nrows == 0) || (ncols == 0); }
    };

    struct ScreenClip
    {
        double z_low = 0.0;
        double z_high = 0.0;
        double L = 0.0;
    };

}

#endif //NPSAT_V2_FLOW_STRUCTURES_H

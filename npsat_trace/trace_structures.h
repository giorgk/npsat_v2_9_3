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
    };

}

#endif //NPSAT_V2_TRACE_STRUCTURES_H

//
// Created by giorgk on 6/23/26.
//

#ifndef TIME_STEP_TRACKING_H
#define TIME_STEP_TRACKING_H

#include <fstream>
#include <sstream>
#include <vector>

#include "helper_func.h"

namespace npsat_flow{
    class TimeStepTracker{
    public:
        TimeStepTracker() = default;

        void read_delta_time_file(const std::string &filename)
        {
            delta_t.clear();

            std::ifstream in(filename);
            if (!in)
                throw std::runtime_error("Cannot open delta-time file: " + filename);

            std::string line;
            unsigned int line_number = 0;
            while (std::getline(in, line))
            {
                ++line_number;

                const std::size_t comment_pos = line.find('#');
                if (comment_pos != std::string::npos)
                    line.erase(comment_pos);

                line = trim(line);
                if (line.empty())
                    continue;

                std::stringstream converter(line);
                double dt = 0.0;
                converter >> dt;

                if (converter.fail() || !(converter >> std::ws).eof())
                {
                    throw std::runtime_error(
                        "Invalid delta-time value in " + filename +
                        " at line " + std::to_string(line_number) + ": " + line);
                }

                delta_t.push_back(dt);
            }

            if (delta_t.empty())
                throw std::runtime_error("Delta-time file is empty: " + filename);
        }

        void initialize(const unsigned int n_simulation_steps,
                    const unsigned int start_file_step)
        {
            if (delta_t.empty())
                throw std::runtime_error("TimeStepTracker: delta_t has not been loaded.");

            nsteps_sim = n_simulation_steps;
            start_step = start_file_step % delta_t.size();
            isim = 0;
        }

        bool done() const
        {
            return isim >= nsteps_sim;
        }

        void advance()
        {
            if (!done())
                ++isim;
        }

        unsigned int simulation_step() const
        {
            return isim;
        }

        unsigned int file_step() const
        {
            return static_cast<unsigned int>((start_step + isim) % delta_t.size());
        }

        double duration() const
        {
            return delta_t[file_step()];
        }

        unsigned int n_file_steps() const
        {
            return static_cast<unsigned int>(delta_t.size());
        }

        unsigned int n_sim_steps() const
        {
            return nsteps_sim;
        }
    private:
        std::vector<double> delta_t;

        unsigned int nsteps_sim = 0;
        unsigned int start_step = 0;
        unsigned int isim = 0;
    };

}

#endif //TIME_STEP_TRACKING_H

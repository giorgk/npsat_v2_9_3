//
// Created by giorgk on 6/27/26.
//

#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H

#include "trace_structures.h"

namespace  npsat_trace{
    class ParticleReader{
    public:
        ParticleReader() = default;
        explicit ParticleReader(std::size_t max_particles_per_chunk)
        : max_particles(max_particles_per_chunk) {}

    private:
        static void parse_direct_line_strict(const std::string &line, ParticleSeed &p);
        static void parse_well_line_strict(const std::string &line, WellRow &w);
        static void emit_well_particles(const WellRow &w, std::vector<ParticleSeed> &out);

        bool has_buffered = false;
        std::string buffered_line;
        std::uint64_t buffered_line_no = 0;

        std::string file;
        std::ifstream in;
        std::uint64_t line_no = 0;
        InputKind kind = InputKind::Unknown;
        bool eof = false;

        std::size_t max_particles = 20000; // default chunk size
    };
}

#endif //PARTICLE_READER_H

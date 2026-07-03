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

        void open(const std::string &filename);
        void close();
        std::vector<ParticleSeed> read_next_chunk();

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

    inline void ParticleReader::open(const std::string &filename) {
        close();
        file = filename;

        in.open(file);
        if (!in.good())
            throw std::runtime_error("Could not open particle input file: " + file);

        line_no = 0;
        kind = InputKind::Unknown;
        eof = false;
    }

    inline void ParticleReader::close(){
        if (in.is_open()) in.close();
        file.clear();
        line_no = 0;
        kind = InputKind::Unknown;
        eof = false;
        has_buffered = false;
        buffered_line.clear();
        buffered_line_no = 0;
    }

    inline std::vector<ParticleSeed> ParticleReader::read_next_chunk() {
        AssertThrow(in.is_open(), dealii::ExcMessage("ParticleReader::open() must be called first."));
        if (eof) return {};

        std::vector<ParticleSeed> out;
        out.reserve(max_particles);

        if (has_buffered) {
            if (kind != InputKind::Direct)
                throw std::runtime_error("Internal ParticleReader error: buffered line exists for non-direct input.");

            ParticleSeed p;
            parse_direct_line_strict(buffered_line, p);
            out.push_back(p);

            buffered_line.clear();
            buffered_line_no = 0;
            has_buffered = false;
        }
        std::string line;
        while (std::getline(in, line)) {
            ++line_no;
            if (is_comment_or_empty(line))
                continue;

            // detect kind on first data line
            if (kind == InputKind::Unknown) {
                const auto toks = split_tokens(line);
                if (toks.size() == 7)
                    kind = InputKind::Direct;
                else if (toks.size() == 9)
                    kind = InputKind::Wells;
                else
                    throw std::runtime_error("Unrecognized particle input format at line " +
                                             std::to_string(line_no) + ": expected 7 or 9 columns, got " +
                                             std::to_string(toks.size()));
            }
            if (kind == InputKind::Direct) {
                // Stop exactly at max_particles for direct particles
                if (out.size() >= max_particles) {
                    // put stream position back one line is messy; instead, we stop only between lines
                    // (we already read this line). Easiest: parse it next call by buffering.
                    buffered_line = line;
                    buffered_line_no = line_no;
                    has_buffered = true;
                    return out;
                }
                ParticleSeed p;
                parse_direct_line_strict(line, p);
                out.push_back(p);
            }
            else {
                // For wells: never stop mid-well. If we hit the limit, we still emit the whole well.
                WellRow w;
                parse_well_line_strict(line, w);
                const std::size_t n_emit = static_cast<std::size_t>(w.nlay) *
                                     static_cast<std::size_t>(w.n_per_layer);

                // Even if out.size() >= max_particles, we still emit this well, then stop.
                emit_well_particles(w, out);

                if (out.size() >= max_particles)
                    return out;
            }
        }// end while
        eof = true;
        return out;
    }

    inline void ParticleReader::emit_well_particles(const WellRow &w, std::vector<ParticleSeed> &out) {
        // TODO revise the generator for the well case
        for (int ilay = 0; ilay < w.nlay; ++ilay) {
            const double s = (w.nlay == 1) ? 0.5 : double(ilay) / double(w.nlay - 1);
            const double z = w.zbot + s * (w.ztop - w.zbot);

            for (int j = 0; j < w.n_per_layer; ++j) {
                ParticleSeed p;
                p.Eid = w.Eid;
                p.Sid = static_cast<id_t>(ilay * w.n_per_layer + j);
                p.x = w.x; // around-well pattern can be injected later
                p.y = w.y;
                p.z = z;
                p.rt = w.rt;
                p.rf = w.rf;

                out.push_back(p);
            }
        }
    }


    inline void ParticleReader::parse_direct_line_strict(const std::string &line, ParticleSeed &p) {
        const auto toks = split_tokens(line);
        if (toks.size() != 7)
            throw std::runtime_error("Direct particle format error: expected 7 columns, got " +
                                     std::to_string(toks.size()));

        // Eid Sid x y z rt rf
        // use wide integers for parse then cast
        std::uint64_t Eid64=0, Sid64=0;
        double x=0,y=0,z=0;
        long long rt=0, rf=0;
        {
            std::istringstream iss;
            std::string s = line;
            normalize_separators(s);
            iss.str(s);
            if (!(iss >> Eid64 >> Sid64 >> x >> y >> z >> rt >> rf))
                throw std::runtime_error("Direct particle format parse failure.");
        }

        p.Eid = static_cast<id_t>(Eid64);
        p.Sid = static_cast<id_t>(Sid64);
        p.x = x; p.y = y; p.z = z;
        p.rt = static_cast<std::int32_t>(rt);
        p.rf = static_cast<std::int32_t>(rf);
    }

    inline void ParticleReader::parse_well_line_strict(const std::string &line, WellRow &w) {
        const auto toks = split_tokens(line);
        if (toks.size() != 9)
            throw std::runtime_error("Well particle format error: expected 9 columns, got " +
                                     std::to_string(toks.size()));

        std::uint64_t Eid64=0;
        double x=0,y=0,ztop=0,zbot=0;
        long long rt=0, rf=0, nlay=0, npl=0;

        {
            std::istringstream iss;
            std::string s = line;
            normalize_separators(s);
            iss.str(s);
            if (!(iss >> Eid64 >> x >> y >> ztop >> zbot >> rt >> rf >> nlay >> npl))
                throw std::runtime_error("Well particle format parse failure.");
        }

        if (nlay < 2 || npl < 2)
            throw std::runtime_error("Well particle format error: nlay and n_per_layer must be >= 2.");

        w.Eid = static_cast<id_t>(Eid64);
        w.x = x; w.y = y;
        w.ztop = ztop; w.zbot = zbot;
        w.rt = static_cast<std::int32_t>(rt);
        w.rf = static_cast<std::int32_t>(rf);
        w.nlay = static_cast<std::int32_t>(nlay);
        w.n_per_layer = static_cast<std::int32_t>(npl);
    }
}

#endif //PARTICLE_READER_H

//
// Created by giorgk on 7/2/26.
//

#ifndef READER_HELPERS_H
#define READER_HELPERS_H

namespace npsat_trace {
    using namespace dealii;

    inline std::string vface_rt0_filename(const std::string &prefix,
                                          const int my_rank,
                                          const unsigned int step_no)
    {
        const std::string step     = Utilities::int_to_string(step_no, 3);
        const std::string str_rank = Utilities::int_to_string(my_rank, 4);

        return prefix + "_Vface_rt0_vals_rank_" + str_rank + "_step_" + step + ".bin";
    }
    // ----------------------------
    // File 1: RT0 face velocities
    // ----------------------------
    struct VFaceRt0Header
    {
        std::uint64_t magic   = 0;
        std::uint32_t version = 0;
        std::uint64_t step_no = 0;
        double time = 0.0;
        double dt   = 0.0;
        std::uint64_t n_local = 0;
        double checksum_sum    = 0.0;
        double checksum_maxabs = 0.0;
    };


    inline void read_vface_rt0_vals_into_vector(
        const std::string &prefix,
        const dealii::IndexSet &owned_now,
        const dealii::IndexSet &relevant_now,
        const int my_rank,
        const unsigned int step_no,
        TrilinosWrappers::MPI::Vector &vface) { // must already be reinit(owned,relevant)

        (void)relevant_now;

        const std::string filename = vface_rt0_filename(prefix, my_rank, step_no);

        std::ifstream in(filename, std::ios::binary);
        if (!in.good())
            throw std::runtime_error("Could not open file for reading: " + filename);

        VFaceRt0Header h;
        read_pod(in, h.magic);
        read_pod(in, h.version);
        read_pod(in, h.step_no);
        read_pod(in, h.time);
        read_pod(in, h.dt);
        read_pod(in, h.n_local);
        read_pod(in, h.checksum_sum);
        read_pod(in, h.checksum_maxabs);

        constexpr std::uint64_t expected_magic = 0x4E505341545F5646ULL; // "NPSAT_VF"
        if (h.magic != expected_magic)
            throw std::runtime_error("Bad magic in file: " + filename);

        if (h.version != 2)
            throw std::runtime_error("Unsupported version in file: " + filename);

        if (h.step_no != static_cast<std::uint64_t>(step_no))
            throw std::runtime_error("Step mismatch in file: " + filename);

        if (h.n_local != owned_now.n_elements())
        {
            throw std::runtime_error("n_local mismatch in file '" + filename +
                                     "': file has " + std::to_string(h.n_local) +
                                     ", but owned_now has " + std::to_string(owned_now.n_elements()));
        }

        std::vector<double> values(static_cast<std::size_t>(h.n_local));
        if (h.n_local > 0)
        {
            in.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(double)));
            if (!in.good())
                throw std::runtime_error("Failed reading values array: " + filename);
        }

        // Optional checksum verification
        long double checksum_sum = 0.0L;
        double checksum_maxabs   = 0.0;
        for (const double v : values)
        {
            checksum_sum += static_cast<long double>(v);
            checksum_maxabs = std::max(checksum_maxabs, std::abs(v));
        }

        const double sum_d = static_cast<double>(checksum_sum);
        const double tol_sum = 1e-12 * std::max(1.0, std::abs(h.checksum_sum));
        const double tol_max = 1e-12 * std::max(1.0, std::abs(h.checksum_maxabs));

        if (std::abs(sum_d - h.checksum_sum) > tol_sum)
            throw std::runtime_error("Checksum sum mismatch in file: " + filename);

        if (std::abs(checksum_maxabs - h.checksum_maxabs) > tol_max)
            throw std::runtime_error("Checksum maxabs mismatch in file: " + filename);

        // Clear and scatter owned entries in the same order as written
        vface = 0.0;

        // Scatter values into owned entries in THE SAME ORDER as written (IndexSet iteration order)
        std::size_t k = 0;
        for (auto it = owned_now.begin(); it != owned_now.end(); ++it, ++k)
        {
            const auto gid = *it;
            vface[gid] = values[k];
        }
        vface.compress(VectorOperation::insert);
        vface.update_ghost_values();
    }




}

#endif //READER_HELPERS_H

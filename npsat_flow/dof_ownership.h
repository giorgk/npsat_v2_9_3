//
// Created by giorgk on 6/24/26.
//

#ifndef DOF_OWNERSHIP_H
#define DOF_OWNERSHIP_H

namespace npsat_flow{
    using namespace dealii;

    struct OwnerRange
    {
        types::global_dof_index begin; // inclusive
        types::global_dof_index end;   // exclusive
        unsigned int rank;
    };

    class OwnershipManager
    {
    public:
        OwnershipManager() = default;
        void reinit(const IndexSet &locally_owned, MPI_Comm comm);
        // void build_cache(const std::vector<types::global_dof_index> &indices, const IndexSet &locally_owned);
        // void build_cache(const SortedVectorMap<unsigned int, std::vector<TraceRef>> &well_to_trace_dof,
        //                  const IndexSet                                             &trace_locally_owned,
        //                  const unsigned int                                          my_rank);
        unsigned int get_owner(const types::global_dof_index idx) const;
        unsigned int get_owner_cached(const types::global_dof_index idx) const;
        void print_statistics(const std::string &prefix) const;
        void reset_cache_stats() const;
        void print_cache_stats(const MPI_Comm comm, const std::string &label = "") const;
        void print_owner_ranges(const int my_rank) const;

    private:
        std::vector<OwnerRange> owner_ranges;
        std::unordered_map<types::global_dof_index, unsigned int> cache;
        mutable unsigned int cache_hits   = 0;
        mutable unsigned int cache_misses = 0;
    };

    inline void OwnershipManager::print_owner_ranges(const int my_rank) const
    {
        for (auto r : owner_ranges)
        {
            std::cout << "Rank " << my_rank << ": Range: [" << r.begin << " - " << r.end << "]" << std::endl;
        }
    }

    inline unsigned int OwnershipManager::get_owner(const types::global_dof_index idx) const
    {
        // Find the first range starting AFTER idx
        auto it = std::upper_bound(
            owner_ranges.begin(), owner_ranges.end(), idx,
            [](const types::global_dof_index value, const OwnerRange &r) {
                return value < r.begin;
            });

        // If it points to the beginning, idx is smaller than any owned range
        AssertThrow(it != owner_ranges.begin(),
                    ExcMessage("Index is below the minimum owned range."));

        // Step back to the range that should contain idx
        --it;

        // Verify the index actually falls within this range (handles gaps)
        AssertThrow(it->begin <= idx && idx < it->end,
                    ExcMessage("Index does not belong to any known owner range."));

        return it->rank;
    }

    inline unsigned int OwnershipManager::get_owner_cached(const types::global_dof_index idx) const
    {
        auto it = cache.find(idx);
        if (it != cache.end()) {
            ++cache_hits;
            return it->second;
        }
        ++cache_misses;
        return get_owner(idx);
    }

    inline void OwnershipManager::reinit(const IndexSet& locally_owned, MPI_Comm comm)
    {
        int n_proc = 0;
        MPI_Comm_size(comm, &n_proc);
        int my_rank = 0;
        MPI_Comm_rank(comm, &my_rank);

        // 1. Extract local ranges
        std::vector<std::pair<types::global_dof_index, types::global_dof_index>> local_intervals;
        for (auto it = locally_owned.begin_intervals(); it != locally_owned.end_intervals(); ++it)
        {
            const auto &acc = *it; // IntervalAccessor
            // Defensive: handle empty interval (begin == end)
            if (acc.begin() == acc.end())
                continue;
            const auto b = static_cast<types::global_dof_index>(*acc.begin());  // first element
            const auto e = static_cast<types::global_dof_index>(acc.last()) + 1; // exclusive end

            if (b < e) // Only store and send ranges that actually contain indices
            {
                local_intervals.emplace_back(b, e);
                std::cout << "Rank " << my_rank << " local interval [" << b << " - " << e <<"]" << std::endl;
            }

        }

        // 2. Pack for MPI
        std::vector<std::uint64_t> send;
        send.reserve(2 * local_intervals.size());
        for (const auto &rng : local_intervals)
        {
            send.push_back(static_cast<std::uint64_t>(rng.first));
            send.push_back(static_cast<std::uint64_t>(rng.second));
        }

        // 3. Allgather sizes and data
        std::vector<int> recv_counts(n_proc, 0);
        int send_count = static_cast<int>(send.size());
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

        std::vector<int> displs(n_proc, 0);
        int total = 0;
        for (int p = 0; p < n_proc; ++p)
        {
            displs[p] = total;
            total += recv_counts[p];
        }

        std::vector<std::uint64_t> recv(static_cast<std::size_t>(total));
        MPI_Allgatherv(send.data(), send_count, MPI_UINT64_T,
                       recv.data(), recv_counts.data(), displs.data(), MPI_UINT64_T, comm);

        // 4. Build and Sort global range list
        owner_ranges.clear();
        owner_ranges.reserve(static_cast<std::size_t>(total / 2));
        for (int p = 0; p < n_proc; ++p)
        {
            for (int k = 0; k < recv_counts[p]; k += 2)
            {
                const auto b = static_cast<types::global_dof_index>(recv[displs[p] + k]);
                const auto e = static_cast<types::global_dof_index>(recv[displs[p] + k + 1]);
                if (b < e)
                    owner_ranges.push_back({b, e, static_cast<unsigned int>(p)});
            }
        }
        std::sort(owner_ranges.begin(), owner_ranges.end(),
                  [](const OwnerRange &a, const OwnerRange &b) { return a.begin < b.begin; });

        print_owner_ranges(my_rank);
    }

    // inline void OwnershipManager::build_cache(const std::vector<types::global_dof_index>& indices, const IndexSet& locally_owned)
    // {
    //     cache.clear();
    //     std::vector<types::global_dof_index> remote_candidates;
    //
    //     for (const auto idx : indices)
    //         if (!locally_owned.is_element(idx))
    //             remote_candidates.push_back(idx);
    //
    //     std::sort(remote_candidates.begin(), remote_candidates.end());
    //     remote_candidates.erase(std::unique(remote_candidates.begin(), remote_candidates.end()),
    //                             remote_candidates.end());
    //     for (const auto idx : remote_candidates)
    //         cache[idx] = get_owner(idx);
    // }

    // inline void OwnershipManager::build_cache(const SortedVectorMap<unsigned int,
    //     std::vector<TraceRef>>& well_to_trace_dof,
    //     const IndexSet& trace_locally_owned,
    //     const unsigned int my_rank) {
    //     cache.clear();
    //     // Heuristic reserve: number of remote trace refs touched by owned wells.
    //     // Not exact but reduces rehashing.
    //     std::size_t reserve_guess = 0;
    //     for (const auto &kv : well_to_trace_dof.data())
    //         reserve_guess += kv.second.size();
    //     cache.reserve(static_cast<std::size_t>(reserve_guess * 11 / 10 + 16));
    //
    //     for (const auto &kv : well_to_trace_dof.data())
    //     {
    //         const auto &trace_list = kv.second;
    //         for (const auto &tr : trace_list)
    //         {
    //             // Skip locals: either by rank or by IndexSet (both are cheap).
    //             // Using IndexSet is safer if trace_rank could be stale (shouldn't be).
    //             if (tr.trace_rank == my_rank)
    //                 continue;
    //
    //             if (trace_locally_owned.is_element(tr.trace_id))
    //                 continue;
    //
    //             auto it = cache.find(tr.trace_id);
    //             if (it == cache.end())
    //             {
    //                 cache.emplace(tr.trace_id, tr.trace_rank);
    //             }
    //             else
    //             {
    //                 // Sanity: same trace_id must always map to same owner rank
    //                 AssertThrow(it->second == tr.trace_rank,
    //                             ExcMessage("Inconsistent trace ownership in well_to_trace_dof."));
    //             }
    //         }
    //     }
    // }

    inline void OwnershipManager::reset_cache_stats() const
    {
        cache_hits   = 0;
        cache_misses = 0;
    }

    inline void OwnershipManager::print_cache_stats(const MPI_Comm comm,
                           const std::string &label ) const
    {
        const unsigned int local_hits   = cache_hits;
        const unsigned int local_misses = cache_misses;

        const unsigned int global_hits =
            Utilities::MPI::sum(local_hits, comm);
        const unsigned int global_misses =
            Utilities::MPI::sum(local_misses, comm);

        if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
            std::cout << "[OwnershipManager]";
            if (!label.empty()) std::cout << " " << label;
            std::cout << " cache hits=" << global_hits
                      << ", misses=" << global_misses
                      << ", hit-rate="
                      << (global_hits + global_misses > 0
                          ? double(global_hits) /
                            double(global_hits + global_misses)
                          : 1.0)
                      << std::endl;
        }
    }

    inline void OwnershipManager::print_statistics(const std::string& prefix) const
    {
        int my_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        // 1. Write the Global Owner Ranges (The map of the whole world)
        std::ofstream range_file(prefix + "_ranges_rank_" + std::to_string(my_rank) + ".txt");
        range_file << "# Global Ownership Ranges gathered by Rank " << my_rank << "\n";
        range_file << "# Begin \t End \t OwnerRank\n";
        for (const auto &r : owner_ranges)
        {
            range_file << r.begin << "\t" << r.end << "\t" << r.rank << "\n";
        }
        range_file.close();

        // 2. Write the Local Cache (Indices this rank specifically cares about)
        std::ofstream cache_file(prefix + "_cache_rank_" + std::to_string(my_rank) + ".txt");
        cache_file << "# Cache entries for Rank " << my_rank << "\n";
        cache_file << "# GlobalIndex \t OwnerRank\n";
        for (const auto &entry : cache)
        {
            cache_file << entry.first << "\t" << entry.second << "\n";
        }
        cache_file.close();
    }

}

#endif //DOF_OWNERSHIP_H

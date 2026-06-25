//
// Created by giorgk on 6/25/26.
//

#ifndef MPI_HELPERS_H
#define MPI_HELPERS_H

#include <vector>
#include <mpi.h>
#include <limits>
#include <stdexcept>
#include <numeric>

namespace npsat_flow {

template <class T>
    std::vector<std::vector<T>> all_to_all_vector_of_vectors(const MPI_Comm comm,
                               const std::vector<std::vector<T>> &send)
    {
        static_assert(std::is_trivially_copyable<T>::value,
                  "all_to_all_vector_of_vectors requires T to be trivially copyable. "
                  "If T is not trivially copyable, you must serialize/pack it.");
        int n_proc = 0;
        MPI_Comm_size(comm, &n_proc);
        if (static_cast<int>(send.size()) != n_proc)
            throw std::runtime_error("send.size() must equal number of MPI ranks.");

        // -------------------------
        // (1) sendcounts in elements
        // -------------------------
        std::vector<int> sendcounts(n_proc, 0);
        for (int r = 0; r < n_proc; ++r)
        {
            const auto n = send[r].size();
            if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("send[r].size() exceeds INT_MAX.");
            sendcounts[r] = static_cast<int>(n);
        }

        // -------------------------
        // (2) exchange counts
        // -------------------------
        std::vector<int> recvcounts(n_proc, 0);
        MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                     recvcounts.data(), 1, MPI_INT,
                     comm);

        // -------------------------
        // (3) displacements in elements
        // -------------------------
        std::vector<int> sdispls(n_proc, 0), rdispls(n_proc, 0);
        for (int r = 1; r < n_proc; ++r)
        {
            sdispls[r] = sdispls[r-1] + sendcounts[r-1];
            rdispls[r] = rdispls[r-1] + recvcounts[r-1];
        }

        const int send_total = std::accumulate(sendcounts.begin(), sendcounts.end(), std::size_t(0));
        const int recv_total = std::accumulate(recvcounts.begin(), recvcounts.end(), std::size_t(0));

        // -------------------------
        // (4) pack send into contiguous buffer
        // -------------------------
        std::vector<T> sendbuf;
        sendbuf.reserve(send_total);
        for (int r = 0; r < n_proc; ++r)
            sendbuf.insert(sendbuf.end(), send[r].begin(), send[r].end());

        // -------------------------
        // (5) alltoallv exchange (in bytes)
        //     We use MPI_BYTE so we don't need to build an MPI_Datatype for T.
        // -------------------------
        std::vector<int> sendcounts_bytes(n_proc), recvcounts_bytes(n_proc);
        std::vector<int> sdispls_bytes(n_proc),    rdispls_bytes(n_proc);

        for (int r = 0; r < n_proc; ++r)
        {
            sendcounts_bytes[r] = sendcounts[r] * static_cast<int>(sizeof(T));
            recvcounts_bytes[r] = recvcounts[r] * static_cast<int>(sizeof(T));
            sdispls_bytes[r]    = sdispls[r]    * static_cast<int>(sizeof(T));
            rdispls_bytes[r]    = rdispls[r]    * static_cast<int>(sizeof(T));
        }

        std::vector<T> recvbuf(static_cast<std::size_t>(recv_total));

        MPI_Alltoallv(reinterpret_cast<const unsigned char*>(sendbuf.data()),
                  sendcounts_bytes.data(),
                  sdispls_bytes.data(),
                  MPI_BYTE,
                  reinterpret_cast<unsigned char*>(recvbuf.data()),
                  recvcounts_bytes.data(),
                  rdispls_bytes.data(),
                  MPI_BYTE,
                  comm);

        // -------------------------
        // (6) unpack into vector<vector<T>> grouped by source rank
        // -------------------------
        std::vector<std::vector<T>> recv(static_cast<std::size_t>(n_proc));
        for (int r = 0; r < n_proc; ++r)
        {
            const int cnt = recvcounts[r];
            recv[r].resize(static_cast<std::size_t>(cnt));

            if (cnt > 0)
            {
                const T *src = recvbuf.data() + rdispls[r];
                std::copy(src, src + cnt, recv[r].begin());
            }
        }
        return recv;
    }

    struct AllToAllStats
    {
        std::string label;

        int n_proc = 0;
        int my_rank = -1;

        // sent_to[j] = number of entries this rank sent to rank j
        std::vector<std::size_t> sent_to;

        // received_from[i] = number of entries this rank received from rank i
        std::vector<std::size_t> received_from;

        std::size_t total_sent = 0;
        std::size_t total_received = 0;

        void init(MPI_Comm comm, const std::string &lbl)
        {
            label = lbl;
            MPI_Comm_size(comm, &n_proc);
            MPI_Comm_rank(comm, &my_rank);
            sent_to.assign(n_proc, 0);
            received_from.assign(n_proc, 0);
            total_sent = total_received = 0;
        }
    };

    template <typename T>
    AllToAllStats compute_send_stats(const std::vector<std::vector<T>> &send_buffers,
                                     MPI_Comm comm,
                                     const std::string &label)
    {
        AllToAllStats s;
        s.init(comm, label);

        for (int j = 0; j < s.n_proc; ++j)
        {
            const std::size_t n = send_buffers[j].size();
            s.sent_to[j] = n;
            s.total_sent += n;
        }
        return s;
    }

    // Pack one Well10Entry into two uint64: [well, col]
    static inline void pack_10(const Well10Entry &e, std::uint64_t &a, std::uint64_t &b)
    {
        a = static_cast<std::uint64_t>(e.well);
        b = static_cast<std::uint64_t>(e.col);
    }

    static inline Well10Entry unpack_10(std::uint64_t a, std::uint64_t b, double v)
    {
        Well10Entry e;
        e.well = static_cast<well_t>(a);
        e.col  = static_cast<dof_t>(b);
        e.val  = v;
        return e;
    }

    static inline void mpi_check(int ierr, const char *msg)
    {
        if (ierr != MPI_SUCCESS) throw std::runtime_error(msg);
    }

    inline void send_receive_mnw_well_row_data(
        const std::vector<std::vector<Well10Entry>> &send_10,
        const std::vector<std::vector<Well11Entry>> &send_11,
        const std::vector<std::vector<WellRhsEntry>> &send_rhs1,
        std::vector<Well10Entry>  &recv_10,
        std::vector<Well11Entry>  &recv_11,
        std::vector<WellRhsEntry> &recv_rhs1,
        MPI_Comm comm,
        std::vector<int> *recvcounts_10_out = nullptr,
        std::vector<int> *recvcounts_11_out = nullptr,
        std::vector<int> *recvcounts_rhs1_out = nullptr)
    {
        int n_proc = 0;
        mpi_check(MPI_Comm_size(comm, &n_proc), "MPI_Comm_size failed");

        // ----------------------------
        // Helper: 2x uint64 indices + double values
        // pack_fn(e, a, b), unpack_fn(a, b, v) -> Entry
        // ----------------------------
        auto alltoallv_entries_2u64 =
            [&](auto pack_fn, auto unpack_fn,
                const auto &send_buckets,
                auto &recv_entries,
                std::vector<int> *recv_counts_out)
        {
            // counts in entries
            std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
            for (int p = 0; p < n_proc; ++p)
                send_counts[p] = static_cast<int>(send_buckets[p].size());

            mpi_check(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                                   recv_counts.data(), 1, MPI_INT,
                                   comm),
                      "MPI_Alltoall(counts) failed");
                // Export recv_counts (per-source counts for THIS rank)
                if (recv_counts_out)
                    *recv_counts_out = recv_counts;

            std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
            int total_send = 0, total_recv = 0;
            for (int p = 0; p < n_proc; ++p)
            {
                send_displs[p] = total_send;
                total_send += send_counts[p];

                recv_displs[p] = total_recv;
                total_recv += recv_counts[p];
            }

            // Flatten send buffers: indices (2*u64 per entry) + values (double)
            std::vector<std::uint64_t> send_idx(static_cast<std::size_t>(2) * total_send);
            std::vector<double>        send_val(static_cast<std::size_t>(total_send));

            {
                int cursor = 0;
                for (int dest = 0; dest < n_proc; ++dest)
                {
                    for (const auto &e : send_buckets[dest])
                    {
                        std::uint64_t a = 0, b = 0;
                        pack_fn(e, a, b);
                        send_idx[2 * cursor + 0] = a;
                        send_idx[2 * cursor + 1] = b;
                        send_val[cursor]         = e.val;
                        ++cursor;
                    }
                }
            }

            // Prepare recv buffers
            std::vector<std::uint64_t> recv_idx(static_cast<std::size_t>(2) * total_recv);
            std::vector<double>        recv_val(static_cast<std::size_t>(total_recv));

            // Scale counts/displs for idx
            std::vector<int> send_counts_i(n_proc), recv_counts_i(n_proc),
                             send_displs_i(n_proc), recv_displs_i(n_proc);
            for (int p = 0; p < n_proc; ++p)
            {
                send_counts_i[p] = 2 * send_counts[p];
                recv_counts_i[p] = 2 * recv_counts[p];
                send_displs_i[p] = 2 * send_displs[p];
                recv_displs_i[p] = 2 * recv_displs[p];
            }

            mpi_check(MPI_Alltoallv(send_idx.data(), send_counts_i.data(), send_displs_i.data(), MPI_UINT64_T,
                                    recv_idx.data(), recv_counts_i.data(), recv_displs_i.data(), MPI_UINT64_T,
                                    comm),
                      "MPI_Alltoallv(indices) failed (2u64)");

            mpi_check(MPI_Alltoallv(send_val.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                                    recv_val.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                                    comm),
                      "MPI_Alltoallv(values) failed (2u64)");

            // Unpack
            recv_entries.clear();
            recv_entries.reserve(static_cast<std::size_t>(total_recv));

            for (int k = 0; k < total_recv; ++k)
            {
                const auto a = recv_idx[2 * k + 0];
                const auto b = recv_idx[2 * k + 1];
                recv_entries.push_back(unpack_fn(a, b, recv_val[k]));
            }
        };

        // ----------------------------
        // Helper: 1x uint64 index + double values
        // pack_fn(e, a), unpack_fn(a, v) -> Entry
        // ----------------------------
        auto alltoallv_entries_1u64 =
            [&](auto pack_fn, auto unpack_fn,
                const auto &send_buckets,
                auto &recv_entries,
                std::vector<int> *recv_counts_out)
        {
            // counts in entries
            std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
            for (int p = 0; p < n_proc; ++p)
                send_counts[p] = static_cast<int>(send_buckets[p].size());

            mpi_check(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                                   recv_counts.data(), 1, MPI_INT,
                                   comm),
                      "MPI_Alltoall(counts) failed");

            // Export recv_counts (per-source counts for THIS rank)
            if (recv_counts_out)
                *recv_counts_out = recv_counts;

            std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
            int total_send = 0, total_recv = 0;
            for (int p = 0; p < n_proc; ++p)
            {
                send_displs[p] = total_send;
                total_send += send_counts[p];

                recv_displs[p] = total_recv;
                total_recv += recv_counts[p];
            }

            // Flatten send buffers: indices (1*u64 per entry) + values (double)
            std::vector<std::uint64_t> send_idx(static_cast<std::size_t>(total_send));
            std::vector<double>        send_val(static_cast<std::size_t>(total_send));

            {
                int cursor = 0;
                for (int dest = 0; dest < n_proc; ++dest)
                {
                    for (const auto &e : send_buckets[dest])
                    {
                        std::uint64_t a = 0;
                        pack_fn(e, a);
                        send_idx[cursor] = a;
                        send_val[cursor] = e.val;
                        ++cursor;
                    }
                }
            }

            // Prepare recv buffers
            std::vector<std::uint64_t> recv_idx(static_cast<std::size_t>(total_recv));
            std::vector<double>        recv_val(static_cast<std::size_t>(total_recv));

            // counts/displs for idx are the same as entry counts
            mpi_check(MPI_Alltoallv(send_idx.data(), send_counts.data(), send_displs.data(), MPI_UINT64_T,
                                    recv_idx.data(), recv_counts.data(), recv_displs.data(), MPI_UINT64_T,
                                    comm),
                      "MPI_Alltoallv(indices) failed (1u64)");

            mpi_check(MPI_Alltoallv(send_val.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                                    recv_val.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                                    comm),
                      "MPI_Alltoallv(values) failed (1u64)");

            // Unpack
            recv_entries.clear();
            recv_entries.reserve(static_cast<std::size_t>(total_recv));

            for (int k = 0; k < total_recv; ++k)
            {
                const auto a = recv_idx[k];
                recv_entries.push_back(unpack_fn(a, recv_val[k]));
            }
        };

        // ------------------------------------------------------------------
        // (1,0): Well10Entry  (row=well, col=trace/master) needs 2 u64
        // You must provide pack_10 / unpack_10 in your namespace or file.
        // ------------------------------------------------------------------
        alltoallv_entries_2u64(
            /*pack*/   [](const Well10Entry &e, std::uint64_t &a, std::uint64_t &b)
                      { pack_10(e, a, b); },
            /*unpack*/ [](std::uint64_t a, std::uint64_t b, double v)
                      { return unpack_10(a, b, v); },
            send_10, recv_10, recvcounts_10_out);

        // ------------------------------------------------------------------
        // (1,1) diagonal: Well11Entry (index = well) needs 1 u64
        // ------------------------------------------------------------------
        auto pack_11 = [](const Well11Entry &e, std::uint64_t &a)
        {
            a = static_cast<std::uint64_t>(e.well);
        };
        auto unpack_11 = [](std::uint64_t a, double v)
        {
            Well11Entry e;
            e.well = static_cast<well_t>(a);
            e.val  = v;
            return e;
        };
        alltoallv_entries_1u64(pack_11, unpack_11, send_11, recv_11, recvcounts_11_out);

        // ------------------------------------------------------------------
        // RHS(1): WellRhsEntry (index = well) needs 1 u64
        // ------------------------------------------------------------------
        auto pack_r = [](const WellRhsEntry &e, std::uint64_t &a)
        {
            a = static_cast<std::uint64_t>(e.well);
        };
        auto unpack_r = [](std::uint64_t a, double v)
        {
            WellRhsEntry e;
            e.well = static_cast<well_t>(a);
            e.val  = v;
            return e;
        };
        alltoallv_entries_1u64(pack_r, unpack_r, send_rhs1, recv_rhs1, recvcounts_rhs1_out);
    }

    inline void send_receive_trace01_data(
        const std::vector<std::vector<Trace01Entry>> &send_01,
        std::vector<Trace01Entry> &recv_01,
        MPI_Comm comm,
        std::vector<int> *recvcounts_out = nullptr)
    {
        int n_proc = 0;
        mpi_check(MPI_Comm_size(comm, &n_proc), "MPI_Comm_size failed");

        // counts in "entries"
        std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
        for (int p = 0; p < n_proc; ++p)
        {
            // Optional guard:
            AssertThrow(send_01[p].size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                        ExcMessage("send_01 bucket too large for MPI int counts"));
            send_counts[p] = static_cast<int>(send_01[p].size());
        }


        mpi_check(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                           recv_counts.data(), 1, MPI_INT,
                           comm),
              "MPI_Alltoall(counts) failed");

        if (recvcounts_out)
            *recvcounts_out = recv_counts;

        std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
        int total_send = 0, total_recv = 0;
        for (int p = 0; p < n_proc; ++p)
        {
            send_displs[p] = total_send;
            total_send += send_counts[p];

            recv_displs[p] = total_recv;
            total_recv += recv_counts[p];
        }

        // Optional fast-path:
        if (total_send == 0 && total_recv == 0) { recv_01.clear(); return; }

        // Pack indices as [row(uint64), col(uint64)] per entry, and values as double per entry
        std::vector<std::uint64_t> send_idx(static_cast<std::size_t>(2) * total_send);
        std::vector<double>        send_val(static_cast<std::size_t>(total_send));

        {
            int cursor = 0;
            for (int dest = 0; dest < n_proc; ++dest)
            {
                for (const auto &e : send_01[dest]) {
                    send_idx[2*cursor + 0] = static_cast<std::uint64_t>(e.row);
                    send_idx[2*cursor + 1] = static_cast<std::uint64_t>(e.col);
                    send_val[cursor]       = e.val;
                    ++cursor;
                }
            }
        }

        std::vector<std::uint64_t> recv_idx(static_cast<std::size_t>(2) * total_recv);
        std::vector<double>        recv_val(static_cast<std::size_t>(total_recv));

        // For idx arrays, counts/displs must be scaled by 2
        std::vector<int> send_counts2(n_proc), recv_counts2(n_proc),
                         send_displs2(n_proc), recv_displs2(n_proc);
        for (int p = 0; p < n_proc; ++p)
        {
            send_counts2[p] = 2 * send_counts[p];
            recv_counts2[p] = 2 * recv_counts[p];
            send_displs2[p] = 2 * send_displs[p];
            recv_displs2[p] = 2 * recv_displs[p];
        }

        mpi_check(MPI_Alltoallv(send_idx.data(), send_counts2.data(), send_displs2.data(), MPI_UINT64_T,
                            recv_idx.data(), recv_counts2.data(), recv_displs2.data(), MPI_UINT64_T,
                            comm),
              "MPI_Alltoallv(indices) failed");

        mpi_check(MPI_Alltoallv(send_val.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                                recv_val.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                                comm),
                  "MPI_Alltoallv(values) failed");

        // Unpack
        recv_01.clear();
        recv_01.reserve(static_cast<std::size_t>(total_recv));
        for (int k = 0; k < total_recv; ++k)
        {
            Trace01Entry e;
            e.row = static_cast<dof_t>(recv_idx[2*k + 0]);
            e.col = static_cast<well_t>(recv_idx[2*k + 1]);
            e.val = recv_val[k];
            recv_01.push_back(e);
        }
    }

    inline void fill_receive_stats_from_recvcounts(AllToAllStats &stats,
                                               const std::vector<int> &recvcounts)
    {
        // recvcounts[i] = how many elements I received from rank i
        stats.total_received = 0;
        for (int i = 0; i < stats.n_proc; ++i)
        {
            const std::size_t n = static_cast<std::size_t>(recvcounts[i]);
            stats.received_from[i] = n;
            stats.total_received += n;
        }
    }

    inline std::string partners_sent_str(const AllToAllStats &st)
    {
        std::ostringstream os;
        bool first = true;
        for (std::size_t j = 0; j < st.sent_to.size(); ++j)
        {
            if (st.sent_to[j] > 0)
            {
                if (!first) os << " ";
                os << "->" << j << ":" << st.sent_to[j];
                first = false;
            }
        }
        if (first) os << "-";
        return os.str();
    }

    inline std::string partners_recv_str(const AllToAllStats &st)
    {
        std::ostringstream os;
        bool first = true;
        for (std::size_t i = 0; i < st.received_from.size(); ++i)
        {
            if (st.received_from[i] > 0)
            {
                if (!first) os << " ";
                os << "<-" << i << ":" << st.received_from[i];
                first = false;
            }
        }
        if (first) os << "-";
        return os.str();
    }

    // Print one aligned line per (rank, exchange).
    inline void print_alltoall_stats_compact_aligned(const AllToAllStats &st,
                                                     MPI_Comm comm,
                                                     const int label_width = 8,
                                                     const int partner_width = 24)
    {
        int my_rank = -1, n_proc = -1;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &n_proc);

        // ordered printing by rank
        for (int r = 0; r < n_proc; ++r)
        {
            MPI_Barrier(comm);

            if (my_rank == r)
            {
                const std::string s_part = partners_sent_str(st);
                const std::string r_part = partners_recv_str(st);

                std::ostringstream os;
                os << "[" << std::left << std::setw(label_width) << st.label << "] "
                   << "rank=" << std::setw(3) << my_rank << " | "
                   << "sent tot=" << std::setw(8) << st.total_sent
                   << std::left << std::setw(partner_width) << ("  " + s_part) << " | "
                   << "recv tot=" << std::setw(8) << st.total_received
                   << std::left << std::setw(partner_width) << ("  " + r_part);

                std::cout << os.str() << std::endl;
            }

            MPI_Barrier(comm);
        }
    }

    inline void check_alltoall_consistency(const AllToAllStats &st, MPI_Comm comm)
    {
        const int n = st.n_proc;

        // Gather all sent_to vectors: matrix S where S[src][dst]
        std::vector<unsigned long long> local_sent(n);
        for (int j = 0; j < n; ++j)
            local_sent[j] = static_cast<unsigned long long>(st.sent_to[j]);

        std::vector<unsigned long long> all_sent(static_cast<std::size_t>(n) * n, 0ULL);

        MPI_Allgather(local_sent.data(), n, MPI_UNSIGNED_LONG_LONG,
                      all_sent.data(),   n, MPI_UNSIGNED_LONG_LONG,
                      comm);

        // Now check: received_from[src] == all_sent[src][my_rank]
        bool ok = true;
        for (int src = 0; src < n; ++src)
        {
            const unsigned long long expected =
                all_sent[static_cast<std::size_t>(src) * n + st.my_rank];
            const unsigned long long got =
                static_cast<unsigned long long>(st.received_from[src]);

            if (expected != got)
            {
                ok = false;
                std::cerr << "[" << st.label << "] Rank " << st.my_rank
                          << " mismatch: from src " << src
                          << " expected " << expected
                          << " got " << got << "\n";
            }
        }

        // Reduce to global result
        const int lok = ok ? 1 : 0;
        const int gok = Utilities::MPI::min(lok, comm); // 1 if all ok

        if (Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << "[" << st.label << "] all-to-all consistency: "
                      << (gok == 1 ? "OK" : "FAILED") << "\n";
    }



}

#endif //MPI_HELPERS_H

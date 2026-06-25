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

}

#endif //MPI_HELPERS_H

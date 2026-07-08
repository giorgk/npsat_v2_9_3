//
// Created by giorgk on 7/5/26.
//

#ifndef NPSAT_V2_STREAMLINE_MERGE_H
#define NPSAT_V2_STREAMLINE_MERGE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace npsat_trace
{
    struct StreamlineRecord
    {
        bool termination = false;
        double pid = 0.0;
        double Eid = 0.0;
        double Sid = 0.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double vmag = 0.0;
        int end_reason = 0;
    };

    inline std::string rank_iter_base_name(const std::string &prefix,
                                           const unsigned int rank,
                                           const unsigned int iter)
    {
        std::ostringstream ss;
        ss << prefix
           << "_streamlines_rank_" << std::setw(4) << std::setfill('0') << rank
           << "_iter_" << std::setw(4) << std::setfill('0') << iter;
        return ss.str();
    }

    inline std::string ordered_rank_iter_base_name(const std::string &prefix,
                                                   const unsigned int rank,
                                                   const unsigned int iter)
    {
        std::ostringstream ss;
        ss << prefix
           << "_streamlines_ordered_rank_" << std::setw(4) << std::setfill('0') << rank
           << "_iter_" << std::setw(4) << std::setfill('0') << iter;
        return ss.str();
    }

    inline StreamlineRecord make_streamline_record(const std::array<double, 7> &row)
    {
        StreamlineRecord rec;

        if (row[0] == -1.0)
        {
            rec.termination = true;
            rec.pid = row[1];
            rec.Eid = row[2];
            rec.Sid = row[3];
            rec.end_reason = static_cast<int>(std::llround(row[4]));
        }
        else
        {
            rec.pid = row[0];
            rec.Eid = row[1];
            rec.Sid = row[2];
            rec.x = row[3];
            rec.y = row[4];
            rec.z = row[5];
            rec.vmag = row[6];
        }

        return rec;
    }

    inline std::uint64_t integer_key_from_double(const double value)
    {
        return static_cast<std::uint64_t>(std::llround(value));
    }

    inline unsigned int streamline_owner_rank(const StreamlineRecord &rec,
                                              const unsigned int n_proc)
    {
        const std::uint64_t eid = integer_key_from_double(rec.Eid);
        const std::uint64_t sid = integer_key_from_double(rec.Sid);
        std::uint64_t h = eid;
        h ^= sid + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return static_cast<unsigned int>(h % n_proc);
    }

    inline bool streamline_record_less(const StreamlineRecord &a,
                                       const StreamlineRecord &b)
    {
        const std::uint64_t a_eid = integer_key_from_double(a.Eid);
        const std::uint64_t b_eid = integer_key_from_double(b.Eid);
        if (a_eid != b_eid) return a_eid < b_eid;

        const std::uint64_t a_sid = integer_key_from_double(a.Sid);
        const std::uint64_t b_sid = integer_key_from_double(b.Sid);
        if (a_sid != b_sid) return a_sid < b_sid;

        const std::uint64_t a_pid = integer_key_from_double(a.pid);
        const std::uint64_t b_pid = integer_key_from_double(b.pid);
        if (a_pid != b_pid) return a_pid < b_pid;

        return static_cast<int>(a.termination) < static_cast<int>(b.termination);
    }

    inline void read_streamline_records_binary(const std::string &filename,
                                               const unsigned int target_rank,
                                               const unsigned int n_proc,
                                               std::vector<StreamlineRecord> &records)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in.good())
            throw std::runtime_error("Could not open binary streamline file: " + filename);

        in.seekg(0, std::ios::end);
        const std::streamoff size = in.tellg();
        if (size < 0)
            throw std::runtime_error("Could not determine binary streamline file size: " + filename);
        if (size % static_cast<std::streamoff>(7 * sizeof(double)) != 0)
            throw std::runtime_error("Binary streamline file has incomplete records: " + filename);
        in.seekg(0, std::ios::beg);

        std::array<double, 7> row;
        while (in.read(reinterpret_cast<char *>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(double))))
        {
            StreamlineRecord rec = make_streamline_record(row);
            if (streamline_owner_rank(rec, n_proc) == target_rank)
                records.push_back(rec);
        }

        if (!in.eof())
            throw std::runtime_error("Failed while reading binary streamline file: " + filename);
    }

    inline void read_streamline_records_ascii(const std::string &filename,
                                              const unsigned int target_rank,
                                              const unsigned int n_proc,
                                              std::vector<StreamlineRecord> &records)
    {
        std::ifstream in(filename);
        if (!in.good())
            throw std::runtime_error("Could not open ASCII streamline file: " + filename);

        std::string line;
        std::uint64_t line_no = 0;
        while (std::getline(in, line))
        {
            ++line_no;
            if (is_comment_or_empty(line))
                continue;

            std::istringstream iss(line);
            std::array<double, 7> row;
            for (double &v : row)
            {
                if (!(iss >> v))
                    throw std::runtime_error("Malformed ASCII streamline record in " +
                                             filename + " at line " + std::to_string(line_no));
            }

            StreamlineRecord rec = make_streamline_record(row);
            if (streamline_owner_rank(rec, n_proc) == target_rank)
                records.push_back(rec);
        }

        if (!in.eof())
            throw std::runtime_error("Failed while reading ASCII streamline file: " + filename);
    }

    inline void write_streamline_record_ascii(std::ostream &out,
                                              const StreamlineRecord &rec)
    {
        if (rec.termination)
        {
            out << -1 << ' '
                << static_cast<long long>(std::llround(rec.pid)) << ' '
                << static_cast<long long>(std::llround(rec.Eid)) << ' '
                << static_cast<long long>(std::llround(rec.Sid)) << ' '
                << rec.end_reason << ' ' << 0 << ' ' << 0 << '\n';
        }
        else
        {
            out << static_cast<long long>(std::llround(rec.pid)) << ' '
                << static_cast<long long>(std::llround(rec.Eid)) << ' '
                << static_cast<long long>(std::llround(rec.Sid)) << ' '
                << rec.x << ' ' << rec.y << ' ' << rec.z << ' '
                << rec.vmag << '\n';
        }
    }

    inline void write_streamline_record_binary(std::ostream &out,
                                               const StreamlineRecord &rec)
    {
        std::array<double, 7> row;
        if (rec.termination)
        {
            row = {{-1.0, rec.pid, rec.Eid, rec.Sid,
                    static_cast<double>(rec.end_reason), 0.0, 0.0}};
        }
        else
        {
            row = {{rec.pid, rec.Eid, rec.Sid, rec.x, rec.y, rec.z, rec.vmag}};
        }

        out.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(double)));
        if (!out)
            throw std::runtime_error("Failed while writing binary ordered streamline record.");
    }

    inline void write_ordered_streamline_records(const std::string &base_name,
                                                 const bool write_ascii,
                                                 const bool write_bin,
                                                 const std::vector<StreamlineRecord> &records)
    {
        if (write_ascii)
        {
            const std::string filename = base_name + ".dat";
            std::ofstream out(filename, std::ios::trunc);
            if (!out.good())
                throw std::runtime_error("Could not open ordered ASCII streamline file: " + filename);

            for (const StreamlineRecord &rec : records)
                write_streamline_record_ascii(out, rec);
        }

        if (write_bin)
        {
            const std::string filename = base_name + ".bin";
            std::ofstream out(filename, std::ios::binary | std::ios::trunc);
            if (!out.good())
                throw std::runtime_error("Could not open ordered binary streamline file: " + filename);

            for (const StreamlineRecord &rec : records)
                write_streamline_record_binary(out, rec);
        }
    }

    inline void merge_streamline_iteration(const std::string &prefix,
                                           const unsigned int iter,
                                           const unsigned int n_proc,
                                           const unsigned int my_rank,
                                           const bool write_ascii,
                                           const bool write_bin,
                                           MPI_Comm mpi_communicator)
    {
        MPI_Barrier(mpi_communicator);

        std::vector<StreamlineRecord> records;
        const bool read_binary = write_bin;

        for (unsigned int rank = 0; rank < n_proc; ++rank)
        {
            const std::string input_base = rank_iter_base_name(prefix, rank, iter);
            if (read_binary)
                read_streamline_records_binary(input_base + ".bin", my_rank, n_proc, records);
            else
                read_streamline_records_ascii(input_base + ".dat", my_rank, n_proc, records);
        }

        std::sort(records.begin(), records.end(), streamline_record_less);
        const std::string output_base = ordered_rank_iter_base_name(prefix, my_rank, iter);
        write_ordered_streamline_records(output_base, write_ascii, write_bin, records);

        MPI_Barrier(mpi_communicator);
    }


}

#endif //NPSAT_V2_STREAMLINE_MERGE_H

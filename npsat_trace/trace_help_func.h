//
// Created by giorgk on 6/27/26.
//

#ifndef TRACE_HELP_FUNC_H
#define TRACE_HELP_FUNC_H

#include <istream>
#include <stdexcept>

namespace npsat_trace {

    // ----------------------------
    // Helper: POD binary IO
    // ----------------------------
    template <class T>
    inline void read_pod(std::istream &in, T &v)
    {
        static_assert(std::is_trivially_copyable<T>::value,
                      "read_pod requires trivially copyable type");

        in.read(reinterpret_cast<char *>(&v),
                static_cast<std::streamsize>(sizeof(T)));

        if (!in)
            throw std::runtime_error("Binary read failed while reading POD value.");
    }

    inline std::string read_string(std::istream &in)
    {
        // Must match write_string:
        // uint32 length, then raw bytes, no null terminator.
        std::uint32_t n = 0;
        read_pod(in, n);

        std::string s;
        s.resize(n);

        if (n > 0)
        {
            in.read(&s[0], static_cast<std::streamsize>(n));

            if (!in)
                throw std::runtime_error("Binary read failed while reading string bytes.");
        }

        return s;
    }

    inline std::vector<double> read_delta_time_file(const std::string &filename)
    {
        std::ifstream in(filename);
        if (!in.good())
            throw std::runtime_error("Could not open delta time file: " + filename);

        std::vector<double> delta_times;
        double dt = 0.0;
        while (in >> dt)
        {
            if (!std::isfinite(dt) || dt <= 0.0)
                throw std::runtime_error("Invalid delta time value in file: " + filename);

            delta_times.push_back(dt);
        }

        if (!in.eof())
            throw std::runtime_error("Failed reading delta time file: " + filename);

        if (delta_times.empty())
            throw std::runtime_error("Delta time file is empty: " + filename);

        return delta_times;
    }

}

#endif //TRACE_HELP_FUNC_H

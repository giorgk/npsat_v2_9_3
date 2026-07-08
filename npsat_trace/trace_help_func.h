//
// Created by giorgk on 6/27/26.
//

#ifndef TRACE_HELP_FUNC_H
#define TRACE_HELP_FUNC_H

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <type_traits>
#include "trace_structures.h"

namespace npsat_trace {
    using namespace  dealii;

    // ----------------------------
    // Helper: POD binary IO
    // ----------------------------
    template <class T>
    inline void write_pod(std::ostream &out, const T &v)
    {
        static_assert(std::is_trivially_copyable<T>::value,
                      "write_pod requires trivially copyable type");

        out.write(reinterpret_cast<const char *>(&v),
                  static_cast<std::streamsize>(sizeof(T)));

        if (!out)
            throw std::runtime_error("Binary write failed while writing POD value.");
    }

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

    inline double coarse_streamline_coord(const double x)
    {
        return std::round(10.0 * x);
    }

    template <int dim, class Properties>
    inline bool update_streamline_bbox(Properties props, const Point<dim> &x)
    {
        bool expanded = false;
        const double qx = coarse_streamline_coord(x[0]);
        const double qy = coarse_streamline_coord(x[1]);
        const double qz = coarse_streamline_coord(x[2]);

        if (qx < props[pBBoxMinX]) { props[pBBoxMinX] = qx; expanded = true; }
        if (qy < props[pBBoxMinY]) { props[pBBoxMinY] = qy; expanded = true; }
        if (qz < props[pBBoxMinZ]) { props[pBBoxMinZ] = qz; expanded = true; }
        if (qx > props[pBBoxMaxX]) { props[pBBoxMaxX] = qx; expanded = true; }
        if (qy > props[pBBoxMaxY]) { props[pBBoxMaxY] = qy; expanded = true; }
        if (qz > props[pBBoxMaxZ]) { props[pBBoxMaxZ] = qz; expanded = true; }

        props[pNoExpandCount] = expanded ? 0.0 : props[pNoExpandCount] + 1.0;
        return expanded;
    }

    template <class Properties>
    inline void increment_streamline_nonexpansion(Properties props)
    {
        props[pNoExpandCount] += 1.0;
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

    struct StreamlineOutput
    {
        std::ofstream ascii;
        std::ofstream bin;
        bool write_ascii = true;
        bool write_bin = false;

        StreamlineOutput(const std::string &base_name,
                         const bool write_ascii_in,
                         const bool write_bin_in)
            : write_ascii(write_ascii_in)
            , write_bin(write_bin_in)
        {
            if (write_ascii)
            {
                const std::string fname = base_name + ".dat";
                ascii.open(fname, std::ios::trunc);
                if (!ascii.good())
                    throw std::runtime_error("Could not open ASCII streamline output: " + fname);
            }

            if (write_bin)
            {
                const std::string fname = base_name + ".bin";
                bin.open(fname, std::ios::binary | std::ios::trunc);
                if (!bin.good())
                    throw std::runtime_error("Could not open binary streamline output: " + fname);
            }
        }

        void close()
        {
            if (ascii.is_open())
                ascii.close();
            if (bin.is_open())
                bin.close();
        }
    };

    struct BinaryStreamlineSample
    {
        std::uint64_t pid = 0;
        std::uint64_t Eid = 0;
        std::uint64_t Sid = 0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double vmag = 0.0;
    };

    struct BinaryStreamlineTrajectory
    {
        std::uint64_t Eid = 0;
        std::uint64_t Sid = 0;
        std::uint64_t termination_pid = 0;
        int end_reason = 0;
        bool has_termination = false;
        std::vector<BinaryStreamlineSample> samples;

        void clear()
        {
            Eid = 0;
            Sid = 0;
            termination_pid = 0;
            end_reason = 0;
            has_termination = false;
            samples.clear();
        }
    };

    inline std::uint64_t streamline_binary_id(const double value)
    {
        return static_cast<std::uint64_t>(std::llround(value));
    }

    inline bool read_binary_streamline_row(std::istream &in,
                                           std::array<double, 7> &row,
                                           const std::string &filename)
    {
        in.read(reinterpret_cast<char *>(row.data()),
                static_cast<std::streamsize>(row.size() * sizeof(double)));

        if (in)
            return true;

        if (in.eof() && in.gcount() == 0)
            return false;

        throw std::runtime_error("Incomplete binary streamline record in: " + filename);
    }

    inline bool binary_streamline_row_is_termination(const std::array<double, 7> &row)
    {
        return row[0] == -1.0;
    }

    inline void append_binary_streamline_sample(BinaryStreamlineTrajectory &trajectory,
                                                const std::array<double, 7> &row)
    {
        BinaryStreamlineSample sample;
        sample.pid = streamline_binary_id(row[0]);
        sample.Eid = streamline_binary_id(row[1]);
        sample.Sid = streamline_binary_id(row[2]);
        sample.x = row[3];
        sample.y = row[4];
        sample.z = row[5];
        sample.vmag = row[6];

        if (trajectory.samples.empty())
        {
            trajectory.Eid = sample.Eid;
            trajectory.Sid = sample.Sid;
        }
        else if (trajectory.Eid != sample.Eid || trajectory.Sid != sample.Sid)
        {
            throw std::runtime_error("Binary streamline file changed Eid/Sid before a termination record.");
        }

        trajectory.samples.push_back(sample);
    }

    inline void set_binary_streamline_termination(BinaryStreamlineTrajectory &trajectory,
                                                  const std::array<double, 7> &row)
    {
        trajectory.has_termination = true;
        trajectory.termination_pid = streamline_binary_id(row[1]);
        trajectory.Eid = streamline_binary_id(row[2]);
        trajectory.Sid = streamline_binary_id(row[3]);
        trajectory.end_reason = static_cast<int>(std::llround(row[4]));
    }

    inline bool read_next_binary_streamline(std::istream &in,
                                            BinaryStreamlineTrajectory &trajectory,
                                            const std::string &filename)
    {
        trajectory.clear();

        std::array<double, 7> row;
        while (read_binary_streamline_row(in, row, filename))
        {
            if (binary_streamline_row_is_termination(row))
            {
                set_binary_streamline_termination(trajectory, row);
                return true;
            }

            append_binary_streamline_sample(trajectory, row);
        }

        if (!trajectory.samples.empty())
            throw std::runtime_error("Binary streamline ended before its termination record in: " + filename);

        return false;
    }

    inline void placeholder_process_binary_streamline(const BinaryStreamlineTrajectory &trajectory)
    {
        // Replace this with application-specific processing.
        (void)trajectory;
    }

    template <class ProcessTrajectory>
    inline void process_binary_streamline_file(const std::string &filename,
                                               ProcessTrajectory process_trajectory)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in.good())
            throw std::runtime_error("Could not open binary streamline file: " + filename);

        BinaryStreamlineTrajectory trajectory;
        while (read_next_binary_streamline(in, trajectory, filename))
            process_trajectory(trajectory);
    }

    inline void process_binary_streamline_file_with_placeholder(const std::string &filename)
    {
        process_binary_streamline_file(filename, placeholder_process_binary_streamline);
    }

    inline bool is_comment_or_empty(const std::string &line)
    {
        auto it = std::find_if_not(line.begin(), line.end(),
                                   [](unsigned char c){ return std::isspace(c); });
        if (it == line.end()) return true;
        return (*it == '#');
    }

    inline void normalize_separators(std::string &s)
    {
        for (char &c : s)
            if (c == ',' || c == ';' || c == '\t') c = ' ';
    }

    inline std::vector<std::string> split_tokens(const std::string &line)
    {
        std::string s = line;
        normalize_separators(s);
        std::istringstream iss(s);
        std::vector<std::string> toks;
        std::string t;
        while (iss >> t) toks.push_back(t);
        return toks;
    }

    static void write_termination(StreamlineOutput &out, const double pid, const double Eid, const double Sid, const int er) {
        if (out.write_ascii)
        {
            out.ascii << -1 << ' '
                << static_cast<long long>(pid) << ' '
                << static_cast<long long>(Eid) << ' '
                << static_cast<long long>(Sid) << ' '
                << er << ' ' << 0 << ' ' << 0 << '\n';
        }

        if (out.write_bin)
        {
            const double record[7] = {
                -1.0,
                pid,
                Eid,
                Sid,
                static_cast<double>(er),
                0.0,
                0.0
            };
            out.bin.write(reinterpret_cast<const char *>(record),
                          static_cast<std::streamsize>(sizeof(record)));
            if (!out.bin)
                throw std::runtime_error("Binary write failed while writing streamline termination record.");
        }
    }

    template <typename T>
    inline T clamp_(T value, T lo, T hi)
    {
        assert(lo <= hi);
        return std::max(lo, std::min(value, hi));
    }

    static void write_sample(StreamlineOutput &out,  const double pid, const double Eid, const double Sid,
                             const dealii::Point<3> &x, const double vmag)
    {
        if (out.write_ascii)
        {
            out.ascii << static_cast<long long>(pid) << ' '
                << static_cast<long long>(Eid) << ' '
                << static_cast<long long>(Sid) << ' '
                << x[0] << ' ' << x[1] << ' ' << x[2] << ' '
                << vmag << '\n';
        }

        if (out.write_bin)
        {
            const double record[7] = {
                pid,
                Eid,
                Sid,
                x[0],
                x[1],
                x[2],
                vmag
            };
            out.bin.write(reinterpret_cast<const char *>(record),
                          static_cast<std::streamsize>(sizeof(record)));
            if (!out.bin)
                throw std::runtime_error("Binary write failed while writing streamline sample record.");
        }
    }

    template <int dim>
    static double compute_step_size_ds(const double diameter, const double vmag, const double dt_remaining,
        const double delta_time, const TimeStepControl tsc, bool &is_stuck_out) {
        AssertThrow(diameter > 0.0, dealii::ExcMessage("Cell diameter() is zero."));

        // distance constraints
        const double ds1 = tsc.max_step;
        const double ds2 = diameter / std::max(1u, tsc.n_steps_per_cell);

        // time constraints converted to distance
        const double ds3 = vmag * tsc.max_step_time;
        const double dt_sub = delta_time / std::max(1u, tsc.n_steps_per_time);
        const double ds4 = vmag * dt_sub;

        double ds = std::min(std::min(ds1, ds2), std::min(ds3, ds4));

        ds = std::min(ds, vmag * dt_remaining);

        // stuck detection threshold (scale with cell size)
        const double ds_min = 1e-14 * std::max(diameter, 1.0);
        is_stuck_out = (ds < ds_min);

        return ds;
    }

    template <int dim, typename CellIterator>
    bool check_cell_point(CellIterator &cell, const Point<dim> &p) {
        using CellIt2 = typename std::decay<CellIterator>::type;

        // Level 0: Check the current cell itself
        if (cell->point_inside(p)) {
            return true;
        }

        // Breadth-First Search (BFS) setup
        // visited: prevents circular re-checking of the same cells
        // current_layer: cells at the current "distance" from the start
        std::set<CellIt2> visited;
        std::vector<CellIt2> current_layer;
        std::vector<CellIt2> next_layer;

        visited.insert(cell);
        current_layer.push_back(cell);

        int nSearch = 0;
        while (nSearch < 3 /* //TODO make this user parameter*/) {
            next_layer.clear();
            for (auto& active_cell : current_layer) {
                // Check all faces of the cells in the current layer
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                    if (active_cell->at_boundary(f))
                        continue;

                    auto neighbor = active_cell->neighbor(f);

                    // Buffer to hold active cells (handles both active neighbors and AMR children)
                    std::vector<CellIt2> candidates;
                    if (neighbor->is_active()) {
                        candidates.push_back(neighbor);
                    }
                    else {
                        // Neighbor is refined. Collect the active neighbor children touching this face.
                        for (unsigned int sf = 0; sf < dealii::GeometryInfo<dim>::max_children_per_face; ++sf) {
                            // This returns an *active* neighbor child on this subface.
                            auto neigh_child = active_cell->neighbor_child_on_subface(f, sf);
                            // neighbor_child_on_subface() can return an invalid iterator in some edge cases
                            // (depending on AMR configuration), so check validity if your iterator supports it.
                            if (neigh_child.state() == dealii::IteratorState::valid)
                                candidates.push_back(neigh_child);
                        }
                    }

                    for (auto& cand : candidates) {
                        // Skip if we've already checked this cell or if it's artificial
                        // (Artificial cells don't have vertex data on this rank)
                        if (visited.find(cand) != visited.end() || cand->is_artificial()) {
                            continue;
                        }

                        // Perform geometric check
                        if (cand->point_inside(p)) {
                            // Point found in a locally owned or ghost cell
                            cell = cand;
                            //std::cout << "Found point in cell " << cell->id().to_string() << std::endl;
                            return true;
                        }

                        // If not found, mark as visited and prepare for the next layer expansion
                        visited.insert(cand);
                        next_layer.push_back(cand);
                    }
                }
            }

            // Advance to the next topological layer
            current_layer = next_layer;
            if (current_layer.empty())
                break; // No more neighbors to check

            nSearch++;
        }
        return false;
    }


    inline BilinearMapCoefficients build_bilinear_map_coefficients(
        const std::array<double,4> &xv, const std::array<double,4> &yv)
    {
        // Convert deal.II vertex order {0,1,2,3} = BL, BR, TL, TR
        // to Q1 order {0,1,2,3} = BL, BR, TR, TL
        const double x0 = xv[0], y0 = yv[0];
        const double x1 = xv[1], y1 = yv[1];
        const double x2 = xv[3], y2 = yv[3]; // swapped
        const double x3 = xv[2], y3 = yv[2]; // swapped

        BilinearMapCoefficients c;
        c.a0 = 0.25 * (x0 + x1 + x2 + x3);
        c.a1 = 0.25 * (-x0 + x1 + x2 - x3);
        c.a2 = 0.25 * (-x0 - x1 + x2 + x3);
        c.a3 = 0.25 * (x0 - x1 + x2 - x3);

        c.b0 = 0.25 * (y0 + y1 + y2 + y3);
        c.b1 = 0.25 * (-y0 + y1 + y2 - y3);
        c.b2 = 0.25 * (-y0 - y1 + y2 + y3);
        c.b3 = 0.25 * (y0 - y1 + y2 - y3);

        return c;
    }

    inline bool getUV_Analytical(double &u, double &v, double x, double y,
                                 const BilinearMapCoefficients &c) {
        const double a0 = c.a0, a1 = c.a1, a2 = c.a2, a3 = c.a3;
        const double b0 = c.b0, b1 = c.b1, b2 = c.b2, b3 = c.b3;

        // Set up Quadratic: A*v^2 + B*v + C = 0
        double dx = x - a0;
        double dy = y - b0;
        // If the cell is effectively affine/parallelogram, solve directly:
        if (std::abs(a3) < 1e-14 && std::abs(b3) < 1e-14) {
            const double det = a1*b2 - a2*b1;
            if (std::abs(det) < 1e-20)
                return false;

            u = ( dx*b2 - dy*a2) / det;
            v = (-dx*b1 + dy*a1) / det;
            return std::isfinite(u) && std::isfinite(v);
        }

        // General bilinear case: eliminate u and solve quadratic for v
        const double A = a3 * b2 - a2 * b3;
        const double B = dx * b3 - a2 * b1 - dy * a3 + b2 * a1;
        const double C = dx * b1 - dy * a1;
        double v1, v2;
        // Solve for v
        if (std::abs(A) < 1e-14) {
            // Degenerates to linear (parallelogram)
            // Linear case (parallelogram-ish)
            if (std::abs(B) < 1e-20)
                return false;
            v1 = -C / B;
            v2 = v1;
        }
        else {
            const double disc = B*B - 4.0*A*C;
            if (disc < 0.0)
                return false;
            const double sq = std::sqrt(disc);
            v1 = (-B + sq) / (2.0*A);
            v2 = (-B - sq) / (2.0*A);
        }

        auto recover_u = [&](double vv, double &uu) -> bool
        {
            const double d1 = a1 + a3*vv;
            const double d2 = b1 + b3*vv;

            if (std::abs(d1) > 1e-20)
            {
                uu = (dx - a2*vv) / d1;
                return std::isfinite(uu);
            }
            if (std::abs(d2) > 1e-20)
            {
                uu = (dy - b2*vv) / d2;
                return std::isfinite(uu);
            }
            return false;
        };

        double u1 = 0.0, u2 = 0.0;
        const bool ok1 = recover_u(v1, u1);
        const bool ok2 = recover_u(v2, u2);

        if (!ok1 && !ok2)
            return false;

        auto score = [](double uu, double vv)
        {
            auto s1 = (uu < -1.0 ? -1.0 - uu : (uu > 1.0 ? uu - 1.0 : 0.0));
            auto s2 = (vv < -1.0 ? -1.0 - vv : (vv > 1.0 ? vv - 1.0 : 0.0));
            return s1 + s2;
        };

        if (ok1 && (!ok2 || score(u1, v1) <= score(u2, v2)))
        {
            u = u1;
            v = v1;
        }
        else
        {
            u = u2;
            v = v2;
        }

        return std::isfinite(u) && std::isfinite(v);
    }

    inline bool getUV_Analytical(double &u, double &v, double x, double y,
                                 const std::array<double,4> &xv, const std::array<double,4> &yv) {
        return getUV_Analytical(u, v, x, y, build_bilinear_map_coefficients(xv, yv));
    }

    inline bool getUV_NewtonRaphson(double &u, double &v,
                                    double tx, double ty,
                                    const std::array<double,4> &xv,
                                    const std::array<double,4> &yv,
                                    NewtonDebugInfo *dbg)
    {
        if (dbg)
        {
            dbg->failure   = NewtonFailure::None;
            dbg->iterations = 0;
            dbg->u = 0.0;
            dbg->v = 0.0;
            dbg->detJ = 0.0;
            dbg->residual = 0.0;
            dbg->curX = 0.0;
            dbg->curY = 0.0;
        }

        // swap indices 2 and 3
        const double x0 = xv[0], y0 = yv[0];
        const double x1 = xv[1], y1 = yv[1];
        const double x2 = xv[3], y2 = yv[3];
        const double x3 = xv[2], y3 = yv[2];

        u = 0.0;
        v = 0.0;

        const int max_iter = 12;
        const double tol = 1e-10;

        for (int i=0; i<max_iter; ++i)
        {
            if (dbg)
                dbg->iterations = i+1;

            const double N0 = 0.25*(1-u)*(1-v);
            const double N1 = 0.25*(1+u)*(1-v);
            const double N2 = 0.25*(1+u)*(1+v);
            const double N3 = 0.25*(1-u)*(1+v);

            const double curX = N0*x0 + N1*x1 + N2*x2 + N3*x3;
            const double curY = N0*y0 + N1*y1 + N2*y2 + N3*y3;

            if (dbg)
            {
                dbg->curX = curX;
                dbg->curY = curY;
            }

            const double rx = tx-curX;
            const double ry = ty-curY;

            const double res = std::sqrt(rx*rx+ry*ry);

            if (dbg)
            {
                dbg->u = u;
                dbg->v = v;
                dbg->residual = res;
            }

            if (res < tol)
                return true;

            const double dxdu =
                0.25 * (-(1-v)*x0 + (1-v)*x1 + (1+v)*x2 - (1+v)*x3);

            const double dxdv =
                0.25 * (-(1-u)*x0 - (1+u)*x1 + (1+u)*x2 + (1-u)*x3);

            const double dydu =
                0.25 * (-(1-v)*y0 + (1-v)*y1 + (1+v)*y2 - (1+v)*y3);

            const double dydv =
                0.25 * (-(1-u)*y0 - (1+u)*y1 + (1+u)*y2 + (1-u)*y3);

            const double detJ = dxdu*dydv - dxdv*dydu;

            if (dbg)
                dbg->detJ = detJ;

            if (std::abs(detJ) < 1e-20)
            {
                if (dbg)
                    dbg->failure = NewtonFailure::SingularJacobian;

                return false;
            }

            const double du = ( dydv*rx - dxdv*ry)/detJ;
            const double dv = (-dydu*rx + dxdu*ry)/detJ;

            u += du;
            v += dv;

            if (dbg)
            {
                dbg->u = u;
                dbg->v = v;
            }

            if (!std::isfinite(u) || !std::isfinite(v))
            {
                if (dbg)
                    dbg->failure = NewtonFailure::NanIterate;

                return false;
            }
        }

        if (dbg)
            dbg->failure = NewtonFailure::MaxIterations;

        return false;

    }


    inline FindHexExitResult find_hex_exit(const Point<3> &l0, const Point<3> &l1,
              const std::array<double, 4> &xv, const std::array<double, 4> &yv,
              const std::array<double, 4> &zb, const std::array<double, 4> &zt) {

        ExitResult result;

        const double eps = 1e-12;
        const Tensor<1,3> dir = l1 - l0;

        const Point<2> p0(l0[0], l0[1]);
        const Point<2> p1(l1[0], l1[1]);
        Tensor<1,2> d2 = p1 - p0;

        static const int edge_a[4] = {0, 1, 3, 2};
        static const int edge_b[4] = {1, 3, 2, 0};

        for (int i = 0; i < 4; ++i) {
            const int a = edge_a[i];
            const int b = edge_b[i];

            Point<2> e0(xv[a], yv[a]);
            Point<2> e1(xv[b], yv[b]);
            Tensor<1,2> e = e1 - e0;

            const double det = d2[0] * (-e[1]) - d2[1] * (-e[0]);

            if (std::abs(det) < eps)
                continue;

            Tensor<1,2> w = e0 - p0;

            const double t = (w[0] * (-e[1]) - w[1] * (-e[0])) / det;
            const double s = (d2[0] * w[1] - d2[1] * w[0]) / det;

            if (t < -eps || t > 1.0 + eps)
                continue;

            if (s < -eps || s > 1.0 + eps)
                continue;

            Point<3> pint = l0 + dir * t;

            const double zbot = (1.0 - s) * zb[a] + s * zb[b];
            const double ztop = (1.0 - s) * zt[a] + s * zt[b];

            if (pint[2] >= zbot - eps && pint[2] <= ztop + eps) {
                result.face_index = 2;
                result.intersection_point = pint;
                return FindHexExitResult(result);
            }
        }

        if (std::abs(dir[2]) > eps) {
            const bool moving_up = (dir[2] > 0.0);
            const int face_id = moving_up ? 5 : 4;
            const std::array<double,4> &zarr = moving_up ? zt : zb;

            double u0 = 0.0, v0 = 0.0;
            if (!getUV_Analytical(u0, v0, l0[0], l0[1], xv, yv))
                return FindHexExitResult();

            const double N0 = 0.25 * (1.0 - u0) * (1.0 - v0);
            const double N1 = 0.25 * (1.0 + u0) * (1.0 - v0);
            const double N2 = 0.25 * (1.0 + u0) * (1.0 + v0);
            const double N3 = 0.25 * (1.0 - u0) * (1.0 + v0);

            const double z_lid_at_l0 =
                N0 * zarr[0] + N1 * zarr[1] + N2 * zarr[3] + N3 * zarr[2];

            if (moving_up) {
                if (l1[2] >= z_lid_at_l0 - eps) {
                    Point<3> pint(l0[0], l0[1], z_lid_at_l0);
                    result.face_index = face_id;
                    result.intersection_point = pint;
                    return FindHexExitResult(result);
                }
            }
            else {
                if (l1[2] <= z_lid_at_l0 + eps) {
                    Point<3> pint(l0[0], l0[1], z_lid_at_l0);
                    result.face_index = face_id;
                    result.intersection_point = pint;
                    return FindHexExitResult(result);
                }
            }
        }
        return FindHexExitResult();
    }





}

#endif //TRACE_HELP_FUNC_H

//
// Created by giorgk on 6/24/26.
//

#ifndef MNWELLS_H
#define MNWELLS_H
#include <deal.II/base/point.h>
#include <deal.II/grid/tria_description.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "TimeSeries.h"
#include "flow_structures.h"
#include "helper_func.h"

namespace npsat_flow {
    using namespace dealii;

    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;
    using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
    using WellValue = std::pair<Point2D, std::size_t>;
    using Polygon = bg::model::polygon<Point2D>;

    struct WellSegmentOut
    {
        // Geometry (each segment is vertical at x,y)
        double x, y;
        double z_bot, z_top;

        // Identifiers
        int Eid;                 // IMPORTANT: well identifier for output
        std::uint32_t cell_id;   // groundwater cell id
        double ze;               // your segment elevation key

        // Flux/diagnostics (same as CSV columns)
        double Qe;
        double cwc;
        double Qwbf_top;
        double Qtot;
        double F_top_well;
        double diff;
    };

    template<int dim>
    struct WellData{
        unsigned int id;
        Point<dim> location_top;      // Well top (X, Y, Ztop)
        Point<dim> location_bottom;   // Well bottom (X, Y, Zbot)
        double screen_length;         // Ztop - Zbot
        //unsigned int local_cell_index; // Which cell contains this well segment
        std::vector<double> pumping_rates; // Time-dependent rates (m³/day per meter)
        double well_radius;
        std::vector<double> well_rates;
        std::vector<types::global_dof_index> trace_dof_indices;

        // Get current pumping rate
        double get_pumping_rate(double time, double time_step) const
        {
            if (pumping_rates.empty())
                return 0.0;

            // Find time step index
            unsigned int step = static_cast<unsigned int>(time / time_step);

            if (step >= pumping_rates.size())
                step = step % pumping_rates.size(); // Repeat pattern

            return pumping_rates[step];
        }
    };

    struct CellWellLink {
        unsigned int well_global_index; // Points to wells[n] in your MNWellCollection
        unsigned int well_owner_rank;
        double cwc = 0.0;                     // Cell-specific well constant
        double cwc_eff = 0.0;                 // Nonlinear wet-screen-adjusted conductance
        double sl = 0.0; //screen length
        double Qe = 0.0; // Q exchange between well and aquifer
        double Qwbf = 0.0; // Wellbore flow towards the top of this cell
        double ze = 0.0; // This is the elevation of the center of the cell
        double h_e = 0.0; //TODO remove this for debuging only
        double w_zbot = 0.0;// Bottom elevation of well
        std::vector<types::global_dof_index> trace_dof_indices; // Unique to this cell/well pair

        // operator for sorting
        bool operator<(const CellWellLink& other) const {
            return well_global_index < other.well_global_index;
        }
    };

    inline double effective_well_link_conductance(const CellWellLink &link,
                                                  const double h_e,
                                                  const bool confined,
                                                  double *wet_screen_length = nullptr,
                                                  const double tiny_cwc_fraction = 1.0e-12)
    {
        if (confined)
        {
            if (wet_screen_length != nullptr)
                *wet_screen_length = link.sl;
            return link.cwc;
        }

        const double screen_bot = link.w_zbot;
        const double screen_top = link.w_zbot + link.sl;
        const double wet_top = std::min(screen_top, h_e);
        const double wet_length = std::max(0.0, wet_top - screen_bot);

        if (wet_screen_length != nullptr)
            *wet_screen_length = wet_length;

        const double wet_fraction =
            (link.sl > 0.0)
            ? std::min(1.0, wet_length / link.sl)
            : 0.0;

        return link.cwc * std::max(wet_fraction, tiny_cwc_fraction);
    }

    struct MNWell
    {
        int Eid;
        double x;
        double y;
        double top;
        double bottom;
        //double Q;
        int q_row;
        double rw;
        double Rskin;
        double Kskin;
        std::vector<types::global_dof_index> trace_dof_indices;
        unsigned int global_index;
    };

    // A POD packet of just the fields that are populated from file.
    struct MNWellPacket
    {
        int     Eid;
        double  x, y;
        double  top, bottom;
        int  q_row;
        double  rw, Rskin, Kskin;
        std::uint32_t global_index;
    };

    static_assert(std::is_trivially_copyable<MNWellPacket>::value,
                "MNWellPacket must be trivially copyable for MPI_Bcast.");

    struct MNWellCollection{
        std::vector<MNWell> wells;
        bgi::rtree<WellValue, bgi::quadratic<16>> well_rtree;
        TimeSeriesData<double> Q_ts;

        void set_time_step_number(const int time_step_number_in)
        {
            time_step_number = time_step_number_in;
        }

        int get_time_step_number() const
        {
            return time_step_number;
        }

        double pumping_rate(const int q_row) const
        {
            return Q_ts.get_value_at_step(q_row, time_step_number);
        }

        void find_wells_in_polygon(const Polygon& quad,
            std::vector<MNWell*>& wells_inside,
            std::vector<double>& screen_length_inside,
            std::vector<double>& well_bot,
            std::vector<double>& X, std::vector<double>& Y,
            std::vector<double>& Ztop, std::vector<double>& Zbot)
        {
            wells_inside.clear();
            bg::model::box<Point2D> bbox;
            bg::envelope(quad, bbox);
            std::vector<WellValue> candidates;
            well_rtree.query(bgi::intersects(bbox), std::back_inserter(candidates));

            Point2D p;
            for (auto& c : candidates)
            {
                MNWell& w = wells[c.second];

                // --- Cheap bounding-box precheck (fast reject) ---
                if (w.x < bbox.min_corner().get<0>() || w.x > bbox.max_corner().get<0>() ||
                    w.y < bbox.min_corner().get<1>() || w.y > bbox.max_corner().get<1>()) {
                    continue; // definitely outside, skip expensive geometry
                    }
                p.set<0>(w.x);
                p.set<1>(w.y);
                //bg::set<0>(p, w.x);
                //bg::set<1>(p, w.y);
                if (bg::within(p, quad))
                {
                    double p_top, p_bot;
                    interpolate_top_bottom(X, Y, Ztop, Zbot, w.x, w.y, p_top, p_bot);
                    if (w.top >= p_bot && p_top >= w.bottom)
                    {
                        ScreenClip clip = well_length_inside_cell(w.bottom, w.top, p_bot, p_top);
                        if (clip.L > 0.5)
                        {// We add this only if there is at least half meter well screen in the cell
                            screen_length_inside.push_back(clip.L);
                            wells_inside.push_back(&w);
                            well_bot.push_back(clip.z_low);
                        }
                    }
                }
            }
        }

        int time_step_number = 0;

        /**
         * @brief Read multi-node well data from an ASCII file.
         *
         * Reads well geometry and skin properties from a delimited text file and
         * constructs both the internal well container and the spatial R-tree index.
         *
         * The first line of the file is assumed to be a header and is skipped.
         *
         * Expected columns:
         *
         * | Column      | Type   | Description |
         * |-------------|--------|-------------|
         * | Eid         | int    | Element or well identifier |
         * | x           | double | X coordinate |
         * | y           | double | Y coordinate |
         * | top         | double | Top elevation of screened interval |
         * | bottom      | double | Bottom elevation of screened interval |
         * | q_row       | int    | Row index in pumping-rate table |
         * | rw          | double | Well radius |
         * | Rskin       | double | Skin radius |
         * | Kskin       | double | Skin hydraulic conductivity |
         *
         * Example input file:
         * @code
         * Eid,x,y,top,bottom,q_row,rw,Rskin,Kskin
         * 1,1200.0,3400.0,45.0,10.0,0,0.15,0.30,0.001
         * 2,1800.0,2900.0,50.0,15.0,1,0.15,0.30,0.001
         * 3,2500.0,2100.0,40.0, 5.0,2,0.20,0.40,0.0005
         * @endcode
         *
         * Notes:
         * - Delimited formats supported by boost::escaped_list_separator are accepted
         *   (e.g. comma-separated files).
         * - Wells are stored internally in the `wells` vector.
         * - A Boost.Geometry R-tree is constructed for efficient spatial queries.
         * - The `global_index` field is assigned sequentially during reading.
         *
         * @param filename Path to the well input file.
         *
         * @return true if the file was successfully read.
         *
         * @throws std::runtime_error if the file cannot be opened.
         */
        bool read_well_data(const std::string& filename)
        {
            wells.clear();
            well_rtree.clear();
            using Tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
            std::ifstream file(filename);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file: " + filename);
            }

            std::string line;
            // Skip header
            std::getline(file, line);

            std::vector<WellValue> rtree_values;
            wells.reserve(100000); // optional but recommended
            rtree_values.reserve(100000);
            std::size_t i = 0;

            while (std::getline(file, line))
            {
                Tokenizer tok(line);
                auto it = tok.begin();
                MNWell w;

                w.Eid = std::stoi(*it++);
                w.x = std::stod(*it++);
                w.y = std::stod(*it++);
                w.top = std::stod(*it++);
                w.bottom = std::stod(*it++);
                w.q_row = std::stoi(*it++);
                //w.Q = 0.0;
                w.rw = std::stod(*it++);
                w.Rskin = std::stod(*it++);
                w.Kskin = std::stod(*it++);
                w.global_index = i;

                wells.push_back(w);
                rtree_values.emplace_back(Point2D(w.x, w.y), i);
                i++;
            }
            well_rtree = bgi::rtree<WellValue, bgi::quadratic<16>>(rtree_values.begin(), rtree_values.end());
            return true;
        }
    };

    inline void rank0_read_wells_distributes(const std::string &filename,
                                           MNWellCollection &wellcollection,
                                           MPI_Comm mpi_communicator)
    {
        const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
        // -----------------------------
        // (1) Rank 0 reads from disk
        // -----------------------------
        if (my_rank == 0)
        {
            // This populates wells and rtree on rank 0.
            wellcollection.read_well_data(filename);

            // Sanity: ensure global_index is consistent (0..n-1)
            for (std::size_t i = 0; i < wellcollection.wells.size(); ++i)
                wellcollection.wells[i].global_index = static_cast<unsigned int>(i);
        }
        else
        {
            // Ensure clean state on non-root ranks before receiving.
            wellcollection.wells.clear();
            wellcollection.well_rtree.clear();
        }

        // -----------------------------
        // (2) Broadcast number of wells
        // -----------------------------
        std::uint64_t n_wells_u64 = 0;
        if (my_rank == 0)
            n_wells_u64 = static_cast<std::uint64_t>(wellcollection.wells.size());

        int ierr = MPI_Bcast(&n_wells_u64, 1, MPI_UINT64_T, 0, mpi_communicator);
        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("MPI_Bcast failed while broadcasting n_wells.");

        const std::size_t n_wells = static_cast<std::size_t>(n_wells_u64);

        // -----------------------------
        // (3) Pack fields on rank 0
        // -----------------------------
        std::vector<MNWellPacket> packets;
        packets.resize(n_wells);
        if (my_rank == 0)
        {
            for (std::size_t i = 0; i < n_wells; ++i)
            {
                const MNWell &w = wellcollection.wells[i];
                MNWellPacket  p;
                p.Eid         = w.Eid;
                p.x           = w.x;
                p.y           = w.y;
                p.top         = w.top;
                p.bottom      = w.bottom;
                p.q_row       = w.q_row;
                p.rw          = w.rw;
                p.Rskin       = w.Rskin;
                p.Kskin       = w.Kskin;
                p.global_index= static_cast<std::uint32_t>(w.global_index);
                packets[i]    = p;
            }
        }

        // -----------------------------
        // (4) Broadcast packed array
        // -----------------------------
        // Broadcast as raw bytes to avoid MPI datatype construction.
        const std::size_t nbytes = packets.size() * sizeof(MNWellPacket);
        // MPI expects an int count; for 200k wells, this is safe.
        if (nbytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Well broadcast buffer too large for MPI_Bcast int count.");

        ierr = MPI_Bcast(reinterpret_cast<void *>(packets.data()),
                     static_cast<int>(nbytes),
                     MPI_BYTE,
                     0,
                     mpi_communicator);

        if (ierr != MPI_SUCCESS)
            throw std::runtime_error("MPI_Bcast failed while broadcasting well packets.");

        // -----------------------------
        // (5) Unpack on all ranks and rebuild rtree
        // -----------------------------
        wellcollection.wells.clear();
        wellcollection.wells.resize(n_wells);

        // Rebuild the rtree from scratch everywhere (including rank 0)
        wellcollection.well_rtree.clear();
        std::vector<WellValue> rtree_values;
        rtree_values.reserve(n_wells);

        for (std::size_t i = 0; i < n_wells; ++i)
        {
            const MNWellPacket &p = packets[i];
            MNWell w;
            w.Eid    = p.Eid;
            w.x      = p.x;
            w.y      = p.y;
            w.top    = p.top;
            w.bottom = p.bottom;
            w.q_row  = p.q_row;
            w.rw     = p.rw;
            w.Rskin  = p.Rskin;
            w.Kskin  = p.Kskin;

            // Fields not populated from file:
            w.trace_dof_indices.clear();

            // Keep a consistent global index across ranks:
            w.global_index = static_cast<unsigned int>(p.global_index);

            wellcollection.wells[i] = std::move(w);

            // Your existing indexing uses 'i' as the rtree payload index.
            // If you want the payload to be w.global_index, use that instead.
            rtree_values.emplace_back(Point2D(wellcollection.wells[i].x,
                                              wellcollection.wells[i].y),
                                      w.global_index);

            wellcollection.well_rtree =
                bgi::rtree<WellValue, bgi::quadratic<16>>(rtree_values.begin(), rtree_values.end());

        }
    }



}

#endif //MNWELLS_H

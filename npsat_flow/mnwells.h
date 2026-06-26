//
// Created by giorgk on 6/24/26.
//

#ifndef MNWELLS_H
#define MNWELLS_H
#include <deal.II/base/point.h>
#include <deal.II/grid/tria_description.h>

#include <boost/geometry/index/rtree.hpp>

#include "TimeSeries.h"
#include "flow_structures.h"
#include "helper_func.h"
#include "boost_helper_func.h"

namespace npsat_flow {
    using namespace dealii;

    //namespace bg = boost::geometry;
    //namespace bgi = boost::geometry::index;
    //using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
    //using WellValue = std::pair<Point2D, std::size_t>;
    //using Polygon = bg::model::polygon<Point2D>;

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

    // Helper functions related to MNW
    template<int dim>
    void calc_ro(typename DoFHandler<dim>::active_cell_iterator& cell, double& ro, double& b )
    {
        Point<dim> v0 = cell->vertex(0);
        Point<dim> v1 = cell->vertex(1);
        Point<dim> v2 = cell->vertex(2);
        Point<dim> v4 = cell->vertex(4);

        double dx2 = 0.0, dy2 = 0.0;
        for (unsigned int d = 0; d < std::min<unsigned int>(2, dim); ++d)
        {
            dx2 += (v1[d] - v0[d]) * (v1[d] - v0[d]);
            dy2 += (v2[d] - v0[d]) * (v2[d] - v0[d]);
        }
        const double dx = std::sqrt(dx2);
        const double dy = std::sqrt(dy2);

        // MNW2 eq. (5): ro = 0.14 * sqrt(dx^2 + dy^2)  :contentReference[oaicite:4]{index=4}
        ro = 0.14 * std::sqrt(dx*dx + dy*dy);

        b = v4[2] - v0[2];
        double min_thickness = 1e-12;
        AssertThrow(b > min_thickness, ExcMessage("Invalid cell geometry: cell thickness = " +
               Utilities::to_string(b) + ". Expected vertex 4 to be above vertex 0."));
    }

    inline double compute_CWC_MNW2(double K, double b, double bw, double rw, double ro, double Kskin, double Rskin)
    {
        // Eq. 9 Skin term
        const double skin = ((K * b) / (Kskin * bw) - 1.0) * std::log(Rskin / rw);
        // Eq. 8 + Eq. 10
        const double numerator   = 2.0 * numbers::PI * b * K;
        const double denominator = std::log(ro / rw) + skin;
        return numerator / denominator;
    }

    inline void print_cell_well_map_csv(
        const std::string &prefix,
        const std::vector<std::pair<unsigned int, std::vector<CellWellLink>>> &local_cell_well_map,
        const int my_rank)
    {
        // Construct filename
        const std::string filename =
            prefix + "_cell_well_" + std::to_string(my_rank) + ".csv";

        std::ofstream out(filename);
        AssertThrow(out.good(), ExcMessage("Cannot open file: " + filename));

        // ---------------------------------------------------------------------
        // CSV header
        // ---------------------------------------------------------------------
        out << "cell_index,"
            << "well_index,"
            << "well_rank,"
            << "cwc,"
            << "sl,"
            << "ze";
        for (unsigned int i = 0; i < 6; ++i)
            out << ",tr_dof" << i;

        out << '\n';

        // ---------------------------------------------------------------------
        // Data
        // ---------------------------------------------------------------------
        for (const auto &cell_entry : local_cell_well_map)
        {
            const unsigned int cell_index = cell_entry.first;
            const auto &links = cell_entry.second;

            for (const auto &link : links)
            {
                // Defensive check: you stated this is always 6
                AssertThrow(link.trace_dof_indices.size() == 6,
                            ExcMessage("Expected exactly 6 trace DoFs"));

                out << cell_index << ","
                    << link.well_global_index << ","
                    << link.well_owner_rank << ","
                    << link.cwc << ","
                    << link.sl << ","
                    << link.ze;

                for (unsigned int i = 0; i < 6; ++i)
                    out << "," << link.trace_dof_indices[i];

                out << '\n';
            }
        }

        out.close();
    }


    template <typename Key, typename T> class SortedVectorMap {
    public:
        SortedVectorMap() = default;
        // Reserve upfront if you know the size
        explicit SortedVectorMap(std::size_t n) { vec.reserve(n); }

        // Insert a new key/value pair (duplicates allowed until sorted)
        void insert(Key key, const T& value)
        {
            vec.emplace_back(key, value);
            sorted = false;
        }

        // Sort by key (must be called before lookup)
        void sort()
        {
            std::sort(vec.begin(), vec.end(),
                [](auto& a, auto& b)
                {
                    return a.first < b.first;
                });

            // Optionally remove duplicates (keep first occurrence)
            vec.erase(std::unique(vec.begin(), vec.end(),
                [](auto& a, auto& b)
                {
                    return a.first == b.first;
                }), vec.end());
            sorted = true;
        }

        // Lookup by key, returns pointer or nullptr
        const T* find(Key key) const
        {
            AssertThrow(sorted, ExcMessage("SortedVectorMap::find() called before sort()."));

            auto it = std::lower_bound(
                vec.begin(), vec.end(), key,
                [](auto& a, int key) { return a.first < key; });
            if (it != vec.end() && it->first == key)
                return &it->second;
            return nullptr;
        }

        // Convenience: throws if not found
        const T& at(Key key) const
        {
            const T* p = find(key);
            if (!p) throw std::out_of_range("Key not found");
            return *p;
        }

        bool contains(Key key) const
        {
            return find(key) != nullptr;
        }

        std::size_t size() const
        {
            return vec.size();
        }

        const std::vector<std::pair<Key,T>>& data() const { return vec; }

        void print_data(const std::string &prefix,
                        const int my_rank) const;

    private:
        std::vector<std::pair<Key, T>> vec;
        mutable bool sorted = false;



    };

    template <>
    inline void
    SortedVectorMap<types::global_dof_index,
                    std::vector<npsat_flow::WellRef>>
    ::print_data(const std::string &prefix,
                 const int my_rank) const
    {
        const std::string filename = prefix + "_trace_to_well_" + std::to_string(my_rank) + ".csv";

        std::ofstream out(filename);
        AssertThrow(out.good(),
                    ExcMessage("Could not open file: " + filename));

        // CSV header
        out << "tr_dof,well_dof,well_rank\n";

        // Data
        for (const auto &kv : vec)
        {
            const types::global_dof_index tr_dof = kv.first;

            for (const auto &wr : kv.second)
            {
                out << tr_dof << ","
                    << wr.well_id << ","
                    << wr.well_rank << "\n";
            }
        }

        out.close();
    }

    template <>
    inline void
    SortedVectorMap<unsigned int,
                    std::vector<npsat_flow::TraceRef>>
    ::print_data(const std::string &prefix,
                 const int my_rank) const
    {
        const std::string filename = prefix + "_well_to_trace_" + std::to_string(my_rank) + ".csv";

        std::ofstream out(filename);
        AssertThrow(out.good(),
                    ExcMessage("Could not open file: " + filename));

        // CSV header
        out << "well_dof,tr_dof,tr_rank\n";

        // Data
        for (const auto &kv : vec)
        {
            const unsigned int well_dof = kv.first;

            for (const auto &tr : kv.second)
            {
                out << well_dof << ","
                    << tr.trace_id << ","
                    << tr.trace_rank << "\n";
            }
        }

        out.close();
    }


    inline void write_wells_as_dataout_1d3(const std::string &filename, const std::vector<WellSegmentOut> &segments) {
        if (segments.empty())
        {
            return;
        }
        Triangulation<1,3> tria;

        std::vector<Point<3>> vertices;
        vertices.reserve(2 * segments.size());

        std::vector<CellData<1>> cells;
        cells.reserve(segments.size());

        for (std::size_t i = 0; i < segments.size(); ++i)
        {
            const WellSegmentOut &s = segments[i];

            const unsigned int v0 = static_cast<unsigned int>(vertices.size());
            vertices.push_back(Point<3>(s.x, s.y, s.z_bot));

            const unsigned int v1 = static_cast<unsigned int>(vertices.size());
            vertices.push_back(Point<3>(s.x, s.y, s.z_top));

            CellData<1> cd;
            cd.vertices[0] = v0;
            cd.vertices[1] = v1;
            cd.material_id = 0;

            cells.push_back(cd);
        }

        SubCellData sub;
        tria.create_triangulation(vertices, cells, sub);

        const unsigned int n_cells = tria.n_active_cells();
        AssertThrow(n_cells == segments.size(), ExcInternalError());

        Vector<double> Eid_cell(n_cells);
        Vector<double> cell_id_cell(n_cells);
        Vector<double> ze_cell(n_cells);
        Vector<double> Qe_cell(n_cells);
        Vector<double> cwc_cell(n_cells);
        Vector<double> Qwbf_top_cell(n_cells);
        Vector<double> Qtot_cell(n_cells);
        Vector<double> F_top_well_cell(n_cells);
        Vector<double> diff_cell(n_cells);

        for (unsigned int i = 0; i < n_cells; ++i) {
            const WellSegmentOut &s = segments[i];
            // IMPORTANT: "Eid" name and value from MNWell::Eid
            Eid_cell[i]        = static_cast<double>(s.Eid);
            cell_id_cell[i]    = static_cast<double>(s.cell_id);
            ze_cell[i]         = s.ze;
            Qe_cell[i]         = s.Qe;
            cwc_cell[i]        = s.cwc;
            Qwbf_top_cell[i]   = s.Qwbf_top;
            Qtot_cell[i]       = s.Qtot;
            F_top_well_cell[i] = s.F_top_well;
            diff_cell[i]       = s.diff;
        }

        DataOut<1, DoFHandler<1, 3> > data_out;
        data_out.attach_triangulation(tria);

        data_out.add_data_vector(Eid_cell,"Eid", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(cell_id_cell, "cell_id", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(ze_cell, "ze", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(Qe_cell, "Qe", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(cwc_cell, "cwc", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(Qwbf_top_cell, "Qwbf_top", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(Qtot_cell, "Qtot", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(F_top_well_cell, "F_top_well", DataOut<1, DoFHandler<1, 3> >::type_cell_data);
        data_out.add_data_vector(diff_cell, "diff", DataOut<1, DoFHandler<1, 3> >::type_cell_data);

        data_out.build_patches();

        std::ofstream out(filename);
        AssertThrow(out.good(), ExcMessage("Could not open: " + filename));
        data_out.write_vtu(out);
    }

    void write_wells_as_legacy_vtk_polydata(const std::string &filename, const std::vector<WellSegmentOut> &segments) {
        if (segments.empty())
            return;
        std::ofstream vtk(filename);
        AssertThrow(vtk.good(), ExcMessage("Could not open: " + filename));

        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "Wellbore segments\n";
        vtk << "ASCII\n";
        vtk << "DATASET POLYDATA\n";
        vtk << std::setprecision(16) << std::scientific;

        // points: 2 per segment
        const std::size_t n_lines  = segments.size();
        const std::size_t n_points = 2 * n_lines;

        vtk << "POINTS " << n_points << " double\n";
        for (const auto &s : segments)
        {
            vtk << s.x << " " << s.y << " " << s.z_bot << "\n";
            vtk << s.x << " " << s.y << " " << s.z_top << "\n";
        }

        vtk << "LINES " << n_lines << " " << (3 * n_lines) << "\n";
        for (std::size_t i = 0; i < n_lines; ++i)
        {
            const std::size_t p0 = 2*i;
            const std::size_t p1 = 2*i + 1;
            vtk << "2 " << p0 << " " << p1 << "\n";
        }

        vtk << "CELL_DATA " << n_lines << "\n";

        auto write_int = [&](const std::string &name, auto getter)
        {
            vtk << "SCALARS " << name << " int 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (const auto &s : segments) vtk << static_cast<int>(getter(s)) << "\n";
        };

        auto write_uint = [&](const std::string &name, auto getter)
        {
            vtk << "SCALARS " << name << " unsigned_int 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (const auto &s : segments) vtk << static_cast<unsigned int>(getter(s)) << "\n";
        };

        auto write_double = [&](const std::string &name, auto getter)
        {
            vtk << "SCALARS " << name << " double 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (const auto &s : segments) vtk << static_cast<double>(getter(s)) << "\n";
        };

        // IMPORTANT: identifier field must be called "Eid"
        write_int("Eid",               [](const WellSegmentOut &s){ return s.Eid; });
        write_uint("cell_id",          [](const WellSegmentOut &s){ return s.cell_id; });
        write_double("ze",             [](const WellSegmentOut &s){ return s.ze; });
        write_double("Qe",             [](const WellSegmentOut &s){ return s.Qe; });
        write_double("CWC",            [](const WellSegmentOut &s){ return s.cwc; });
        write_double("Qwbf_top",       [](const WellSegmentOut &s){ return s.Qwbf_top; });
        write_double("Qtot",           [](const WellSegmentOut &s){ return s.Qtot; });
        write_double("F_top_well",     [](const WellSegmentOut &s){ return s.F_top_well; });
        write_double("diff",           [](const WellSegmentOut &s){ return s.diff; });

        vtk.flush();
    }




}

#endif //MNWELLS_H

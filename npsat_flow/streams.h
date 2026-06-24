//
// Created by giorgk on 6/24/26.
//

#ifndef STREAMS_H
#define STREAMS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include "nanoflann.hpp"
#include "TimeSeries.h"
#include "poly_x_poly.h"

namespace npsat_flow{
    using namespace dealii;

    struct StreamKDPoint
    {
        double x = 0.0;
        double y = 0.0;
        int id = -1;
    };

    struct StreamKDCloud
    {
        std::vector<StreamKDPoint> pts;

        inline size_t kdtree_get_point_count() const
        {
            return pts.size();
        }

        inline double kdtree_get_pt(const size_t idx,
                                    const size_t dim) const
        {
            return dim == 0 ? pts[idx].x : pts[idx].y;
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX &) const
        {
            return false;
        }
    };

    using StreamKDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<double, StreamKDCloud>, StreamKDCloud, 2>;

    template <int dim>
    class StreamCollection
    {
    public:
        struct Stream
        {
            unsigned int id = 0;

            std::vector<double> xpoly;
            std::vector<double> ypoly;

            int row_id = -1;
            double length = 0.0;

            double xmin = 0.0;
            double xmax = 0.0;
            double ymin = 0.0;
            double ymax = 0.0;

            double radius = 0.0;

            Point<dim - 1> centroid;
        };

        StreamCollection()
            :
            stream_multiplier(1.0),
            max_stream_radius(0.0),
            intersection_area_tolerance(1.0e-8)
        {}

        void clear()
        {
            streams.clear();
            stream_cloud.pts.clear();
            stream_index.reset();
            max_stream_radius = 0.0;
        }

        bool empty() const
        {
            return streams.empty();
        }

        unsigned int size() const
        {
            return static_cast<unsigned int>(streams.size());
        }

        /*!
         * \brief Read stream network information from an ASCII input file.
         *
         * The first line gives the number of stream segments.
         *
         * Each stream can be given either as:
         *
         * \code
         * 2  Q_rate  width
         * x1 y1
         * x2 y2
         * \endcode
         *
         * or as an explicit polygon:
         *
         * \code
         * 4  Q_rate
         * x1 y1
         * x2 y2
         * x3 y3
         * x4 y4
         * \endcode
         *
         * Complete example:
         *
         * \code
         * 3
         *
         * 2 -0.001 20.0
         * 1000.0 500.0
         * 1500.0 700.0
         *
         * 4 -0.0005
         * 2000.0 1000.0
         * 2200.0 1000.0
         * 2200.0 1200.0
         * 2000.0 1200.0
         *
         * 3 -0.0002
         * 3000.0 1500.0
         * 3200.0 1700.0
         * 3100.0 1900.0
         * \endcode
         *
         * For two-point streams, width is interpreted as half-width.
         *
         * \param filename Name of the stream input file.
         * \return True if reading and KD-tree construction succeeded.
         */
        bool read_streams(const std::string &geometry_filename, const std::string &rates_filename, MPI_Comm comm)
        {
            clear();

            rates.read_data(rates_filename, comm);
            rates.multiply_by_factor(stream_multiplier);

            std::vector<double> stream_buffer;

            read_stream_geometry_as_buffer(geometry_filename, stream_buffer, comm);

            build_streams_from_buffer(stream_buffer);

            build_stream_index();

            return true;
        }

        bool find_streams_in_polygon(
            std::vector<const Stream *>      &streams_in_cell,
            const std::vector<double>  &cell_x,
            const std::vector<double>  &cell_y) const
        {
            streams_in_cell.clear();

            if (streams.empty())
                return false;

            std::vector<StreamIntersection> intersections;

            if (!collect_stream_intersections(intersections, cell_x, cell_y))
                return false;

            streams_in_cell.reserve(intersections.size());


            for (unsigned int i = 0; i < intersections.size(); ++i)
                streams_in_cell.push_back(intersections[i].stream);

            return true;
        }

        bool find_stream_recharge_in_polygon(
            std::vector<double>        &xc,
            std::vector<double>        &yc,
            std::vector<double>        &area,
            std::vector<double>        &q,
            const std::vector<double>  &cell_x,
            const std::vector<double>  &cell_y) const
        {
            xc.clear();
            yc.clear();
            area.clear();
            q.clear();

            if (streams.empty())
                return false;

            std::vector<StreamIntersection> intersections;
            if (!collect_stream_intersections(intersections, cell_x, cell_y))
                return false;

            xc.reserve(intersections.size());
            yc.reserve(intersections.size());
            area.reserve(intersections.size());
            q.reserve(intersections.size());

            for (unsigned int i = 0; i < intersections.size(); ++i)
            {
                const StreamIntersection &item = intersections[i];
                AssertThrow(item.stream != nullptr, ExcInternalError());
                AssertThrow(item.stream->row_id >= 0,
                            ExcMessage("Stream row_id must be nonnegative to read stream rates."));

                xc.push_back(item.xc);
                yc.push_back(item.yc);
                area.push_back(item.area);

                const double stream_rate =
                    rates.get_value_at_step(item.stream->row_id, time_step_number);
                q.push_back(item.area * stream_rate);
            }
            return true;
        }

        void set_time_step_number(const int time_step_number_in)
        {
            time_step_number = time_step_number_in;
        }

        int get_time_step_number() const
        {
            return time_step_number;
        }

        double stream_multiplier;

    private:
        struct StreamIntersection
        {
            const Stream *stream = nullptr;
            double xc = 0.0;
            double yc = 0.0;
            double area = 0.0;
        };

        std::vector<Stream> streams;

        TimeSeriesData<double> rates;
        int time_step_number = 0;

        StreamKDCloud stream_cloud;
        std::shared_ptr<StreamKDTree> stream_index;

        double max_stream_radius;
        double intersection_area_tolerance;

        static void read_stream_geometry_as_buffer(const std::string  &geometry_filename,
            std::vector<double> &buffer, MPI_Comm comm)
        {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);

            if (rank == 0)
            {
                std::ifstream in(geometry_filename.c_str());
                if (!in)
                    throw std::runtime_error("Cannot open stream geometry file: " + geometry_filename);

                unsigned int nseg = 0;
                in >> nseg;

                buffer.clear();
                buffer.push_back(static_cast<double>(nseg));

                for (unsigned int iseg = 0; iseg < nseg; ++iseg)
                {
                    unsigned int npoints = 0;
                    int row_id = -1;
                    double width = 0.0;

                    in >> npoints >> row_id;

                    if (npoints == 2)
                        in >> width;

                    if (npoints < 2 || npoints > 4)
                        throw std::runtime_error("Invalid stream polygon size.");

                    buffer.push_back(static_cast<double>(npoints));
                    buffer.push_back(static_cast<double>(row_id));

                    if (npoints == 2)
                        buffer.push_back(width);

                    for (unsigned int i = 0; i < npoints; ++i)
                    {
                        double x = 0.0;
                        double y = 0.0;

                        in >> x >> y;

                        buffer.push_back(x);
                        buffer.push_back(y);
                    }
                }
            }

            int buffer_size = static_cast<int>(buffer.size());
            MPI_Bcast(&buffer_size, 1, MPI_INT, 0, comm);

            if (rank != 0)
                buffer.resize(buffer_size);

            MPI_Bcast(buffer.data(), buffer_size, MPI_DOUBLE, 0, comm);
        }

        void build_streams_from_buffer(const std::vector<double> &buffer)
        {
            streams.clear();
            stream_cloud.pts.clear();
            stream_index.reset();
            max_stream_radius = 0.0;

            if (buffer.empty())
                return;

            unsigned int pos = 0;

            const unsigned int nseg = static_cast<unsigned int>(buffer[pos++]);
            streams.reserve(nseg);

            for (unsigned int iseg = 0; iseg < nseg; ++iseg)
            {
                const unsigned int npoints = static_cast<unsigned int>(buffer[pos++]);
                const int row_id = static_cast<int>(buffer[pos++]);
                double width = 0.0;

                if (npoints == 2)
                    width = buffer[pos++];

                std::vector<double> xx;
                std::vector<double> yy;
                if (npoints == 2)
                {
                    Point<dim - 1> A;
                    Point<dim - 1> B;

                    A[0] = buffer[pos++];
                    A[1] = buffer[pos++];

                    B[0] = buffer[pos++];
                    B[1] = buffer[pos++];

                    create_river_outline(xx, yy, A, B, width);
                }
                else
                {
                    xx.resize(npoints);
                    yy.resize(npoints);

                    for (unsigned int i = 0; i < npoints; ++i)
                    {
                        xx[i] = buffer[pos++];
                        yy[i] = buffer[pos++];
                    }
                }

                Stream s;
                s.id = iseg;
                s.row_id = row_id;
                s.xpoly = xx;
                s.ypoly = yy;

                update_geometry(s);
                streams.push_back(s);
            }
        }

        void build_stream_index()
        {
            stream_cloud.pts.clear();
            stream_cloud.pts.reserve(streams.size());

            for (unsigned int i = 0; i < streams.size(); ++i)
            {
                StreamKDPoint p;
                p.x = streams[i].centroid[0];
                p.y = streams[i].centroid[1];
                p.id = static_cast<int>(i);

                stream_cloud.pts.push_back(p);
            }

            if (stream_cloud.pts.empty())
                return;

            stream_index = std::make_shared<StreamKDTree>(
                2,
                stream_cloud,
                nanoflann::KDTreeSingleIndexAdaptorParams(10));

            stream_index->buildIndex();
        }

        void find_candidate_streams(
            const std::vector<double> &cell_x,
            const std::vector<double> &cell_y,
            std::vector<unsigned int> &candidate_ids) const
        {
            candidate_ids.clear();

            if (!stream_index || stream_cloud.pts.empty())
                return;

            double cx = 0.0;
            double cy = 0.0;

            for (unsigned int i = 0; i < cell_x.size(); ++i)
            {
                cx += cell_x[i];
                cy += cell_y[i];
            }

            cx /= static_cast<double>(cell_x.size());
            cy /= static_cast<double>(cell_y.size());

            double cell_radius = 0.0;

            for (unsigned int i = 0; i < cell_x.size(); ++i)
            {
                const double dx = cell_x[i] - cx;
                const double dy = cell_y[i] - cy;

                cell_radius = std::max(cell_radius, std::sqrt(dx * dx + dy * dy));
            }

            const double search_radius = cell_radius + max_stream_radius;
            const double search_radius_squared = search_radius * search_radius;

            const double query_pt[2] = {cx, cy};

            std::vector<nanoflann::ResultItem<typename StreamKDTree::IndexType, double> > matches;
            nanoflann::SearchParameters params;

            stream_index->radiusSearch(query_pt, search_radius_squared,matches, params);

            candidate_ids.reserve(matches.size());

            for (size_t i = 0; i < matches.size(); ++i)
            {
                const typename StreamKDTree::IndexType cloud_id = matches[i].first;
                const int stream_id = stream_cloud.pts[cloud_id].id;

                if (stream_id >= 0)
                    candidate_ids.push_back(static_cast<unsigned int>(stream_id));
            }
        }

        bool collect_stream_intersections(
        std::vector<StreamIntersection> &intersections,
        const std::vector<double>       &cell_x,
        const std::vector<double>       &cell_y) const
        {
            intersections.clear();

            if (streams.empty())
                return false;

            std::vector<unsigned int> candidate_ids;
            find_candidate_streams(cell_x, cell_y, candidate_ids);

            if (candidate_ids.empty())
                return false;

            const double cell_xmin = *std::min_element(cell_x.begin(), cell_x.end());
            const double cell_xmax = *std::max_element(cell_x.begin(), cell_x.end());
            const double cell_ymin = *std::min_element(cell_y.begin(), cell_y.end());
            const double cell_ymax = *std::max_element(cell_y.begin(), cell_y.end());

            for (unsigned int k = 0; k < candidate_ids.size(); ++k)
            {
                const Stream &s = streams[candidate_ids[k]];
                bool bbox_crosses = bbox_intersects(cell_xmin, cell_xmax,
                                         cell_ymin, cell_ymax,
                                         s.xmin, s.xmax,
                                         s.ymin, s.ymax);

                if (!bbox_crosses)
                    continue;

                double xci = 0.0;
                double yci = 0.0;

                try
                {
                    const double area =
                        pxp::polyXpoly_fast(cell_x, cell_y, s.xpoly, s.ypoly, xci, yci);

                    if (area <= intersection_area_tolerance)
                        continue;

                    StreamIntersection item;
                    item.stream = &s;
                    item.xc = xci;
                    item.yc = yci;
                    item.area = area;

                    intersections.push_back(item);
                }
                catch (...)
                {
                    std::cout << "Stream polygon intersection failed." << std::endl;
                }
            }

            return !intersections.empty();
        }

        void update_geometry(Stream &s)
        {
            s.xmin =  std::numeric_limits<double>::max();
            s.ymin =  std::numeric_limits<double>::max();
            s.xmax = -std::numeric_limits<double>::max();
            s.ymax = -std::numeric_limits<double>::max();

            double xc = 0.0;
            double yc = 0.0;

            for (unsigned int i = 0; i < s.xpoly.size(); ++i)
            {
                s.xmin = std::min(s.xmin, s.xpoly[i]);
                s.xmax = std::max(s.xmax, s.xpoly[i]);

                s.ymin = std::min(s.ymin, s.ypoly[i]);
                s.ymax = std::max(s.ymax, s.ypoly[i]);

                xc += s.xpoly[i];
                yc += s.ypoly[i];
            }

            const double n = static_cast<double>(s.xpoly.size());

            s.centroid[0] = xc / n;
            s.centroid[1] = yc / n;

            s.radius = 0.0;

            for (unsigned int i = 0; i < s.xpoly.size(); ++i)
            {
                const double dx = s.xpoly[i] - s.centroid[0];
                const double dy = s.ypoly[i] - s.centroid[1];
                s.radius = std::max(s.radius, std::sqrt(dx * dx + dy * dy));
            }

            max_stream_radius = std::max(max_stream_radius, s.radius);
        }

        static bool bbox_intersects(const double axmin,
                                const double axmax,
                                const double aymin,
                                const double aymax,
                                const double bxmin,
                                const double bxmax,
                                const double bymin,
                                const double bymax)
        {
            if (axmax < bxmin) return false;
            if (axmin > bxmax) return false;
            if (aymax < bymin) return false;
            if (aymin > bymax) return false;
            return true;
        }

        static void create_river_outline(std::vector<double> &xx,
                                     std::vector<double> &yy,
                                     const Point<dim - 1> &A,
                                     const Point<dim - 1> &B,
                                     const double width)
        {
            xx.clear();
            yy.clear();

            const double dx = B[0] - A[0];
            const double dy = B[1] - A[1];

            const double len = std::sqrt(dx * dx + dy * dy);

            if (len < 1.0e-12)
            {
                std::cout << "Stream segment has nearly zero length." << std::endl;
                return;
            }

            const double nx = -dy / len;
            const double ny =  dx / len;

            xx.push_back(A[0] + width * nx);
            yy.push_back(A[1] + width * ny);

            xx.push_back(B[0] + width * nx);
            yy.push_back(B[1] + width * ny);

            xx.push_back(B[0] - width * nx);
            yy.push_back(B[1] - width * ny);

            xx.push_back(A[0] - width * nx);
            yy.push_back(A[1] - width * ny);
        }

    };

}

#endif //STREAMS_H

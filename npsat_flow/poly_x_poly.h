//
// Created by giorgk on 6/24/26.
//

#ifndef POLY_X_POLY_H
#define POLY_X_POLY_H

#include <vector>
#include <cmath>

namespace npsat_flow {
    namespace pxp{
        struct Point2D
        {
            double x = 0.0;
            double y = 0.0;
        };

        inline double cross2d(const Point2D &a,
                              const Point2D &b,
                              const Point2D &c)
        {
            return (b.x - a.x) * (c.y - a.y)
                 - (b.y - a.y) * (c.x - a.x);
        }

        inline double polygon_signed_area(const std::vector<Point2D> &p)
        {
            double a = 0.0;

            const unsigned int n = static_cast<unsigned int>(p.size());

            for (unsigned int i = 0; i < n; ++i)
            {
                const Point2D &p1 = p[i];
                const Point2D &p2 = p[(i + 1) % n];

                a += p1.x * p2.y - p2.x * p1.y;
            }

            return 0.5 * a;
        }

        inline Point2D line_intersection(const Point2D &p1,
                                 const Point2D &p2,
                                 const Point2D &q1,
                                 const Point2D &q2)
        {
            const double a1 = p2.y - p1.y;
            const double b1 = p1.x - p2.x;
            const double c1 = a1 * p1.x + b1 * p1.y;

            const double a2 = q2.y - q1.y;
            const double b2 = q1.x - q2.x;
            const double c2 = a2 * q1.x + b2 * q1.y;

            const double det = a1 * b2 - a2 * b1;

            Point2D r;

            if (std::abs(det) < 1.0e-30)
            {
                r.x = 0.5 * (p1.x + p2.x);
                r.y = 0.5 * (p1.y + p2.y);
                return r;
            }

            r.x = (b2 * c1 - b1 * c2) / det;
            r.y = (a1 * c2 - a2 * c1) / det;

            return r;
        }

        inline bool inside_clip_edge(const Point2D &a,
                             const Point2D &b,
                             const Point2D &p,
                             const double clip_orientation)
        {
            const double c = cross2d(a, b, p);

            if (clip_orientation >= 0.0)
                return c >= -1.0e-12;

            return c <= 1.0e-12;
        }

        inline double polygon_area_centroid(const std::vector<Point2D> &p,
                                    double &xc,
                                    double &yc)
        {
            xc = 0.0;
            yc = 0.0;

            const unsigned int n = static_cast<unsigned int>(p.size());

            if (n < 3)
                return 0.0;

            double a2 = 0.0;
            double cx = 0.0;
            double cy = 0.0;

            for (unsigned int i = 0; i < n; ++i)
            {
                const Point2D &p0 = p[i];
                const Point2D &p1 = p[(i + 1) % n];

                const double cross = p0.x * p1.y - p1.x * p0.y;

                a2 += cross;
                cx += (p0.x + p1.x) * cross;
                cy += (p0.y + p1.y) * cross;
            }

            if (std::abs(a2) < 1.0e-30)
                return 0.0;

            xc = cx / (3.0 * a2);
            yc = cy / (3.0 * a2);

            return std::abs(0.5 * a2);
        }

        inline double polyXpoly_fast(const std::vector<double> &x1,
                             const std::vector<double> &y1,
                             const std::vector<double> &x2,
                             const std::vector<double> &y2,
                             double &xc,
                             double &yc)
        {
            xc = 0.0;
            yc = 0.0;

            if (x1.size() < 3 || x2.size() < 3)
                return 0.0;

            std::vector<Point2D> subject;
            std::vector<Point2D> clip;

            subject.reserve(x1.size());
            clip.reserve(x2.size());

            for (unsigned int i = 0; i < x1.size(); ++i)
                subject.push_back(Point2D{x1[i], y1[i]});

            for (unsigned int i = 0; i < x2.size(); ++i)
                clip.push_back(Point2D{x2[i], y2[i]});

            const double clip_orientation = polygon_signed_area(clip);

            std::vector<Point2D> output = subject;

            for (unsigned int i = 0; i < clip.size(); ++i)
            {
                const Point2D &A = clip[i];
                const Point2D &B = clip[(i + 1) % clip.size()];

                std::vector<Point2D> input = output;
                output.clear();

                if (input.empty())
                    break;

                Point2D S = input.back();

                for (unsigned int j = 0; j < input.size(); ++j)
                {
                    const Point2D &E = input[j];

                    const bool E_inside = inside_clip_edge(A, B, E, clip_orientation);
                    const bool S_inside = inside_clip_edge(A, B, S, clip_orientation);

                    if (E_inside)
                    {
                        if (!S_inside)
                            output.push_back(line_intersection(S, E, A, B));

                        output.push_back(E);
                    }
                    else if (S_inside)
                    {
                        output.push_back(line_intersection(S, E, A, B));
                    }

                    S = E;
                }
            }

            return polygon_area_centroid(output, xc, yc);
        }

    }
}

#endif //POLY_X_POLY_H

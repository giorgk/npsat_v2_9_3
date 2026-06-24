//
// Created by giorgk on 6/23/26.
//

#ifndef HELPER_FUNC_H
#define HELPER_FUNC_H

#include <algorithm>
#include <sstream>
#include <vector>
#include <fstream>

#include "flow_structures.h"

namespace npsat_flow{
    //using namespace dealii;

    inline bool parse_double_or_file(const std::string& input, double& value, std::string& filename)
    {
        std::stringstream ss(input);

        double tmp;
        // Try parsing entire string as double
        if ((ss >> tmp) && (ss >> std::ws).eof())
        {
            value = tmp;
            filename.clear();
            return true;
        }
        // Otherwise assume filename
        filename = input;
        return false;
    }

    inline std::string trim(const std::string& s)
    {
        auto start = std::find_if_not(s.begin(), s.end(),
            [](unsigned char c) { return std::isspace(c); });

        auto end = std::find_if_not(s.rbegin(), s.rend(),
            [](unsigned char c) { return std::isspace(c); }).base();

        if (start >= end)
            return "";

        return std::string(start, end);
    }

    inline bool path_is_absolute(const std::string &path)
    {
        return !path.empty() && path.front() == '/';
    }

    inline std::string join_paths(const std::string &base, const std::string &path)
    {
        if (base.empty() || path.empty() || path_is_absolute(path))
            return path;

        if (base.back() == '/')
            return base + path;

        return base + "/" + path;
    }

    inline std::string resolve_relative_path(const std::string &base, const std::string &path)
    {
        const std::string clean_path = trim(path);
        if (clean_path.empty() || path_is_absolute(clean_path))
            return clean_path;

        return join_paths(base, clean_path);
    }

    template<typename T>
    std::vector<T> parse_list(const std::string& str, char delimiter = ',')
    {
        std::vector<T> values;

        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter))
        {
            token = trim(token);

            std::stringstream converter(token);

            T value;
            converter >> value;

            if (converter.fail())
            {
                throw std::runtime_error(
                    "Failed parsing value \"" + token + "\"");
            }

            values.push_back(value);
        }

        return values;
    }

    template<typename T>
    bool try_parse_list(const std::string& str, std::vector<T>& values, char delimiter = ',')
    {
        std::vector<T> parsed_values;

        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter))
        {
            token = trim(token);
            if (token.empty())
                return false;

            std::stringstream converter(token);

            T value;
            converter >> value;

            if (converter.fail() || !(converter >> std::ws).eof())
                return false;

            parsed_values.push_back(value);
        }

        if (parsed_values.empty())
            return false;

        values = parsed_values;
        return true;
    }

    inline bool path_exists(const std::string &path)
    {
        struct stat path_info;
        return stat(path.c_str(), &path_info) == 0;
    }
    inline bool path_is_directory(const std::string &path)
    {
        struct stat path_info;
        return stat(path.c_str(), &path_info) == 0 && S_ISDIR(path_info.st_mode);
    }

    template <typename T>
    inline T clamp_(T value, T lo, T hi)
    {
        return std::max(lo, std::min(value, hi));
    }

    inline ScreenClip well_length_inside_cell(double w_bottom, double w_top, double p_bot, double p_top)
    {
        ScreenClip c;
        c.z_low  = std::max(w_bottom, p_bot);
        c.z_high = std::min(w_top,   p_top);
        c.L      = c.z_high - c.z_low;
        if (c.L <= 0.0)
            c = ScreenClip{};
        return c;
    }

    inline void interpolate_top_bottom(
        const std::vector<double> &X,
        const std::vector<double> &Y,
        const std::vector<double> &top,
        const std::vector<double> &bot,
        double x, double y,
        double &p_top, double &p_bot)
    {
        // Solve for (u, v) using Newton iteration on bilinear mapping
        double u = 0.5, v = 0.5; // good initial guess

        for (int iter = 0; iter < 5; ++iter)
        {
            // Shape functions
            double N0 = (1-u)*(1-v);
            double N1 = u*(1-v);
            double N2 = u*v;
            double N3 = (1-u)*v;

            // Mapped x,y
            double xx = N0*X[0] + N1*X[1] + N2*X[2] + N3*X[3];
            double yy = N0*Y[0] + N1*Y[1] + N2*Y[2] + N3*Y[3];

            // Residuals
            double rx = xx - x;
            double ry = yy - y;

            // Derivatives wrt u
            double dN0du = -(1-v);
            double dN1du = (1-v);
            double dN2du = v;
            double dN3du = -v;

            // Derivatives wrt v
            double dN0dv = -(1-u);
            double dN1dv = -u;
            double dN2dv = u;
            double dN3dv = (1-u);
            double dxdu = dN0du*X[0] + dN1du*X[1] + dN2du*X[2] + dN3du*X[3];
            double dydu = dN0du*Y[0] + dN1du*Y[1] + dN2du*Y[2] + dN3du*Y[3];
            double dxdv = dN0dv*X[0] + dN1dv*X[1] + dN2dv*X[2] + dN3dv*X[3];
            double dydv = dN0dv*Y[0] + dN1dv*Y[1] + dN2dv*Y[2] + dN3dv*Y[3];

            // Solve 2x2 system for du, dv
            double det = dxdu*dydv - dydu*dxdv;
            double du = (-rx*dydv + ry*dxdv) / det;
            double dv = (-dxdu*ry + dydu*rx) / det; u += du; v += dv;
        }

        // Final shape functions
        double N0 = (1-u)*(1-v);
        double N1 = u*(1-v);
        double N2 = u*v;
        double N3 = (1-u)*v;

        // Interpolate top and bottom
        p_top = N0*top[0] + N1*top[1] + N2*top[2] + N3*top[3];
        p_bot = N0*bot[0] + N1*bot[1] + N2*bot[2] + N3*bot[3];
    }



}

#endif //HELPER_FUNC_H

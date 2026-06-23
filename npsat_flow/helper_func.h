//
// Created by giorgk on 6/23/26.
//

#ifndef HELPER_FUNC_H
#define HELPER_FUNC_H

#include <algorithm>
#include <sstream>
#include <vector>
#include <fstream>

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

}

#endif //HELPER_FUNC_H

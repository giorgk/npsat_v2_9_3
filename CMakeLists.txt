cmake_minimum_required(VERSION 3.5.0)

project(npsat_project LANGUAGES CXX)

find_package(deal.II 9.3.0
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
if(NOT deal.II_FOUND)
    message(FATAL_ERROR
            "*** Could not locate a (sufficiently recent) version of deal.II ***"
    )
endif()

deal_ii_initialize_cached_variables()

find_path(BOOST_INCLUDE_DIR
        NAMES boost/program_options.hpp
        PATHS /usr/include
        REQUIRED
)

set(BOOST_PROGRAM_OPTIONS_LIB
        /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.74.0
)

message(STATUS "Boost include dir: ${BOOST_INCLUDE_DIR}")
message(STATUS "Boost program_options lib: ${BOOST_PROGRAM_OPTIONS_LIB}")

# ------------------------------------------------------------
# Executable: npsat_v2
# ------------------------------------------------------------
add_executable(npsat_v2
        npsat_v2.cpp
)

deal_ii_setup_target(npsat_v2)
target_include_directories(npsat_v2 PRIVATE ${BOOST_INCLUDE_DIR})
target_link_libraries(npsat_v2 ${BOOST_PROGRAM_OPTIONS_LIB})

# ------------------------------------------------------------
# Executable: trace
# ------------------------------------------------------------
add_executable(npsat_trace
        npsat_trace.cpp
)

deal_ii_setup_target(npsat_trace)
target_include_directories(npsat_trace PRIVATE ${BOOST_INCLUDE_DIR})
target_link_libraries(npsat_trace ${BOOST_PROGRAM_OPTIONS_LIB})

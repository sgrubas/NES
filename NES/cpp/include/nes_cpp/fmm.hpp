// Factored fast marching method (FMM): traveltimes from point sources on a regular 2-D or 3-D
// grid. Implements docs/FMM.md.
#pragma once

#include "nes_cpp/index.hpp"

#include <span>
#include <vector>

namespace nes_cpp {

// A regular grid whose node 0 is at the origin. Axes are in coordinate order, (x, z) or
// (x, y, z) with z last. Arrays on the grid are flat and C-order: the last axis is contiguous.
struct Grid {
    std::vector<Index> shape;    // number of nodes along each axis
    std::vector<double> spacing; // node spacing h along each axis
};

// Traveltime T from a point source by the factored fast marching method (docs/FMM.md).
//
//   velocity  v at every node, flat C-order, size = product of grid.shape; every value finite
//             and > 0 (for example in km/s). Read only; not copied.
//   grid      2 or 3 axes; shape >= 1 and finite spacing > 0 on every axis (for example in km).
//   source    multi-index of the source node, one entry per axis, inside the grid.
//   order     finite-difference order, 1 or 2.
//
// Returns T at every node, flat C-order like velocity, in units of spacing / velocity (km and
// km/s give s). Throws std::invalid_argument if an input breaks these rules.
std::vector<double> fmm(std::span<const double> velocity, const Grid& grid,
                        std::span<const Index> source, int order);

// Traveltime tables for several point sources, solved in parallel with OpenMP, one source per
// thread (docs/FMM.md, "Many sources").
//
//   velocity, grid, order   as for fmm().
//   sources   multi-indices of the sources, flat and row-major: n_sources x dim values, where
//             source s is sources[s * dim .. s * dim + dim). Every source inside the grid.
//
// Returns T for every source, flat C-order with shape (n_sources, n_0, n_1[, n_2]): T at node i
// for source s is at s * n + i, n = number of nodes. Equal bit for bit to calling fmm() for each
// source, for any number of threads. Throws std::invalid_argument if an input breaks the rules.
std::vector<double> fmm_multi(std::span<const double> velocity, const Grid& grid,
                              std::span<const Index> sources, int order);

} // namespace nes_cpp

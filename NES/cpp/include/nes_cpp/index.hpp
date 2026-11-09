// The index type used throughout nes_cpp.
#pragma once

#include <cstddef>

namespace nes_cpp {

// Signed index type for grid nodes, points and loops: neighbour offsets are negative, and
// OpenMP 2.0 (MSVC) accepts only signed loop variables.
using Index = std::ptrdiff_t;

} // namespace nes_cpp

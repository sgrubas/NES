// A parallel loop over independent iterations with OpenMP, shared by fmm_multi (one source per
// iteration) and NesModel (one batch of points per iteration). Internal to the library.
#pragma once

#include "nes_cpp/index.hpp"

#include <exception>

namespace nes_cpp::detail {

// Calls body(i) for i = 0, ..., n - 1, spread over the OpenMP threads; each call runs on one
// thread, in no fixed order. schedule(dynamic) gives the next iteration to whichever thread is
// free. An exception must not leave an OpenMP parallel region (the program would terminate), so
// the first exception thrown by `body` is kept and rethrown after the loop. Without OpenMP the
// pragmas are ignored and the loop runs serially.
template <typename Body> void parallel_for(Index n, const Body& body) {
    std::exception_ptr error;
#pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < n; ++i) { // signed index: OpenMP 2.0 (MSVC) requires it
        try {
            body(i);
        } catch (...) {
#pragma omp critical
            if (!error) {
                error = std::current_exception();
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

} // namespace nes_cpp::detail

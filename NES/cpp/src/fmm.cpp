// Factored fast marching method, first and second order, on 2-D and 3-D grids (docs/FMM.md).
//
// One code path serves both dimensions: a node is a flat C-order index, and its neighbours
// along axis k are node - stride[k] and node + stride[k] (docs/FMM.md, "Problem and grid").
// fmm_multi solves several sources in parallel with OpenMP (docs/FMM.md, "Many sources").

#include "nes_cpp/fmm.hpp"

#include "fmm_update.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace nes_cpp {

// Contract in fmm_update.hpp; method in docs/FMM.md, "Solving the update".
double detail::solve_update(std::span<Term> terms, double slowness) {
    // Sorted by beta, the term to drop when there is no real root is always the last one.
    std::sort(terms.begin(), terms.end(),
              [](const Term& a, const Term& b) { return a.beta < b.beta; });

    for (std::size_t n_used = terms.size(); n_used > 1; --n_used) {
        const std::span<const Term> used = terms.first(n_used);
        double A = 0.0;
        double mean_beta = 0.0; // the alpha^2-weighted mean of beta
        for (const Term& t : used) {
            A += t.alpha * t.alpha;
            mean_beta += t.alpha * t.alpha * t.beta;
        }
        mean_beta /= A;
        double V = 0.0;
        for (const Term& t : used) {
            V += t.alpha * t.alpha * (t.beta - mean_beta) * (t.beta - mean_beta);
        }
        const double discriminant = slowness * slowness - V;
        if (discriminant >= 0.0) {
            return mean_beta + std::sqrt(discriminant / A); // the larger root
        }
    }
    // A single term always has a real root.
    return terms[0].beta + slowness / terms[0].alpha;
}

namespace { // helpers used only in this file

using detail::Term;

constexpr double infinity = std::numeric_limits<double>::infinity();

// Throws std::invalid_argument unless velocity, grid and order follow the contract in fmm.hpp.
void check_grid(std::span<const double> velocity, const Grid& grid, int order) {
    const std::size_t dim = grid.shape.size();
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument("fmm: the grid needs 2 or 3 axes, got " + std::to_string(dim));
    }
    if (grid.spacing.size() != dim) {
        throw std::invalid_argument("fmm: the spacing needs one value per axis");
    }
    Index n_nodes = 1;
    for (std::size_t k = 0; k < dim; ++k) {
        if (grid.shape[k] < 1) {
            throw std::invalid_argument("fmm: the grid needs >= 1 node on every axis");
        }
        if (!std::isfinite(grid.spacing[k]) || grid.spacing[k] <= 0) {
            throw std::invalid_argument("fmm: the spacing must be finite and > 0");
        }
        n_nodes *= grid.shape[k];
    }
    if (std::ssize(velocity) != n_nodes) {
        throw std::invalid_argument("fmm: velocity has " + std::to_string(velocity.size()) +
                                    " values, the grid has " + std::to_string(n_nodes) + " nodes");
    }
    for (const double v : velocity) {
        if (!std::isfinite(v) || v <= 0) { // the first test also rejects NaN
            throw std::invalid_argument("fmm: velocity must be finite and > 0 at every node");
        }
    }
    if (order != 1 && order != 2) {
        throw std::invalid_argument("fmm: order must be 1 or 2, got " + std::to_string(order));
    }
}

// Throws std::invalid_argument unless `source` is a multi-index of a node of `grid`.
void check_source(const Grid& grid, std::span<const Index> source) {
    if (source.size() != grid.shape.size()) {
        throw std::invalid_argument("fmm: a source needs one index per axis");
    }
    for (std::size_t k = 0; k < source.size(); ++k) {
        if (source[k] < 0 || source[k] >= grid.shape[k]) {
            throw std::invalid_argument("fmm: a source is outside the grid");
        }
    }
}

// The state of one solve: the grid, the factor T0, the unknown tau and the accepted flags.
struct Marching {
    std::span<const double> velocity;   // view of the caller's array, not a copy
    Grid grid;                          // a copy: two or three numbers per axis
    std::vector<Index> stride;          // step of the flat index along each axis
    std::vector<Index> source;          // multi-index of the source
    int order;                          // finite-difference order, 1 or 2
    Index source_node = 0;              // flat index of the source
    double s0 = 0.0;                    // slowness at the source
    std::vector<double> T0;             // s0 * |x - x_s| at every node
    std::vector<double> tau;            // T = T0 * tau; infinity until a node is first updated
    std::vector<std::uint8_t> accepted; // 1 once tau is final. Not std::vector<bool>, whose
                                        // packed bits are slower to read and write.

    Marching(std::span<const double> velocity, const Grid& grid, std::span<const Index> source,
             int order);

    // Index of `node` along `axis` (docs/FMM.md, "Memory layout").
    Index coordinate(Index node, std::size_t axis) const {
        return node / stride[axis] % grid.shape[axis];
    }

    double T(Index node) const { return T0[node] * tau[node]; }

    double update(Index node) const;
    void march();
};

Marching::Marching(std::span<const double> velocity_, const Grid& grid_,
                   std::span<const Index> source_, int order_)
    : velocity(velocity_), grid(grid_), stride(grid_.shape.size()),
      source(source_.begin(), source_.end()), order(order_) {
    const std::size_t dim = grid.shape.size();
    stride[dim - 1] = 1; // C-order: the last axis is contiguous
    for (std::size_t k = dim - 1; k > 0; --k) {
        stride[k - 1] = stride[k] * grid.shape[k];
    }
    const Index n_nodes = stride[0] * grid.shape[0];

    for (std::size_t k = 0; k < dim; ++k) {
        source_node += source[k] * stride[k];
    }
    s0 = 1.0 / velocity[source_node];

    T0.resize(n_nodes);
    for (Index node = 0; node < n_nodes; ++node) {
        double distance_squared = 0.0;
        for (std::size_t k = 0; k < dim; ++k) {
            const double dx =
                static_cast<double>(coordinate(node, k) - source[k]) * grid.spacing[k];
            distance_squared += dx * dx;
        }
        T0[node] = s0 * std::sqrt(distance_squared);
    }
    tau.assign(n_nodes, infinity);
    accepted.assign(n_nodes, 0);
}

// New tau at `node` from its accepted neighbours (docs/FMM.md, "First-order upwind
// discretization" and "Second-order upwind discretization"). `node` is not the source and has
// at least one accepted neighbour.
double Marching::update(Index node) const {
    std::array<Term, 3> terms{}; // at most one term per axis
    std::size_t n_terms = 0;
    for (std::size_t k = 0; k < grid.shape.size(); ++k) {
        const Index i_k = coordinate(node, k);
        const Index n_k = grid.shape[k];

        // Upwind direction along k: toward the accepted neighbour with the smaller T; backward
        // wins a tie. sigma = -1 backward, +1 forward, 0 if no neighbour along k is accepted.
        int sigma = 0;
        if (i_k > 0 && accepted[node - stride[k]]) {
            sigma = -1;
        }
        if (i_k < n_k - 1 && accepted[node + stride[k]] &&
            (sigma == 0 || T(node + stride[k]) < T(node - stride[k]))) {
            sigma = 1;
        }
        if (sigma == 0) {
            continue; // the axis is left out
        }
        const Index upwind = node + sigma * stride[k];    // the upwind neighbour
        const Index upwind2 = upwind + sigma * stride[k]; // the node after it

        // Second order needs upwind2 inside the grid, accepted, and earlier than upwind.
        const bool upwind2_inside = sigma < 0 ? i_k >= 2 : i_k <= n_k - 3;
        const bool second_order =
            order == 2 && upwind2_inside && accepted[upwind2] && T(upwind2) < T(upwind);

        const double h = grid.spacing[k];
        const double x_k = static_cast<double>(i_k - source[k]) * h; // relative to the source
        const double dT0 = s0 * s0 * x_k / T0[node];                 // dT0/dx_k
        if (second_order) { // difference (3 tau_i - 4 tau_upwind + tau_upwind2) / (2h)
            const double alpha = 1.5 * T0[node] / h - sigma * dT0;
            const double beta = T0[node] * (4.0 * tau[upwind] - tau[upwind2]) / (2.0 * h * alpha);
            terms[n_terms++] = Term{alpha, beta};
        } else { // difference (tau_i - tau_upwind) / h
            const double alpha = T0[node] / h - sigma * dT0;
            terms[n_terms++] = Term{alpha, T0[node] * tau[upwind] / (h * alpha)};
        }
    }
    return detail::solve_update(std::span(terms).first(n_terms), 1.0 / velocity[node]);
}

// Accepts the nodes in order of increasing T (docs/FMM.md, "Marching").
void Marching::march() {
    // Min-heap of (T, node) pairs. Pairs compare by T, then by node, so ties pop in node order.
    using Entry = std::pair<double, Index>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> front;

    // Recomputes tau at a neighbour that is not accepted; pushes the node again if tau changed.
    auto update_neighbour = [&](Index neighbour) {
        if (accepted[neighbour]) {
            return;
        }
        const double new_tau = update(neighbour);
        if (new_tau != tau[neighbour]) {
            tau[neighbour] = new_tau;
            front.push({T(neighbour), neighbour});
        }
    };

    tau[source_node] = 1.0;
    front.push({0.0, source_node});
    while (!front.empty()) {
        const auto [popped_T, node] = front.top(); // a copy: pop() destroys the top entry
        front.pop();
        if (accepted[node] || popped_T != T(node)) {
            continue; // stale entry: the node was accepted or updated after this push
        }
        accepted[node] = 1;

        for (std::size_t k = 0; k < grid.shape.size(); ++k) {
            const Index i_k = coordinate(node, k);
            if (i_k > 0) {
                update_neighbour(node - stride[k]);
            }
            if (i_k < grid.shape[k] - 1) {
                update_neighbour(node + stride[k]);
            }
        }
    }
}

// Solves for one source and writes T = T0 * tau into `T` (one value per node). The inputs must
// already be checked.
void solve(std::span<const double> velocity, const Grid& grid, std::span<const Index> source,
           int order, std::span<double> T) {
    Marching marching(velocity, grid, source, order);
    marching.march();
    for (std::size_t node = 0; node < T.size(); ++node) {
        T[node] = marching.T0[node] * marching.tau[node];
    }
}

} // namespace

std::vector<double> fmm(std::span<const double> velocity, const Grid& grid,
                        std::span<const Index> source, int order) {
    check_grid(velocity, grid, order);
    check_source(grid, source);
    std::vector<double> T(velocity.size());
    solve(velocity, grid, source, order, T);
    return T;
}

std::vector<double> fmm_multi(std::span<const double> velocity, const Grid& grid,
                              std::span<const Index> sources, int order) {
    check_grid(velocity, grid, order);
    const std::size_t dim = grid.shape.size();
    if (sources.size() % dim != 0) {
        throw std::invalid_argument("fmm: sources need one index per axis for every source");
    }
    const auto n_sources = static_cast<Index>(sources.size() / dim);
    for (Index s = 0; s < n_sources; ++s) {
        check_source(grid, sources.subspan(static_cast<std::size_t>(s) * dim, dim));
    }

    const std::size_t n_nodes = velocity.size();
    std::vector<double> table(static_cast<std::size_t>(n_sources) * n_nodes);
    const std::span<double> all_T(table);

    // One source per iteration, each solved by one thread (docs/FMM.md, "Many sources").
    detail::parallel_for(n_sources, [&](Index s) {
        const auto i = static_cast<std::size_t>(s);
        solve(velocity, grid, sources.subspan(i * dim, dim), order,
              all_T.subspan(i * n_nodes, n_nodes));
    });
    return table;
}

} // namespace nes_cpp

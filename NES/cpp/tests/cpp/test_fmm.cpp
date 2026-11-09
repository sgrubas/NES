// Tests of nes_cpp::fmm and fmm_multi (docs/FMM.md): the update rule with hand-computed values;
// traveltimes against exact ones in a homogeneous medium (exact up to round-off) and in linear
// velocity gradients (convergence as the spacing halves); published errors of Treister & Haber
// (2016); fmm_multi against serial fmm.

#include "nes_cpp/fmm.hpp"

#include "fmm_update.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using nes_cpp::Grid;
using nes_cpp::Index;
using Point = std::vector<double>;

namespace {

Index node_count(const Grid& grid) {
    Index n = 1;
    for (const Index n_k : grid.shape) {
        n *= n_k;
    }
    return n;
}

// Position of the node with flat C-order index `node`.
Point position(const Grid& grid, Index node) {
    Point x(grid.shape.size());
    for (std::size_t k = x.size(); k-- > 0;) { // the last axis varies fastest
        x[k] = static_cast<double>(node % grid.shape[k]) * grid.spacing[k];
        node /= grid.shape[k];
    }
    return x;
}

// Position of the node with multi-index `index`.
Point position(const Grid& grid, const std::vector<Index>& index) {
    Point x(index.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] = static_cast<double>(index[k]) * grid.spacing[k];
    }
    return x;
}

double distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        sum += (a[k] - b[k]) * (a[k] - b[k]);
    }
    return std::sqrt(sum);
}

// f(x) at every node, flat C-order.
std::vector<double> sample(const Grid& grid, const std::function<double(const Point&)>& f) {
    std::vector<double> values(node_count(grid));
    for (Index node = 0; node < node_count(grid); ++node) {
        values[node] = f(position(grid, node));
    }
    return values;
}

// Grid with spacing h on the box [0, size[0]] x [0, size[1]] (x [0, size[2]]).
Grid grid_on_box(const Point& size, double h) {
    Grid grid;
    for (const double length : size) {
        grid.shape.push_back(static_cast<Index>(std::lround(length / h)) + 1);
        grid.spacing.push_back(h);
    }
    return grid;
}

// Multi-index of the node nearest to `point` on a grid with spacing h.
std::vector<Index> nearest_node(const Point& point, double h) {
    std::vector<Index> index;
    for (const double x : point) {
        index.push_back(static_cast<Index>(std::lround(x / h)));
    }
    return index;
}

// Exact traveltime between x_source and x when the velocity grows linearly in one direction with
// gradient magnitude a != 0; v_source and v_x are the velocities at the two points.
double linear_velocity_time(const Point& x_source, const Point& x, double v_source, double v_x,
                            double a) {
    const double r = distance(x_source, x);
    return std::acosh(1.0 + a * a * r * r / (2.0 * v_source * v_x)) / std::abs(a);
}

struct Errors {
    double max; // largest |T - T_exact|
    double rms; // root mean square of T - T_exact over all nodes
};

// Errors of fmm for v = v0 + a * x[axis] on the box `size` with spacing h, with the source at the
// node nearest to `source`.
Errors linear_velocity_errors(const Point& size, const Point& source, double h, std::size_t axis,
                              double v0, double a, int order) {
    const Grid grid = grid_on_box(size, h);
    const std::vector<Index> source_node = nearest_node(source, h);
    const auto v = [&](const Point& x) { return v0 + a * x[axis]; };
    const std::vector<double> T = nes_cpp::fmm(sample(grid, v), grid, source_node, order);

    const Point x_source = position(grid, source_node);
    Errors errors{0.0, 0.0};
    for (Index node = 0; node < node_count(grid); ++node) {
        const Point x = position(grid, node);
        const double error =
            std::abs(T[node] - linear_velocity_time(x_source, x, v(x_source), v(x), a));
        errors.max = std::max(errors.max, error);
        errors.rms += error * error;
    }
    errors.rms = std::sqrt(errors.rms / static_cast<double>(node_count(grid)));
    return errors;
}

} // namespace

TEST_CASE("solve_update takes the larger root and drops the largest beta when it is complex",
          "[fmm]") {
    using Catch::Matchers::WithinRel;
    using nes_cpp::detail::solve_update;
    using Terms = std::vector<nes_cpp::detail::Term>;

    // One term: tau = beta + s / alpha.
    Terms one{{2.0, 0.5}};
    CHECK(solve_update(one, 3.0) == 2.0);

    // Two upwind terms, alpha = 1: mean beta 0.1, V = 0.02, tau = 0.1 + sqrt(0.98 / 2) = 0.8.
    Terms upwind{{1.0, 0.0}, {1.0, 0.2}};
    CHECK_THAT(solve_update(upwind, 1.0), WithinRel(0.8, 1e-15));

    // Real roots, tau = 0.6 + sqrt(0.14) = 0.974 < beta = 1.2: the second term points downwind,
    // but it is kept, as in eikonalfm. (Treister & Haber's check would drop it and return 1.)
    Terms downwind{{1.0, 0.0}, {1.0, 1.2}};
    CHECK_THAT(solve_update(downwind, 1.0), WithinRel(0.6 + std::sqrt(0.14), 1e-15));

    // No real root (V = 2 > s^2 = 1): the term with beta = 2 is dropped.
    Terms complex_roots{{1.0, 0.0}, {1.0, 2.0}};
    CHECK(solve_update(complex_roots, 1.0) == 1.0);

    // Three terms given unsorted: the one with beta = 3 is dropped first, and the remaining two
    // are valid: tau = 0.05 + sqrt((1 - 0.005) / 2).
    Terms three{{1.0, 3.0}, {1.0, 0.0}, {1.0, 0.1}};
    CHECK_THAT(solve_update(three, 1.0), WithinRel(0.05 + std::sqrt(0.4975), 1e-15));
}

TEST_CASE("fmm is exact up to round-off in a homogeneous medium", "[fmm]") {
    struct Case {
        Grid grid;
        std::vector<Index> source;
    };
    // Unequal spacings; sources inside, on a corner and on a face; 2-D and 3-D.
    const std::vector<Case> cases = {
        {{{41, 31}, {0.01, 0.015}}, {17, 9}},
        {{{41, 31}, {0.01, 0.015}}, {0, 0}},
        {{{41, 31}, {0.01, 0.015}}, {40, 13}},
        {{{17, 13, 21}, {0.02, 0.03, 0.025}}, {8, 5, 11}},
        {{{17, 13, 21}, {0.02, 0.03, 0.025}}, {16, 12, 0}},
    };
    const double v = 2.5; // km/s

    for (std::size_t c = 0; c < cases.size(); ++c) {
        for (const int order : {1, 2}) {
            const Grid& grid = cases[c].grid;
            const std::vector<double> velocity(node_count(grid), v);
            const std::vector<double> T = nes_cpp::fmm(velocity, grid, cases[c].source, order);

            const Point x_source = position(grid, cases[c].source);
            double max_relative_error = 0.0;
            for (Index node = 0; node < node_count(grid); ++node) {
                const double exact = distance(x_source, position(grid, node)) / v;
                if (exact == 0.0) {
                    CHECK(T[node] == 0.0); // the source
                } else {
                    max_relative_error =
                        std::max(max_relative_error, std::abs(T[node] / exact - 1.0));
                }
            }
            INFO("case " << c << ", order " << order);
            CHECK(max_relative_error < 1e-11);
        }
    }
}

// v = 2 km/s + z * 1/s, sources inside the domain. The order-1 max error halves when h halves.
// The order-2 max error drops only by about 3, because nodes where the scheme falls back to first
// order (near the curves where a component of grad T changes sign) dominate it, as Treister &
// Haber (2016, Section 5) report; its RMS error drops by about 3.5 in 2-D and 3.35 in 3-D with
// these interior sources (about 4.5 with the source on the top surface, as in their tests). The
// 3-D grids start at h = 0.025: on coarser grids the ratios are not yet asymptotic.
TEST_CASE("fmm errors shrink with the spacing in a vertical gradient", "[fmm]") {
    struct Setup {
        Point size;             // km
        Point source;           // km, a node on every grid below
        std::vector<double> hs; // km
    };
    const std::vector<Setup> setups = {
        {{2.0, 1.0}, {0.5, 0.2}, {0.02, 0.01, 0.005}},
        {{1.0, 0.5, 0.5}, {0.25, 0.25, 0.1}, {0.025, 0.0125}},
    };
    using Catch::Matchers::WithinAbs;
    for (const Setup& s : setups) {
        const std::size_t z_axis = s.size.size() - 1;
        for (const int order : {1, 2}) {
            std::vector<Errors> e;
            for (const double h : s.hs) {
                e.push_back(linear_velocity_errors(s.size, s.source, h, z_axis, 2.0, 1.0, order));
            }
            for (std::size_t i = 1; i < e.size(); ++i) {
                const double max_ratio = e[i - 1].max / e[i].max;
                const double rms_ratio = e[i - 1].rms / e[i].rms;
                INFO(s.size.size() << "-D, order " << order << ", h " << s.hs[i - 1] << " -> "
                                   << s.hs[i] << ", RMS ratio " << rms_ratio);
                if (order == 1) {
                    CHECK_THAT(max_ratio, WithinAbs(2.0, 0.1));
                } else {
                    CHECK(max_ratio > 2.5);
                    CHECK(rms_ratio > 3.2);
                }
            }
        }
    }
}

// Test case 2 of Treister & Haber (2016) in 2-D: v = 1/s0 + a * x1 with s0 = 2 and a = 1 on
// [0, 4] x [0, 8], source at (0, 4) on the slow boundary. Their Table 2 lists [max, RMS] errors
// with three significant digits; ours must agree within one unit of the third digit. (Seven of the
// eight values round to the published ones; the first-order max error at h = 1/40 is 2.651e-2,
// published as 2.66e-2.)
TEST_CASE("fmm reproduces Table 2 of Treister & Haber (2016)", "[fmm]") {
    struct Row {
        double h;
        int order;
        double max; // published errors
        double rms;
    };
    const std::vector<Row> table = {
        {1.0 / 40, 1, 2.66e-2, 1.01e-2},
        {1.0 / 80, 1, 1.32e-2, 5.05e-3},
        {1.0 / 40, 2, 4.86e-4, 2.90e-4},
        {1.0 / 80, 2, 1.67e-4, 7.38e-5},
    };
    // One unit of the third significant digit of x, for example 0.01e-2 for 2.66e-2.
    const auto unit = [](double x) { return std::pow(10.0, std::floor(std::log10(x)) - 2); };

    using Catch::Matchers::WithinAbs;
    for (const Row& row : table) {
        const Errors e =
            linear_velocity_errors({4.0, 8.0}, {0.0, 4.0}, row.h, 0, 0.5, 1.0, row.order);
        INFO("h = 1/" << std::lround(1.0 / row.h) << ", order " << row.order);
        CHECK_THAT(e.max, WithinAbs(row.max, unit(row.max)));
        CHECK_THAT(e.rms, WithinAbs(row.rms, unit(row.rms)));
    }
}

// Each source is solved by one thread, so the table must not depend on the thread count
// (docs/FMM.md, "Many sources").
TEST_CASE("fmm_multi equals serial fmm bit for bit, for any thread count", "[fmm]") {
    struct Case {
        Grid grid;
        std::vector<Index> sources; // n_sources x dim, row-major
    };
    const std::vector<Case> cases = {
        {{{61, 41}, {0.01, 0.01}}, {0, 0, 30, 20, 60, 5, 12, 40, 45, 33}},
        {{{21, 17, 13}, {0.02, 0.02, 0.02}}, {0, 0, 0, 10, 8, 6, 20, 16, 12}},
    };
    for (const Case& c : cases) {
        const std::size_t dim = c.grid.shape.size();
        // A smooth, heterogeneous medium: 2 km/s plus a gradient and a ripple.
        const std::vector<double> velocity = sample(
            c.grid, [](const Point& x) { return 2.0 + x.back() + 0.3 * std::sin(10.0 * x[0]); });
        for (const int order : {1, 2}) {
            std::vector<double> serial;
            for (std::size_t s = 0; s < c.sources.size() / dim; ++s) {
                const std::span<const Index> source(c.sources.data() + s * dim, dim);
                const std::vector<double> T = nes_cpp::fmm(velocity, c.grid, source, order);
                serial.insert(serial.end(), T.begin(), T.end());
            }
#ifdef _OPENMP
            const int default_threads = omp_get_max_threads();
            for (const int threads : {1, 4}) {
                omp_set_num_threads(threads);
                INFO(dim << "-D, order " << order << ", " << threads << " threads");
                CHECK(nes_cpp::fmm_multi(velocity, c.grid, c.sources, order) == serial);
            }
            omp_set_num_threads(default_threads);
#else
            CHECK(nes_cpp::fmm_multi(velocity, c.grid, c.sources, order) == serial);
#endif
        }
    }
}

TEST_CASE("fmm rejects invalid input", "[fmm]") {
    const Grid grid{{3, 4}, {1.0, 1.0}};
    const std::vector<double> v(12, 1.0);
    const std::vector<Index> source{1, 2};
    CHECK_NOTHROW(nes_cpp::fmm(v, grid, source, 1));
    CHECK_NOTHROW(nes_cpp::fmm(v, grid, source, 2));

    using std::invalid_argument;
    CHECK_THROWS_AS(nes_cpp::fmm(v, Grid{{12}, {1.0}}, std::vector<Index>{1}, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, Grid{{3, 4}, {1.0}}, source, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, Grid{{3, 4}, {1.0, 0.0}}, source, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(std::vector<double>(11, 1.0), grid, source, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, grid, std::vector<Index>{1}, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, grid, std::vector<Index>{3, 0}, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, grid, std::vector<Index>{0, -1}, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, grid, source, 0), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm(v, grid, source, 3), invalid_argument);

    std::vector<double> bad_v = v;
    bad_v[5] = 0.0;
    CHECK_THROWS_AS(nes_cpp::fmm(bad_v, grid, source, 1), invalid_argument);
    bad_v[5] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(nes_cpp::fmm(bad_v, grid, source, 1), invalid_argument);

    // fmm_multi: the same rules, and a whole number of sources.
    CHECK(nes_cpp::fmm_multi(v, grid, std::vector<Index>{1, 2, 0, 0}, 1).size() == 24);
    CHECK(nes_cpp::fmm_multi(v, grid, std::vector<Index>{}, 1).empty());
    CHECK_THROWS_AS(nes_cpp::fmm_multi(v, grid, std::vector<Index>{1, 2, 0}, 1), invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm_multi(v, grid, std::vector<Index>{1, 2, 3, 0}, 1),
                    invalid_argument);
    CHECK_THROWS_AS(nes_cpp::fmm_multi(bad_v, grid, source, 1), invalid_argument);
}

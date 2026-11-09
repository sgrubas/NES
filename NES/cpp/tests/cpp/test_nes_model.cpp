// Tests of nes_cpp::NesModel (docs/NES_INFERENCE.md, docs/WEIGHTS_FORMAT.md) with weight files
// written by hand: the loader, the forward pass against a scalar re-implementation, every
// activation, malformed files, and independence of the thread count.

#include "nes_cpp/nes_model.hpp"

#include "weight_file.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using nes_cpp::NesModel;

namespace {

// NES-OP, dim 2, one hidden layer of 3 tanh units (docs/WEIGHTS_FORMAT.md lists the keys).
const std::vector<std::string> op_header = {
    "format = nes-weights", "version = 1",       "solver = NES_OP",     "dim = 2",
    "xs = 0.5 0.25",        "input_scale = 2.0", "factored = 1",        "out_vscale = 1",
    "slow_min = 0.25",      "slow_max = 0.5",    "widths = 2 3",        "act = tanh",
    "act_n = 1.0",          "act_a = 1.5",       "act_adaptive = 1",    "out_act = sigmoid",
    "out_act_n = 2.0",      "out_act_a = 0.75",  "out_act_adaptive = 1"};
const std::vector<double> op_weights = {0.1,  -0.2, 0.3,  0.4, 0.5, -0.6, // kernel, row by row
                                        0.01, 0.02, 0.03,                 // bias
                                        0.7,  -0.8, 0.9,                  // head kernel
                                        0.05};                            // head bias

// The same network written out with scalars (docs/NES_INFERENCE.md, "Network").
double op_reference(double x, double z) {
    const double d[2] = {x - 0.5, z - 0.25};
    const double W[2][3] = {{0.1, -0.2, 0.3}, {0.4, 0.5, -0.6}};
    const double b[3] = {0.01, 0.02, 0.03};
    const double w[3] = {0.7, -0.8, 0.9};
    double logit = 0.05;
    for (int j = 0; j < 3; ++j) {
        double zj = b[j];
        for (int i = 0; i < 2; ++i) {
            zj += 2.0 * d[i] * W[i][j]; // input_scale = 2
        }
        logit += std::tanh(1.5 * zj) * w[j]; // c = n * a = 1 * 1.5
    }
    const double tau = 1.0 / (1.0 + std::exp(-1.5 * logit)); // c = 2 * 0.75
    return (0.25 + (0.5 - 0.25) * tau) * std::hypot(d[0], d[1]);
}

// NES-TP, dim 2, one hidden layer of 2 sin units, sigmoid output without out_vscale.
std::vector<std::string> tp_header(const std::string& reciprocity) {
    return {"format = nes-weights",
            "version = 1",
            "solver = NES_TP",
            "dim = 2",
            "reciprocity = " + reciprocity,
            "input_scale = 0.5",
            "factored = 1",
            "out_vscale = 0",
            "slow_min = 0.2",
            "slow_max = 0.6",
            "widths = 4 2",
            "act = sin",
            "act_n = 1.0",
            "act_a = 1.0",
            "act_adaptive = 0",
            "out_act = sigmoid",
            "out_act_n = 1.0",
            "out_act_a = 1.0",
            "out_act_adaptive = 0"};
}
const std::vector<double> tp_weights = {0.3,  -0.1,  0.2, 0.4,  -0.5,
                                        0.25, 0.1,   0.6, // kernel 4 x 2
                                        0.05, -0.05, 1.2, -0.7, 0.1};

// Logit of the TP network for features [s, r] (before any swap averaging).
double tp_logit(const double s[2], const double r[2]) {
    const double f[4] = {0.5 * s[0], 0.5 * s[1], 0.5 * r[0], 0.5 * r[1]};
    const double W[4][2] = {{0.3, -0.1}, {0.2, 0.4}, {-0.5, 0.25}, {0.1, 0.6}};
    const double b[2] = {0.05, -0.05};
    const double w[2] = {1.2, -0.7};
    double logit = 0.1;
    for (int j = 0; j < 2; ++j) {
        double zj = b[j];
        for (int i = 0; i < 4; ++i) {
            zj += f[i] * W[i][j];
        }
        logit += std::sin(zj) * w[j];
    }
    return logit;
}

} // namespace

TEST_CASE("NesModel evaluates a hand-written NES-OP network", "[nes]") {
    const WeightFile file("op", op_header, op_weights);
    const NesModel model = NesModel::load(file.path);
    CHECK(model.dim() == 2);
    CHECK_FALSE(model.two_point());
    CHECK(model.n_inputs() == 2);

    const std::vector<double> points = {0.9, 0.1, 0.0, 1.0, 1.3, -0.4, 0.5, 0.25}; // last: source
    const std::vector<double> T = model.traveltime(points);
    REQUIRE(T.size() == 4);
    for (std::size_t p = 0; p < 3; ++p) {
        CHECK_THAT(T[p], WithinRel(op_reference(points[2 * p], points[2 * p + 1]), 1e-14));
    }
    CHECK(T[3] == 0.0); // factored: T = 0 at the source
}

TEST_CASE("NesModel evaluates NES-TP networks with and without reciprocity", "[nes]") {
    const double s[2] = {0.2, 0.7};
    const double r[2] = {0.9, -0.3};
    const double distance = std::hypot(r[0] - s[0], r[1] - s[1]);
    const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    const std::vector<double> pair = {s[0], s[1], r[0], r[1]};
    const std::vector<double> swapped = {r[0], r[1], s[0], s[1]};

    SECTION("none") {
        const WeightFile file("tp_none", tp_header("none"), tp_weights);
        const NesModel model = NesModel::load(file.path);
        CHECK(model.n_inputs() == 4);
        CHECK_THAT(model.traveltime(pair)[0], WithinRel(sigmoid(tp_logit(s, r)) * distance, 1e-14));
    }
    SECTION("output") {
        const WeightFile file("tp_output", tp_header("output"), tp_weights);
        const NesModel model = NesModel::load(file.path);
        const double logit = 0.5 * (tp_logit(s, r) + tp_logit(r, s)); // mean before out_act
        CHECK_THAT(model.traveltime(pair)[0], WithinRel(sigmoid(logit) * distance, 1e-14));
        CHECK(model.traveltime(pair)[0] == model.traveltime(swapped)[0]); // exact reciprocity
    }
}

TEST_CASE("NesModel applies every supported activation and its slope", "[nes]") {
    // T(x) = g(c * x0) with c = act_n * act_a = 2 * 0.5 = 1: one hidden unit with kernel [1, 0]
    // and bias 0, a linear head and output, no factorization or bounding, source at the origin.
    // So the gradient is (g'(x0), 0).
    struct Function {
        std::string name;
        std::function<double(double)> g;
        std::function<double(double)> slope;
    };
    const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    const std::vector<Function> functions = {
        {"gauss", [](double x) { return std::exp(-x * x); },
         [](double x) { return -2.0 * x * std::exp(-x * x); }},
        {"tanh", [](double x) { return std::tanh(x); },
         [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); }},
        {"sigmoid", sigmoid, [&](double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }},
        {"sin", [](double x) { return std::sin(x); }, [](double x) { return std::cos(x); }},
        {"swish", [](double x) { return x / (1.0 + std::exp(-x)); },
         [&](double x) { return sigmoid(x) + x * sigmoid(x) * (1.0 - sigmoid(x)); }},
        {"linear", [](double x) { return x; }, [](double) { return 1.0; }},
        {"softplus",
         [](double x) { return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); },
         sigmoid},
        {"elu", [](double x) { return x > 0 ? x : std::expm1(x); },
         [](double x) { return x > 0 ? 1.0 : std::exp(x); }},
        {"relu", [](double x) { return std::max(x, 0.0); },
         [](double x) { return x > 0 ? 1.0 : 0.0; }},
    };
    const std::vector<double> xs = {-40.0, -3.0, -0.5, -1e-3, 0.0, 1e-3, 0.7, 2.5, 40.0};

    for (const auto& [name, g, slope] : functions) {
        const std::vector<std::string> header = {"format = nes-weights",
                                                 "version = 1",
                                                 "solver = NES_OP",
                                                 "dim = 2",
                                                 "xs = 0 0",
                                                 "input_scale = 1",
                                                 "factored = 0",
                                                 "out_vscale = 0",
                                                 "slow_min = 1",
                                                 "slow_max = 1",
                                                 "widths = 2 1",
                                                 "act = " + name,
                                                 "act_n = 2",
                                                 "act_a = 0.5",
                                                 "act_adaptive = 1",
                                                 "out_act = linear",
                                                 "out_act_n = 1",
                                                 "out_act_a = 1",
                                                 "out_act_adaptive = 0"};
        const WeightFile file("act_" + name, header, {1.0, 0.0, 0.0, 1.0, 0.0});
        const NesModel model = NesModel::load(file.path);
        std::vector<double> points;
        for (const double x : xs) {
            points.insert(points.end(), {x, 0.3});
        }
        const std::vector<double> T = model.traveltime(points);
        const std::vector<double> G = model.gradient(points);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            INFO(name << "(" << xs[i] << ")");
            CHECK_THAT(T[i], WithinRel(g(xs[i]), 1e-14) || WithinAbs(g(xs[i]), 1e-300));
            CHECK_THAT(G[2 * i], WithinRel(slope(xs[i]), 1e-14) || WithinAbs(slope(xs[i]), 1e-300));
            CHECK(G[2 * i + 1] == 0.0); // T does not depend on x1
        }
    }
}

TEST_CASE("NesModel gradient agrees with central differences of T", "[nes]") {
    // Central differences with step 1e-6 have a truncation error of about 1e-12 and a rounding
    // error of about 1e-16 / 1e-6 = 1e-10 for these O(1) traveltimes, hence the 1e-8 tolerance.
    const double step = 1e-6;
    const auto central_differences = [&](const NesModel& model, const std::vector<double>& point) {
        std::vector<double> G;
        for (std::size_t j = 0; j < point.size(); ++j) {
            std::vector<double> plus = point;
            std::vector<double> minus = point;
            plus[j] += step;
            minus[j] -= step;
            G.push_back((model.traveltime(plus)[0] - model.traveltime(minus)[0]) / (2.0 * step));
        }
        return G;
    };
    const auto check_point = [&](const NesModel& model, const std::vector<double>& point) {
        const std::vector<double> G = model.gradient(point);
        const std::vector<double> expected = central_differences(model, point);
        REQUIRE(G.size() == expected.size());
        for (std::size_t j = 0; j < G.size(); ++j) {
            INFO("coordinate " << j);
            CHECK_THAT(G[j], WithinAbs(expected[j], 1e-8));
        }
    };

    SECTION("NES-OP") {
        const WeightFile file("op_gradient", op_header, op_weights);
        const NesModel model = NesModel::load(file.path);
        check_point(model, {0.9, 0.1});
        check_point(model, {0.0, 1.0});
        check_point(model, {1.3, -0.4});
        CHECK(model.gradient(std::vector<double>{0.5, 0.25}) == std::vector<double>{0.0, 0.0});
    }
    SECTION("NES-TP") {
        for (const std::string reciprocity : {"none", "output"}) {
            INFO("reciprocity " << reciprocity);
            const WeightFile file("tp_gradient_" + reciprocity, tp_header(reciprocity), tp_weights);
            const NesModel model = NesModel::load(file.path);
            check_point(model, {0.2, 0.7, 0.9, -0.3});
            check_point(model, {-0.4, 0.1, 0.3, 0.8});
        }
    }
    SECTION("NES-TP reciprocity: dT/dxs at (s, r) equals dT/dxr at (r, s)") {
        const WeightFile file("tp_gradient_swap", tp_header("output"), tp_weights);
        const NesModel model = NesModel::load(file.path);
        const std::vector<double> G = model.gradient(std::vector<double>{0.2, 0.7, 0.9, -0.3});
        const std::vector<double> G_swapped =
            model.gradient(std::vector<double>{0.9, -0.3, 0.2, 0.7});
        CHECK(G[0] == G_swapped[2]); // the same two terms, added in the other order
        CHECK(G[1] == G_swapped[3]);
        CHECK(G[2] == G_swapped[0]);
        CHECK(G[3] == G_swapped[1]);
    }
}

TEST_CASE("NesModel rejects malformed weight files", "[nes]") {
    // `header` with the line starting with `key` replaced by `line` (removed if `line` is empty).
    const auto changed = [](const std::string& key, const std::string& line) {
        std::vector<std::string> header;
        for (const std::string& h : op_header) {
            if (h.rfind(key + " =", 0) != 0) {
                header.push_back(h);
            } else if (!line.empty()) {
                header.push_back(line);
            }
        }
        return header;
    };
    std::vector<double> short_weights(op_weights.begin(), op_weights.end() - 1);
    std::vector<double> long_weights = op_weights;
    long_weights.push_back(0.0);
    std::vector<std::string> unknown = op_header;
    unknown.push_back("colour = blue");

    const std::vector<std::pair<std::vector<std::string>, std::vector<double>>> bad = {
        {changed("out_act_a", ""), op_weights},              // missing key
        {unknown, op_weights},                               // unknown key
        {changed("version", "version = 2"), op_weights},     // other version
        {changed("act", "act = mish"), op_weights},          // unsupported activation
        {changed("widths", "widths = 3 3"), op_weights},     // widths do not fit dim
        {changed("act_a", "act_a = 1 2"), op_weights},       // one a per hidden layer
        {changed("slow_min", "slow_min = abc"), op_weights}, // not a number
        {changed("factored", "factored = 2"), op_weights},   // not a flag
        {op_header, short_weights},                          // truncated
        {op_header, long_weights},                           // extra bytes
    };
    for (std::size_t i = 0; i < bad.size(); ++i) {
        const WeightFile file("bad_" + std::to_string(i), bad[i].first, bad[i].second);
        INFO("case " << i);
        CHECK_THROWS_AS(NesModel::load(file.path), std::invalid_argument);
    }
    CHECK_THROWS_AS(NesModel::load("no/such/file.nesw"), std::invalid_argument);

    const WeightFile file("op_good", op_header, op_weights);
    const NesModel model = NesModel::load(file.path);
    CHECK_THROWS_AS(model.traveltime(std::vector<double>{0.1, 0.2, 0.3}), std::invalid_argument);
    CHECK_THROWS_AS(model.gradient(std::vector<double>{0.1, 0.2, 0.3}), std::invalid_argument);
    CHECK(model.traveltime(std::vector<double>{}).empty());
    CHECK(model.gradient(std::vector<double>{}).empty());
}

TEST_CASE("NesModel results do not depend on the number of threads", "[nes]") {
    const WeightFile file("op_threads", op_header, op_weights);
    const NesModel model = NesModel::load(file.path);
    std::mt19937 random(7);
    std::uniform_real_distribution<double> uniform(-1.0, 2.0);
    std::vector<double> points(2 * 3000); // many batches of points
    std::generate(points.begin(), points.end(), [&] { return uniform(random); });

#ifdef _OPENMP
    const int default_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const std::vector<double> T_one = model.traveltime(points);
    const std::vector<double> G_one = model.gradient(points);
    omp_set_num_threads(4);
    const std::vector<double> T_four = model.traveltime(points);
    const std::vector<double> G_four = model.gradient(points);
    omp_set_num_threads(default_threads);
    CHECK(T_one == T_four);
    CHECK(G_one == G_four);
#else
    SKIP("built without OpenMP");
#endif
}

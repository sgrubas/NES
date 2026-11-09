// Tests of NesModel training (docs/NES_TRAINING.md): the loss gradient against central differences
// for every activation and option, Adam against the formula, save and load, and reproducibility.

#include "nes_cpp/nes_model.hpp"

#include "weight_file.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::WithinAbs;
using nes_cpp::NesModel;
using nes_cpp::TrainOptions;

namespace {

// A small NES network with random weights (docs/WEIGHTS_FORMAT.md).
struct Network {
    bool two_point = false;
    std::string reciprocity = "none";
    std::string act = "tanh";
    bool act_adaptive = true;
    std::string out_act = "sigmoid";
    bool out_adaptive = true;
    bool factored = true;
    bool out_vscale = true;
    std::vector<int> hidden = {5, 4};
    double slow_min = 0.25;
    double slow_max = 1.0;

    std::vector<std::string> header() const {
        const int n_in = two_point ? 4 : 2;
        std::string widths = std::to_string(n_in);
        std::string act_a;
        for (std::size_t k = 0; k < hidden.size(); ++k) {
            widths += " " + std::to_string(hidden[k]);
            act_a += (k ? " " : "") + std::to_string(act_adaptive ? 1.2 - 0.3 * k : 1.0);
        }
        std::vector<std::string> lines = {
            "format = nes-weights", "version = 1",
            std::string("solver = ") + (two_point ? "NES_TP" : "NES_OP"), "dim = 2"};
        lines.push_back(two_point ? "reciprocity = " + reciprocity : "xs = 0.5 0.25");
        lines.insert(lines.end(), {"input_scale = 0.8", "factored = " + std::to_string(factored),
                                   "out_vscale = " + std::to_string(out_vscale),
                                   "slow_min = " + std::to_string(slow_min),
                                   "slow_max = " + std::to_string(slow_max), "widths = " + widths,
                                   "act = " + act, "act_n = 1.5", "act_a = " + act_a,
                                   "act_adaptive = " + std::to_string(act_adaptive),
                                   "out_act = " + out_act, "out_act_n = 1.0",
                                   std::string("out_act_a = ") + (out_adaptive ? "0.9" : "1.0"),
                                   "out_act_adaptive = " + std::to_string(out_adaptive)});
        return lines;
    }

    std::vector<double> weights(unsigned seed) const {
        std::mt19937 random(seed);
        std::normal_distribution<double> normal(0.0, 0.6);
        std::size_t count = 0;
        int n_in = two_point ? 4 : 2;
        for (const int n_out : hidden) {
            count += static_cast<std::size_t>((n_in + 1) * n_out);
            n_in = n_out;
        }
        count += static_cast<std::size_t>(n_in + 1); // head
        std::vector<double> values(count);
        std::generate(values.begin(), values.end(), [&] { return normal(random); });
        return values;
    }
};

// Points away from the source, and a smooth velocity at their receivers.
void make_points(bool two_point, int n, std::vector<double>& points,
                 std::vector<double>& velocity) {
    std::mt19937 random(3);
    std::uniform_real_distribution<double> uniform(-0.5, 1.5);
    points.clear();
    velocity.clear();
    for (int p = 0; p < n; ++p) {
        const int n_in = two_point ? 4 : 2;
        for (int j = 0; j < n_in; ++j) {
            points.push_back(uniform(random));
        }
        const double z = points.back(); // receiver depth
        velocity.push_back(1.5 + 0.5 * z * z);
    }
}

// Central differences with step 1e-6 of a loss of order 1: truncation about 1e-12, rounding about
// 1e-16 / 1e-6 = 1e-10, hence the tolerance 1e-8.
void check_gradient_against_differences(NesModel& model, const std::vector<double>& points,
                                        const std::vector<double>& velocity) {
    const std::vector<double> theta = model.parameters();
    std::vector<double> gradient(theta.size());
    std::vector<double> unused(theta.size());
    model.loss_and_gradient(points, velocity, gradient);
    const double step = 1e-6;
    for (std::size_t i = 0; i < theta.size(); ++i) {
        std::vector<double> shifted = theta;
        shifted[i] = theta[i] + step;
        model.set_parameters(shifted);
        const double plus = model.loss_and_gradient(points, velocity, unused);
        shifted[i] = theta[i] - step;
        model.set_parameters(shifted);
        const double minus = model.loss_and_gradient(points, velocity, unused);
        INFO("parameter " << i << " of " << theta.size());
        CHECK_THAT(gradient[i], WithinAbs((plus - minus) / (2.0 * step), 1e-8));
    }
    model.set_parameters(theta);
}

} // namespace

TEST_CASE("Loss gradient agrees with central differences", "[train]") {
    std::vector<Network> networks;
    networks.push_back(Network{}); // NES-OP, adaptive a
    networks.push_back(Network{.two_point = true,
                               .reciprocity = "output",
                               .act = "sin",
                               .out_adaptive = false}); // NES-TP, swap mean
    networks.push_back(Network{.two_point = true,
                               .act = "softplus",
                               .out_act = "linear",
                               .out_adaptive = false,
                               .factored = false,
                               .out_vscale = false}); // options off
    for (const std::string act : {"gauss", "sigmoid", "swish", "linear", "elu", "relu"}) {
        networks.push_back(Network{.act = act});
    }
    for (const std::string out_act : {"tanh", "softplus", "gauss"}) {
        networks.push_back(Network{.out_act = out_act});
    }
    for (std::size_t c = 0; c < networks.size(); ++c) {
        const Network& network = networks[c];
        INFO("network " << c << ": " << network.act << " / " << network.out_act
                        << (network.two_point ? ", NES-TP " + network.reciprocity : ", NES-OP"));
        const WeightFile file("train_" + std::to_string(c), network.header(),
                              network.weights(static_cast<unsigned>(c)));
        NesModel model = NesModel::load(file.path);
        std::vector<double> points;
        std::vector<double> velocity;
        make_points(network.two_point, 6, points, velocity);
        check_gradient_against_differences(model, points, velocity);
    }
}

TEST_CASE("Two train steps are two Adam steps", "[train]") {
    const Network network;
    const WeightFile file("adam", network.header(), network.weights(1));
    NesModel model = NesModel::load(file.path);
    NesModel reference = NesModel::load(file.path);
    std::vector<double> points;
    std::vector<double> velocity;
    make_points(false, 40, points, velocity);

    TrainOptions options;
    options.epochs = 2;
    options.batch_size = 40; // one batch per epoch: the order of the points does not matter
    options.learning_rate = 0.01;
    options.decay = 0.5; // visible in the second step
    model.train(points, velocity, options);

    // Adam written out from docs/NES_TRAINING.md, "Optimiser".
    std::vector<double> theta = reference.parameters();
    std::vector<double> m(theta.size(), 0.0);
    std::vector<double> v(theta.size(), 0.0);
    std::vector<double> g(theta.size());
    for (int t = 0; t < 2; ++t) {
        reference.set_parameters(theta);
        reference.loss_and_gradient(points, velocity, g);
        const double rate = options.learning_rate / (1.0 + options.decay * t);
        const double correction = std::sqrt(1.0 - std::pow(options.beta2, t + 1)) /
                                  (1.0 - std::pow(options.beta1, t + 1));
        for (std::size_t i = 0; i < theta.size(); ++i) {
            m[i] = options.beta1 * m[i] + (1.0 - options.beta1) * g[i];
            v[i] = options.beta2 * v[i] + (1.0 - options.beta2) * g[i] * g[i];
            theta[i] -= rate * correction * m[i] / (std::sqrt(v[i]) + options.epsilon);
        }
    }
    const std::vector<double> trained = model.parameters();
    for (std::size_t i = 0; i < theta.size(); ++i) {
        INFO("parameter " << i);
        CHECK_THAT(trained[i], WithinAbs(theta[i], 1e-12));
    }
}

TEST_CASE("Training lowers the loss, with the same result for any number of threads", "[train]") {
    // Homogeneous medium, v = 2: the exact factor is tau = 0.5 everywhere.
    Network network;
    network.hidden = {16, 16};
    const WeightFile file("train_loss", network.header(), network.weights(2));
    std::vector<double> points;
    std::vector<double> velocity;
    make_points(false, 512, points, velocity);
    std::fill(velocity.begin(), velocity.end(), 2.0);
    TrainOptions options;
    options.epochs = 30;
    options.batch_size = 64;
    options.learning_rate = 0.01;

    NesModel model = NesModel::load(file.path);
#ifdef _OPENMP
    const int default_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const std::vector<double> history = model.train(points, velocity, options);
#ifdef _OPENMP
    omp_set_num_threads(4);
    NesModel model_4_threads = NesModel::load(file.path);
    const std::vector<double> history_4_threads = model_4_threads.train(points, velocity, options);
    omp_set_num_threads(default_threads);
    CHECK(history == history_4_threads);
    CHECK(model.parameters() == model_4_threads.parameters());
#endif
    REQUIRE(history.size() == 30);
    INFO("loss " << history.front() << " -> " << history.back());
    CHECK(history.back() < 0.2 * history.front());
}

TEST_CASE("save writes a file that load reads back exactly", "[train]") {
    for (const bool two_point : {false, true}) {
        Network network;
        network.two_point = two_point;
        network.reciprocity = "output";
        const WeightFile file(std::string("save_") + (two_point ? "tp" : "op"), network.header(),
                              network.weights(4));
        const NesModel model = NesModel::load(file.path);
        const auto saved = std::filesystem::temp_directory_path() / "nes_cpp_test_saved.nesw";
        model.save(saved);
        const NesModel loaded = NesModel::load(saved);
        std::filesystem::remove(saved);
        std::vector<double> points;
        std::vector<double> velocity;
        make_points(two_point, 20, points, velocity);
        CHECK(loaded.parameters() == model.parameters());
        CHECK(loaded.traveltime(points) == model.traveltime(points));
        CHECK(loaded.gradient(points) == model.gradient(points));
    }
}

TEST_CASE("Training functions reject inconsistent input", "[train]") {
    const Network network;
    const WeightFile file("train_bad", network.header(), network.weights(5));
    NesModel model = NesModel::load(file.path);
    std::vector<double> points;
    std::vector<double> velocity;
    make_points(false, 4, points, velocity);
    std::vector<double> gradient(model.parameters().size());
    using std::invalid_argument;
    CHECK_THROWS_AS(model.set_parameters(std::vector<double>(3)), invalid_argument);
    CHECK_THROWS_AS(model.loss_and_gradient(points, std::vector<double>(3), gradient),
                    invalid_argument);
    std::vector<double> short_gradient(2);
    CHECK_THROWS_AS(model.loss_and_gradient(points, velocity, short_gradient), invalid_argument);
    TrainOptions bad;
    bad.batch_size = 0;
    CHECK_THROWS_AS(model.train(points, velocity, bad), invalid_argument);
    CHECK_THROWS_AS(model.train(points, std::vector<double>(1), TrainOptions{}), invalid_argument);
}

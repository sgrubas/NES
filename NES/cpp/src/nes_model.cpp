// NES inference: loading and saving a network, its activations, and T and its gradient in batches
// (docs/NES_INFERENCE.md). The weight file itself is read by weights_file.hpp. Training
// (nes_train.cpp) uses the same network pass and batch evaluation, with order 2.

#include "nes_cpp/nes_model.hpp"

#include "parallel.hpp"
#include "weights_file.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace nes_cpp {
namespace { // helpers used only in this file

// Points per batch (docs/NES_INFERENCE.md, "Batched evaluation"). A gradient batch keeps one slope
// matrix per layer; with 128 rows these stay in a core's L2 cache.
constexpr Index batch_size = 128;

// The names in the weight file, in the order of NesModel::Function.
constexpr const char* activation_names[] = {"gauss",  "tanh",     "sigmoid", "sin", "swish",
                                            "linear", "softplus", "elu",     "relu"};

// The shortest text that reads back as the same double, as Python's repr writes it.
std::string number(double x) {
    char text[32];
    const auto [end, error] = std::to_chars(text, text + sizeof text, x);
    return std::string(text, end);
}

} // namespace

NesModel::Function NesModel::function_named(const std::string& name) {
    for (int i = 0; i < std::ssize(activation_names); ++i) {
        if (name == activation_names[i]) {
            return static_cast<Function>(i);
        }
    }
    throw std::invalid_argument("unsupported activation '" + name + "'");
}

NesModel NesModel::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::invalid_argument("NesModel: cannot open " + path.string());
    }
    detail::WeightsHeader header(file, path.string());
    if (header.text("format") != "nes-weights" || header.number<int>("version") != 1) {
        header.fail("not a version 1 nes-weights file");
    }

    NesModel model;
    const std::string solver = header.text("solver");
    if (solver != "NES_OP" && solver != "NES_TP") {
        header.fail("solver must be NES_OP or NES_TP");
    }
    model.two_point_ = solver == "NES_TP";
    model.dim_ = header.number<int>("dim");
    if (model.dim_ != 2 && model.dim_ != 3) {
        header.fail("dim must be 2 or 3");
    }
    if (model.two_point_) {
        const std::string reciprocity = header.text("reciprocity");
        if (reciprocity != "none" && reciprocity != "output") {
            header.fail("reciprocity must be none or output");
        }
        model.swap_mean_ = reciprocity == "output";
    } else {
        const std::vector<double> xs = header.numbers<double>("xs");
        if (std::ssize(xs) != model.dim_) {
            header.fail("xs needs dim values");
        }
        model.xs_ = Eigen::Map<const Eigen::RowVectorXd>(xs.data(), model.dim_);
    }
    model.input_scale_ = header.number<double>("input_scale");
    model.factored_ = header.flag("factored");
    model.out_vscale_ = header.flag("out_vscale");
    model.slow_min_ = header.number<double>("slow_min");
    model.slow_max_ = header.number<double>("slow_max");

    const std::vector<Index> widths = header.numbers<Index>("widths");
    if (widths.size() < 2 || widths[0] != model.n_inputs() ||
        std::any_of(widths.begin(), widths.end(), [](Index w) { return w < 1; })) {
        header.fail("widths must be the number of inputs, then >= 1 hidden widths");
    }
    const auto n_hidden = static_cast<Index>(widths.size()) - 1;

    Activation act;
    Activation& out = model.weights_.out_activation;
    try {
        act.function = function_named(header.text("act"));
        out.function = function_named(header.text("out_act"));
    } catch (const std::invalid_argument& e) {
        header.fail(e.what());
    }
    act.n = header.number<double>("act_n");
    act.adaptive = header.flag("act_adaptive");
    const std::vector<double> act_a = header.numbers<double>("act_a");
    if (std::ssize(act_a) != n_hidden) {
        header.fail("act_a needs one value per hidden layer");
    }
    out.n = header.number<double>("out_act_n");
    out.a = header.number<double>("out_act_a");
    out.adaptive = header.flag("out_act_adaptive");
    header.check_all_read();

    for (Index k = 0; k < n_hidden; ++k) {
        const Index n_in = widths[k];
        const Index n_out = widths[k + 1];
        const std::vector<double> kernel = detail::read_doubles(file, n_in * n_out, header);
        const std::vector<double> bias = detail::read_doubles(file, n_out, header);
        Layer layer;
        layer.kernel = Eigen::Map<const RowMajorMatrix>(kernel.data(), n_in, n_out);
        layer.bias = Eigen::Map<const Eigen::RowVectorXd>(bias.data(), n_out);
        layer.activation = act;
        layer.activation.a = act_a[k];
        model.weights_.hidden.push_back(std::move(layer));
    }
    const std::vector<double> head_kernel = detail::read_doubles(file, widths.back(), header);
    model.weights_.head_kernel =
        Eigen::Map<const Eigen::VectorXd>(head_kernel.data(), widths.back());
    model.weights_.head_bias = detail::read_doubles(file, 1, header)[0];
    if (file.peek() != std::ifstream::traits_type::eof()) {
        header.fail("extra bytes after the weights");
    }
    return model;
}

void NesModel::save(const std::filesystem::path& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::invalid_argument("NesModel: cannot write " + path.string());
    }
    const auto row = [](const auto& values) { // space-separated numbers
        std::string text;
        for (const double x : values) {
            text += (text.empty() ? "" : " ") + number(x);
        }
        return text;
    };
    const Activation& act = weights_.hidden.front().activation;
    const Activation& out = weights_.out_activation;
    std::vector<double> widths{static_cast<double>(n_inputs())};
    std::vector<double> act_a;
    for (const Layer& layer : weights_.hidden) {
        widths.push_back(static_cast<double>(layer.kernel.cols()));
        act_a.push_back(layer.activation.a);
    }
    file << "format = nes-weights\nversion = 1\n"
         << "solver = " << (two_point_ ? "NES_TP" : "NES_OP") << "\ndim = " << dim_ << "\n";
    if (two_point_) {
        file << "reciprocity = " << (swap_mean_ ? "output" : "none") << "\n";
    } else {
        file << "xs = " << row(xs_) << "\n";
    }
    file << "input_scale = " << number(input_scale_) << "\nfactored = " << factored_
         << "\nout_vscale = " << out_vscale_ << "\nslow_min = " << number(slow_min_)
         << "\nslow_max = " << number(slow_max_) << "\nwidths = " << row(widths)
         << "\nact = " << activation_names[static_cast<int>(act.function)]
         << "\nact_n = " << number(act.n) << "\nact_a = " << row(act_a)
         << "\nact_adaptive = " << act.adaptive
         << "\nout_act = " << activation_names[static_cast<int>(out.function)]
         << "\nout_act_n = " << number(out.n) << "\nout_act_a = " << number(out.a)
         << "\nout_act_adaptive = " << out.adaptive << "\nend_header\n";
    const auto write = [&](const double* values, Index count) {
        file.write(reinterpret_cast<const char*>(values),
                   static_cast<std::streamsize>(count * Index{sizeof(double)}));
    };
    for (const Layer& layer : weights_.hidden) {
        write(layer.kernel.data(), layer.kernel.size()); // row-major, as the format wants
        write(layer.bias.data(), layer.bias.size());
    }
    write(weights_.head_kernel.data(), weights_.head_kernel.size());
    write(&weights_.head_bias, 1);
    if (!file) {
        throw std::invalid_argument("NesModel: cannot write " + path.string());
    }
}

// g, g' and g'' of `function` at x, elementwise, up to `order` (docs/NES_INFERENCE.md,
// "Activations" and "Gradient"; docs/NES_TRAINING.md, "Activation derivatives").
void NesModel::evaluate(Function function, const Eigen::ArrayXXd& x, int order, Derivatives& d) {
    switch (function) {
    case Function::gauss:
        d.g = (-x.square()).exp();
        if (order >= 1) {
            d.dg = -2.0 * x * d.g;
        }
        if (order >= 2) {
            d.d2g = (4.0 * x.square() - 2.0) * d.g;
        }
        break;
    case Function::tanh:
        d.g = x.tanh();
        if (order >= 1) {
            d.dg = 1.0 - d.g.square();
        }
        if (order >= 2) {
            d.d2g = -2.0 * d.g * d.dg;
        }
        break;
    case Function::sigmoid:
        d.g = 1.0 / (1.0 + (-x).exp());
        if (order >= 1) {
            d.dg = d.g * (1.0 - d.g);
        }
        if (order >= 2) {
            d.d2g = d.dg * (1.0 - 2.0 * d.g);
        }
        break;
    case Function::sin:
        d.g = x.sin();
        if (order >= 1) {
            d.dg = x.cos();
        }
        if (order >= 2) {
            d.d2g = -d.g;
        }
        break;
    case Function::swish: {
        const Eigen::ArrayXXd sigmoid = 1.0 / (1.0 + (-x).exp());
        d.g = x * sigmoid;
        if (order >= 1) {
            d.dg = sigmoid + x * sigmoid * (1.0 - sigmoid);
        }
        if (order >= 2) {
            d.d2g = sigmoid * (1.0 - sigmoid) * (2.0 + x * (1.0 - 2.0 * sigmoid));
        }
        break;
    }
    case Function::linear:
        d.g = x;
        if (order >= 1) {
            d.dg.setOnes(x.rows(), x.cols());
        }
        if (order >= 2) {
            d.d2g.setZero(x.rows(), x.cols());
        }
        break;
    case Function::softplus: // max(x, 0) + log(1 + exp(-|x|)): no overflow for large x
        d.g = x.max(0.0) + (-x.abs()).exp().log1p();
        if (order >= 1) {
            d.dg = 1.0 / (1.0 + (-x).exp()); // sigmoid(x)
        }
        if (order >= 2) {
            d.d2g = d.dg * (1.0 - d.dg);
        }
        break;
    case Function::elu: // expm1 is accurate near 0; select() picks x for x > 0
        d.g = (x > 0.0).select(x, x.expm1());
        if (order >= 1) {
            d.dg = (x > 0.0).select(1.0, x.exp());
        }
        if (order >= 2) {
            d.d2g = (x > 0.0).select(0.0, x.exp());
        }
        break;
    case Function::relu:
        d.g = x.max(0.0);
        if (order >= 1) {
            d.dg = (x > 0.0).cast<double>(); // 0 at x = 0, as JAX
        }
        if (order >= 2) {
            d.d2g.setZero(x.rows(), x.cols());
        }
        break;
    }
}

// The network over the feature rows `h` (docs/NES_INFERENCE.md, "Hidden layers", "Head" and
// "Gradient"); `order` as in Pass.
NesModel::Pass NesModel::network_pass(Eigen::ArrayXXd h, int order) const {
    Pass pass;
    for (const Layer& layer : weights_.hidden) {
        // z = h W + b: one matrix product per layer
        Eigen::ArrayXXd z = ((h.matrix() * layer.kernel).rowwise() + layer.bias).array();
        Derivatives d;
        evaluate(layer.activation.function, z * layer.activation.c(), order, d);
        if (order == 2) {
            pass.h.push_back(std::move(h));
            pass.z.push_back(std::move(z));
        }
        h = std::move(d.g);
        if (order >= 1) {
            pass.act.push_back(std::move(d)); // the slopes, for the backward pass
        }
    }
    pass.logit = (h.matrix() * weights_.head_kernel).array() + weights_.head_bias;
    if (order == 0) {
        return pass;
    }
    const auto L = weights_.hidden.size();
    if (order == 2) {
        pass.h.push_back(std::move(h));
        pass.g.resize(L);
        pass.q.resize(L);
    }
    // g_L = w for every row, then q_k = g_k * c g'(c z_k) and g_(k-1) = q_k W_k^T
    Eigen::ArrayXXd g = weights_.head_kernel.transpose().replicate(pass.logit.rows(), 1).array();
    for (std::size_t k = L; k-- > 0;) {
        const Layer& layer = weights_.hidden[k];
        Eigen::ArrayXXd q = g * (layer.activation.c() * pass.act[k].dg);
        if (order == 2) {
            pass.g[k] = std::move(g);
            pass.q[k] = q;
        }
        g = (q.matrix() * layer.kernel.transpose()).array();
    }
    pass.dlogit = std::move(g);
    return pass;
}

// One batch of points, one per row (docs/NES_INFERENCE.md, "Network" and "Gradient"); `order`
// as in Pass.
NesModel::Batch NesModel::evaluate_batch(const Eigen::MatrixXd& points, int order) const {
    const Index n = points.rows();
    Batch b;
    // NES_TP rows: source, then receiver; NES_OP rows: the receiver (the source is fixed)
    const Eigen::MatrixXd xr = points.rightCols(dim_);
    const Eigen::MatrixXd xs =
        two_point_ ? Eigen::MatrixXd(points.leftCols(dim_)) : Eigen::MatrixXd(xs_.replicate(n, 1));
    const Eigen::MatrixXd d = xr - xs;
    b.pass = network_pass(((two_point_ ? points : d) * input_scale_).array(), order);
    b.logit = b.pass.logit;
    b.dlogit = b.pass.dlogit;
    if (swap_mean_) { // reciprocity 'output': mean over swapping source and receiver
        Eigen::MatrixXd swapped(n, 2 * dim_);
        swapped << xr, xs;
        b.swapped = network_pass((swapped * input_scale_).array(), order);
        b.logit = 0.5 * (b.logit + b.swapped.logit);
        if (order >= 1) { // the swapped pass saw [xr, xs]: exchange its halves back
            b.dlogit = 0.5 * (b.dlogit + exchange_halves(b.swapped.dlogit));
        }
    }

    const Activation& out = weights_.out_activation;
    evaluate(out.function, b.logit * out.c(), order, b.psi);
    if (out_vscale_) {
        b.scale = slow_max_ - slow_min_;
    }
    const Eigen::ArrayXd tau = (out_vscale_ ? slow_min_ : 0.0) + b.scale * b.psi.g.col(0);
    b.r = factored_ ? Eigen::ArrayXd(d.rowwise().norm().array()) : Eigen::ArrayXd::Ones(n);
    b.T = b.r * tau;
    if (order == 0) {
        return b;
    }
    b.dlogit *= input_scale_; // features = s * inputs
    b.gamma = b.r * b.scale * out.c() * b.psi.dg.col(0);
    // dr/du: d_k / r for receiver coordinate k, -d_k / r for source coordinate k, and 0 at the
    // source, as in NES
    b.dr.setZero(n, n_inputs());
    if (factored_) {
        const Eigen::ArrayXd inverse_r = (b.r > 0.0).select(1.0 / b.r, 0.0);
        for (Index j = 0; j < n_inputs(); ++j) {
            const double sign = two_point_ && j < dim_ ? -1.0 : 1.0;
            b.dr.col(j) = sign * d.col(j % dim_).array() * inverse_r;
        }
    }
    b.grad_T = (b.dlogit.colwise() * b.gamma) + (b.dr.colwise() * tau);
    return b;
}

// Number of points in `points`; throws if the size is not a multiple of n_inputs().
Index NesModel::count_points(std::span<const double> points) const {
    const auto n_in = static_cast<Index>(n_inputs());
    if (std::ssize(points) % n_in != 0) {
        throw std::invalid_argument("NesModel: points need " + std::to_string(n_in) +
                                    " coordinates each");
    }
    return std::ssize(points) / n_in;
}

// All points in fixed batches, whatever the number of threads, so that every point gets the same
// operations; the batches run in parallel. Computes the gradient too if `G` is not empty.
void NesModel::evaluate(std::span<const double> points, std::span<double> T,
                        std::span<double> G) const {
    const auto n_in = static_cast<Index>(n_inputs());
    const auto n = static_cast<Index>(T.size());
    const Index n_batches = (n + batch_size - 1) / batch_size;
    detail::parallel_for(n_batches, [&](Index k) {
        const Index first = k * batch_size;
        const Index count = std::min(batch_size, n - first);
        const Batch b = evaluate_batch(
            Eigen::Map<const RowMajorMatrix>(points.data() + first * n_in, count, n_in),
            G.empty() ? 0 : 1);
        Eigen::Map<Eigen::ArrayXd>(T.data() + first, count) = b.T;
        if (!G.empty()) {
            Eigen::Map<RowMajorMatrix>(G.data() + first * n_in, count, n_in) = b.grad_T.matrix();
        }
    });
}

std::vector<double> NesModel::traveltime(std::span<const double> points) const {
    std::vector<double> T(static_cast<std::size_t>(count_points(points)));
    evaluate(points, T, {});
    return T;
}

std::vector<double> NesModel::gradient(std::span<const double> points) const {
    const auto n = static_cast<std::size_t>(count_points(points));
    std::vector<double> T(n); // computed on the way
    std::vector<double> G(n * static_cast<std::size_t>(n_inputs()));
    evaluate(points, T, G);
    return G;
}

} // namespace nes_cpp

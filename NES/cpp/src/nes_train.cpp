// NES training: the parameters as one vector, the loss gradient by double backpropagation, Adam,
// and the training loop (docs/NES_TRAINING.md).

#include "nes_cpp/nes_model.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace nes_cpp {
namespace { // helpers used only in this file

// Points per chunk of a batch: chunks run in parallel and are added in a fixed order
// (docs/NES_TRAINING.md, "Parallelism and reproducibility").
constexpr Index chunk_size = 128;

// Calls visit(block) for every block of trainable values of `weights`, as a contiguous span, in
// the order of docs/NES_TRAINING.md ("Parameters"). `Weights` may be const or not.
template <typename Weights, typename Visit> void for_each_block(Weights& weights, Visit visit) {
    for (auto& layer : weights.hidden) {
        visit(std::span(layer.kernel.data(), static_cast<std::size_t>(layer.kernel.size())));
        visit(std::span(layer.bias.data(), static_cast<std::size_t>(layer.bias.size())));
    }
    visit(std::span(weights.head_kernel.data(),
                    static_cast<std::size_t>(weights.head_kernel.size())));
    visit(std::span(&weights.head_bias, 1));
    for (auto& layer : weights.hidden) {
        if (layer.activation.adaptive) {
            visit(std::span(&layer.activation.a, 1));
        }
    }
    if (weights.out_activation.adaptive) {
        visit(std::span(&weights.out_activation.a, 1));
    }
}

// One Adam step as in Keras 3, with NES's inverse-time decay (docs/NES_TRAINING.md, "Optimiser").
void adam_step(std::vector<double>& theta, std::span<const double> gradient, std::vector<double>& m,
               std::vector<double>& v, Index step, const TrainOptions& o) {
    const auto t = static_cast<double>(step);
    const double learning_rate = o.learning_rate / (1.0 + o.decay * t);
    const double alpha = learning_rate * std::sqrt(1.0 - std::pow(o.beta2, t + 1.0)) /
                         (1.0 - std::pow(o.beta1, t + 1.0));
    for (std::size_t i = 0; i < theta.size(); ++i) {
        m[i] += (gradient[i] - m[i]) * (1.0 - o.beta1); // = beta1 m + (1 - beta1) g, as Keras
        v[i] += (gradient[i] * gradient[i] - v[i]) * (1.0 - o.beta2);
        theta[i] -= m[i] * alpha / (std::sqrt(v[i]) + o.epsilon);
    }
}

} // namespace

std::vector<double> NesModel::parameters() const {
    std::vector<double> values;
    for_each_block(weights_, [&](std::span<const double> block) {
        values.insert(values.end(), block.begin(), block.end());
    });
    return values;
}

void NesModel::set_parameters(std::span<const double> values) {
    if (values.size() != parameters().size()) {
        throw std::invalid_argument("NesModel: set_parameters needs " +
                                    std::to_string(parameters().size()) + " values");
    }
    std::size_t next = 0;
    for_each_block(weights_, [&](std::span<double> block) {
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(next), block.size(),
                    block.begin());
        next += block.size();
    });
}

// Adds to `grad` the parameter gradient of one network pass (order 2), given the loss derivatives
// with respect to its logits (`logit_hat`) and to g_0 (`g_hat`) (docs/NES_TRAINING.md, steps 5
// and 6).
void NesModel::reverse_pass(const Pass& pass, const Eigen::ArrayXXd& logit_hat,
                            Eigen::ArrayXXd g_hat, Weights& grad) const {
    const auto L = weights_.hidden.size();
    std::vector<Eigen::ArrayXXd> z_back(L); // z_k's share of the loss through the backward pass
    for (std::size_t k = 0; k < L; ++k) {   // step 5: reverse of the backward pass
        const Layer& layer = weights_.hidden[k];
        const Activation& act = layer.activation;
        const Derivatives& d = pass.act[k];
        Layer& out = grad.hidden[k];
        const Eigen::ArrayXXd q_hat = (g_hat.matrix() * layer.kernel).array();
        out.kernel += g_hat.matrix().transpose() * pass.q[k].matrix();
        z_back[k] = q_hat * pass.g[k] * (act.c() * act.c()) * d.d2g;
        if (act.adaptive) { // d(phi')/da = n g' + c n z g''
            out.activation.a +=
                (q_hat * pass.g[k] * (act.n * d.dg + act.c() * act.n * pass.z[k] * d.d2g)).sum();
        }
        g_hat = q_hat * (act.c() * d.dg);
    }
    grad.head_kernel += g_hat.colwise().sum().matrix().transpose(); // g_L = w for every point

    // step 6: reverse of the forward pass, from the logits
    Eigen::ArrayXXd h_hat = (logit_hat.matrix() * weights_.head_kernel.transpose()).array();
    grad.head_kernel += pass.h.back().matrix().transpose() * logit_hat.matrix();
    grad.head_bias += logit_hat.sum();
    for (std::size_t k = L; k-- > 0;) {
        const Layer& layer = weights_.hidden[k];
        const Activation& act = layer.activation;
        const Derivatives& d = pass.act[k];
        Layer& out = grad.hidden[k];
        const Eigen::ArrayXXd z_hat = h_hat * (act.c() * d.dg) + z_back[k];
        if (act.adaptive) { // d(phi)/da = n z g'
            out.activation.a += (h_hat * act.n * pass.z[k] * d.dg).sum();
        }
        out.kernel += pass.h[k].matrix().transpose() * z_hat.matrix();
        out.bias += z_hat.colwise().sum().matrix();
        h_hat = (z_hat.matrix() * layer.kernel.transpose()).array();
    }
}

// Sum of `weight` * |e_p| over a chunk of points (rows), and the gradient of that sum added to
// `grad` (docs/NES_TRAINING.md, "Gradient of the loss", steps 1 to 4).
double NesModel::chunk_loss_gradient(const Eigen::MatrixXd& points, const Eigen::ArrayXd& velocity,
                                     double weight, Weights& grad) const {
    const Batch b = evaluate_batch(points, 2); // T, grad T and what their derivatives need

    // Step 1: the residual of the receiver equation and its derivative.
    const Eigen::ArrayXXd grad_r = b.grad_T.rightCols(dim_);
    const Eigen::ArrayXd residual =
        (grad_r.square().rowwise().sum() * velocity * velocity - 1.0) / 2.0;
    const Eigen::ArrayXd sign = residual.sign(); // 0 at 0, as JAX
    Eigen::ArrayXXd grad_T_hat = Eigen::ArrayXXd::Zero(points.rows(), n_inputs());
    grad_T_hat.rightCols(dim_) = grad_r.colwise() * (weight * sign * velocity * velocity);

    // Steps 2 and 3: to gamma, tau, dlogit/du, the logit and the output a.
    const Activation& out = weights_.out_activation;
    const double c = out.c();
    const Eigen::ArrayXd gamma_hat = (grad_T_hat * b.dlogit).rowwise().sum();
    const Eigen::ArrayXd tau_hat = (grad_T_hat * b.dr).rowwise().sum();
    const Eigen::ArrayXXd dlogit_hat = grad_T_hat.colwise() * b.gamma;
    const Eigen::ArrayXd dpsi = b.psi.dg.col(0);
    const Eigen::ArrayXd d2psi = b.psi.d2g.col(0);
    const Eigen::ArrayXd logit_hat =
        gamma_hat * b.r * b.scale * (c * c) * d2psi + tau_hat * b.scale * c * dpsi;
    if (out.adaptive) { // d(psi)/da = n l g', d(psi')/da = n g' + c n l g''
        const Eigen::ArrayXd l = b.logit.col(0);
        grad.out_activation.a +=
            (gamma_hat * b.r * b.scale * (out.n * dpsi + c * out.n * l * d2psi) +
             tau_hat * b.scale * out.n * l * dpsi)
                .sum();
    }

    // Step 4 and the reverse passes: g_0's derivative is s times that of dlogit/du; with
    // reciprocity 'output' each pass gets half, the swapped one with its halves exchanged.
    const double share = swap_mean_ ? 0.5 : 1.0;
    reverse_pass(b.pass, share * logit_hat, share * input_scale_ * dlogit_hat, grad);
    if (swap_mean_) {
        reverse_pass(b.swapped, share * logit_hat,
                     share * input_scale_ * exchange_halves(dlogit_hat), grad);
    }
    return weight * residual.abs().sum();
}

double NesModel::loss_and_gradient(std::span<const double> points, std::span<const double> velocity,
                                   std::span<double> gradient) const {
    const Index n = count_points(points);
    if (std::ssize(velocity) != n || gradient.size() != parameters().size()) {
        throw std::invalid_argument("NesModel: one velocity per point and one gradient value per "
                                    "parameter are needed");
    }
    // Chunks in parallel, each with its own gradient, added afterwards in chunk order.
    const auto n_in = static_cast<Index>(n_inputs());
    const Index n_chunks = (n + chunk_size - 1) / chunk_size;
    std::vector<double> losses(static_cast<std::size_t>(n_chunks));
    std::vector<std::vector<double>> gradients(static_cast<std::size_t>(n_chunks));
    detail::parallel_for(n_chunks, [&](Index k) {
        const Index first = k * chunk_size;
        const Index count = std::min(chunk_size, n - first);
        const Eigen::Map<const RowMajorMatrix> chunk(points.data() + first * n_in, count, n_in);
        const Eigen::Map<const Eigen::ArrayXd> v(velocity.data() + first, count);
        Weights grad = weights_; // same shapes and adaptive flags, values set to zero below
        for_each_block(grad,
                       [](std::span<double> block) { std::fill(block.begin(), block.end(), 0.0); });
        const auto i = static_cast<std::size_t>(k);
        losses[i] = chunk_loss_gradient(chunk, v, 1.0 / static_cast<double>(n), grad);
        for_each_block(grad, [&](std::span<const double> block) {
            gradients[i].insert(gradients[i].end(), block.begin(), block.end());
        });
    });
    std::fill(gradient.begin(), gradient.end(), 0.0);
    double loss = 0.0;
    for (std::size_t i = 0; i < losses.size(); ++i) {
        loss += losses[i];
        std::transform(gradient.begin(), gradient.end(), gradients[i].begin(), gradient.begin(),
                       std::plus<>());
    }
    return loss;
}

std::vector<double> NesModel::train(std::span<const double> points,
                                    std::span<const double> velocity, const TrainOptions& options) {
    const Index n = count_points(points);
    if (std::ssize(velocity) != n || n == 0) {
        throw std::invalid_argument("NesModel: train needs points and one velocity per point");
    }
    if (options.epochs < 0 || options.batch_size < 1) {
        throw std::invalid_argument("NesModel: train needs epochs >= 0 and batch_size >= 1");
    }
    const auto n_in = static_cast<std::size_t>(n_inputs());
    std::vector<double> theta = parameters();
    std::vector<double> m(theta.size(), 0.0);
    std::vector<double> v(theta.size(), 0.0);
    std::vector<double> gradient(theta.size());
    std::vector<Index> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), Index{0});
    std::mt19937_64 random(options.seed); // the standard fixes its sequence for a given seed

    std::vector<double> batch_points;
    std::vector<double> batch_velocity;
    std::vector<double> history;
    Index step = 0;
    for (int epoch = 0; epoch < options.epochs; ++epoch) {
        // Fisher-Yates shuffle, written out: std::shuffle may differ between standard libraries.
        for (Index i = n - 1; i > 0; --i) {
            const auto j = static_cast<Index>(random() % static_cast<std::uint64_t>(i + 1));
            std::swap(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(j)]);
        }
        double loss_sum = 0.0;
        for (Index first = 0; first < n; first += options.batch_size) {
            const Index count = std::min(options.batch_size, n - first);
            batch_points.clear();
            batch_velocity.clear();
            for (Index i = first; i < first + count; ++i) {
                const auto p = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                batch_points.insert(batch_points.end(),
                                    points.begin() + static_cast<std::ptrdiff_t>(p * n_in),
                                    points.begin() + static_cast<std::ptrdiff_t>((p + 1) * n_in));
                batch_velocity.push_back(velocity[p]);
            }
            loss_sum += loss_and_gradient(batch_points, batch_velocity, gradient) *
                        static_cast<double>(count);
            adam_step(theta, gradient, m, v, step, options);
            set_parameters(theta);
            ++step;
        }
        history.push_back(loss_sum / static_cast<double>(n));
    }
    return history;
}

} // namespace nes_cpp

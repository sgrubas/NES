// NES inference and training: traveltimes, their gradient and a minimal training loop for a Neural
// Eikonal Solver network (NES_OP or NES_TP) stored in a weight file. Implements
// docs/NES_INFERENCE.md and docs/NES_TRAINING.md; file format: docs/WEIGHTS_FORMAT.md.
#pragma once

#include "nes_cpp/index.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nes_cpp {

// Settings of NesModel::train (docs/NES_TRAINING.md, "Optimiser" and "Data").
struct TrainOptions {
    int epochs = 1;
    Index batch_size = 32;       // points per Adam step (Keras' default)
    double learning_rate = 3e-3; // NES's default for NES_OP; 5e-3 for NES_TP
    double decay = 5e-4;         // step t uses learning_rate / (1 + decay * t)
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-7;
    std::uint64_t seed = 0; // order of the points in each epoch
};

// A NES traveltime network. The evaluation methods are const and may run in several threads at
// once; train() and set_parameters() change the weights.
class NesModel {
  public:
    // Reads a weight file written by tools/export_nes_weights.py or save(). Throws
    // std::invalid_argument if the file cannot be read, is malformed, or describes a network this
    // version cannot evaluate.
    static NesModel load(const std::filesystem::path& path);

    // Writes the network to a weight file that load() and tools/import_nes_weights.py read.
    // Throws std::invalid_argument if the file cannot be written.
    void save(const std::filesystem::path& path) const;

    int dim() const { return dim_; }
    bool two_point() const { return two_point_; } // true for NES_TP

    // Coordinates per point: dim for NES_OP (receiver), 2 * dim for NES_TP (source, then receiver).
    int n_inputs() const { return two_point_ ? 2 * dim_ : dim_; }

    // Traveltimes at n points. `points` holds n * n_inputs() values, point after point (row-major),
    // coordinates in NES order (z last). Returns the n traveltimes. Batches of points run in
    // parallel (OpenMP); the result does not depend on the number of threads. Throws
    // std::invalid_argument if the number of values is not a multiple of n_inputs().
    std::vector<double> traveltime(std::span<const double> points) const;

    // Gradient of T at n points (docs/NES_INFERENCE.md, "Gradient"): n * n_inputs() values, point
    // after point, the derivatives with respect to the point's coordinates in input order:
    // dT/dxr for NES_OP, [dT/dxs, dT/dxr] for NES_TP. 0 at the source. Reverse mode, about twice
    // the cost of traveltime(). Same rules for `points` as traveltime().
    std::vector<double> gradient(std::span<const double> points) const;

    // The trainable parameters as one vector, in the order of docs/NES_TRAINING.md ("Parameters").
    std::vector<double> parameters() const;

    // Replaces the trainable parameters (same order and length as parameters()); throws
    // std::invalid_argument for a wrong length.
    void set_parameters(std::span<const double> values);

    // NES's training loss on the given points, taken as one batch (docs/NES_TRAINING.md, "Loss"),
    // and its gradient with respect to the parameters, written to `gradient` (same length as
    // parameters()). `velocity` holds the velocity at each point's receiver. Chunks of points run
    // in parallel; the result does not depend on the number of threads. Throws
    // std::invalid_argument for inconsistent sizes.
    double loss_and_gradient(std::span<const double> points, std::span<const double> velocity,
                             std::span<double> gradient) const;

    // Trains the parameters with Adam on the given points and receiver velocities
    // (docs/NES_TRAINING.md). Returns the mean loss of each epoch. Adam's moments start at zero in
    // every call. Throws std::invalid_argument for inconsistent sizes or options.
    std::vector<double> train(std::span<const double> points, std::span<const double> velocity,
                              const TrainOptions& options);

  private:
    // The functions g of an activation g(c * z) (docs/NES_INFERENCE.md, "Activations").
    enum class Function { gauss, tanh, sigmoid, sin, swish, linear, softplus, elu, relu };

    struct Activation {
        Function function = Function::linear;
        double n = 1.0;        // constant factor
        double a = 1.0;        // trainable factor if `adaptive`
        bool adaptive = false; // whether `a` is trained
        double c() const { return n * a; }
    };

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    struct Layer {
        RowMajorMatrix kernel;   // (n_in, n_out), row-major as in the weight file
        Eigen::RowVectorXd bias; // (n_out)
        Activation activation;
    };

    // The trainable part of the network; the same structure holds the gradient during training.
    struct Weights {
        std::vector<Layer> hidden;
        Eigen::VectorXd head_kernel; // (n_L)
        double head_bias = 0.0;
        Activation out_activation;
    };

    // An activation function and its derivatives at x, elementwise (docs/NES_TRAINING.md).
    struct Derivatives {
        Eigen::ArrayXXd g;   // g(x)
        Eigen::ArrayXXd dg;  // g'(x), if order >= 1
        Eigen::ArrayXXd d2g; // g''(x), if order >= 2
    };

    // One pass of the network over feature rows. Order 0: the logits; order 1: also the backward
    // pass for dlogit/dfeatures; order 2: also the layer values of the training gradient
    // (docs/NES_TRAINING.md, "Gradient of the loss").
    struct Pass {
        Eigen::ArrayXXd logit;
        Eigen::ArrayXXd dlogit;         // g_0 = dlogit/dfeatures (order >= 1)
        std::vector<Derivatives> act;   // g', g'' at c z_k (order >= 1)
        std::vector<Eigen::ArrayXXd> h; // h_0 (the features), ..., h_L (order 2)
        std::vector<Eigen::ArrayXXd> z; // z_k (order 2)
        std::vector<Eigen::ArrayXXd> g; // g_k = dlogit/dh_k (order 2)
        std::vector<Eigen::ArrayXXd> q; // q_k = dlogit/dz_k (order 2)
    };

    // A batch of points (one per row) through the whole model: T = r tau and, from order 1,
    // grad_T = gamma dlogit + tau dr (docs/NES_INFERENCE.md, "Network" and "Gradient").
    struct Batch {
        Eigen::ArrayXd r;       // |xr - xs| if factored, else 1
        Pass pass;              // over [xs, xr] (NES_TP) or xr - xs (NES_OP)
        Pass swapped;           // over [xr, xs], for reciprocity 'output'
        Eigen::ArrayXXd logit;  // after the reciprocity mean
        Derivatives psi;        // the output activation at c * logit
        double scale = 1.0;     // slow_max - slow_min with out_vscale, else 1
        Eigen::ArrayXd T;       // traveltimes
        Eigen::ArrayXXd dlogit; // dlogit/dinputs (order >= 1)
        Eigen::ArrayXd gamma;   // dT/dlogit (order >= 1)
        Eigen::ArrayXXd dr;     // dr/dinputs (order >= 1)
        Eigen::ArrayXXd grad_T; // dT/dinputs (order >= 1)
    };

    NesModel() = default; // instances come from load()

    static Function function_named(const std::string& name);
    static void evaluate(Function function, const Eigen::ArrayXXd& x, int order, Derivatives& d);
    Pass network_pass(Eigen::ArrayXXd h, int order) const;
    Batch evaluate_batch(const Eigen::MatrixXd& points, int order) const;
    void evaluate(std::span<const double> points, std::span<double> T, std::span<double> G) const;
    Index count_points(std::span<const double> points) const;

    // [a_s, a_r] -> [a_r, a_s]: the source and receiver halves of NES_TP rows exchanged.
    Eigen::ArrayXXd exchange_halves(const Eigen::ArrayXXd& a) const {
        Eigen::ArrayXXd b(a.rows(), a.cols());
        b << a.rightCols(dim_), a.leftCols(dim_);
        return b;
    }

    void reverse_pass(const Pass& pass, const Eigen::ArrayXXd& logit_hat, Eigen::ArrayXXd g_hat,
                      Weights& grad) const;
    double chunk_loss_gradient(const Eigen::MatrixXd& points, const Eigen::ArrayXd& velocity,
                               double weight, Weights& grad) const;

    int dim_ = 0;
    bool two_point_ = false;
    bool swap_mean_ = false; // NES_TP with reciprocity 'output'
    Eigen::RowVectorXd xs_;  // source position (NES_OP)
    double input_scale_ = 1.0;
    bool factored_ = true;
    bool out_vscale_ = true;
    double slow_min_ = 0.0; // 1 / v_max
    double slow_max_ = 0.0; // 1 / v_min
    Weights weights_;
};

} // namespace nes_cpp

// Python module `nes_cpp`: pybind11 bindings for the C++ eikonal kernels.
//
// What crosses the Python <-> C++ boundary:
//   * Input arrays are taken as py::array_t<double, py::array::c_style>. A C-contiguous float64
//     array is passed to C++ without a copy. Any other array that NumPy can cast safely to
//     float64 (float32, integers, Fortran order, strided views) is first converted to a
//     C-contiguous float64 copy by pybind11. Arrays that cannot be cast safely (complex, strings)
//     raise TypeError. Source indices are int64 arrays, converted the same way.
//   * Results are new float64 NumPy arrays: the C++ std::vector is copied once into the array.
//   * The GIL is released while a kernel runs, so other Python threads can run at the same time.
//     Do not modify the input arrays from another thread during a call.
//   * std::invalid_argument from C++ becomes ValueError in Python.

#include "nes_cpp/fmm.hpp"
#include "nes_cpp/nes_model.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>            // std::vector <-> Python sequences
#include <pybind11/stl/filesystem.h> // std::filesystem::path <-> str or pathlib.Path

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using nes_cpp::Index;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style>;
using IndexArray = py::array_t<Index, py::array::c_style>;

// The grid of `v`: its shape, and spacing h per axis.
nes_cpp::Grid grid_of(const DoubleArray& v, const std::vector<double>& h) {
    nes_cpp::Grid grid;
    grid.shape.assign(v.shape(), v.shape() + v.ndim());
    grid.spacing = h;
    return grid;
}

// A view of the values of a C-contiguous array (no copy).
template <typename T> std::span<const T> values_of(const py::array_t<T, py::array::c_style>& a) {
    return {a.data(), static_cast<std::size_t>(a.size())};
}

// A new NumPy array holding a copy of `values` (pybind11 copies a pointer given without an owner),
// of the given shape or else 1-D.
py::array_t<double> to_numpy(const std::vector<double>& values,
                             std::vector<py::ssize_t> shape = {}) {
    if (shape.empty()) {
        shape = {static_cast<py::ssize_t>(values.size())};
    }
    return py::array_t<double>(shape, values.data());
}

// Returns f() computed with the GIL released, so that other Python threads run meanwhile. The GIL
// is taken back when `release` goes out of scope, also if f throws. f must not touch Python
// objects.
template <typename F> auto without_gil(F f) {
    py::gil_scoped_release release;
    return f();
}

py::array_t<double> fmm(const DoubleArray& v, const std::vector<Index>& source,
                        const std::vector<double>& h, int order) {
    const nes_cpp::Grid grid = grid_of(v, h);
    const std::span<const double> velocity = values_of(v);
    const auto T = without_gil([&] { return nes_cpp::fmm(velocity, grid, source, order); });
    return to_numpy(T, std::vector<py::ssize_t>(v.shape(), v.shape() + v.ndim()));
}

py::array_t<double> fmm_multi(const DoubleArray& v, const IndexArray& sources,
                              const std::vector<double>& h, int order) {
    if (sources.ndim() != 2 || sources.shape(1) != v.ndim()) {
        throw std::invalid_argument("fmm_multi: sources must have shape (n_sources, v.ndim)");
    }
    const nes_cpp::Grid grid = grid_of(v, h);
    const std::span<const double> velocity = values_of(v);
    const std::span<const Index> source_indices = values_of(sources);
    const auto table =
        without_gil([&] { return nes_cpp::fmm_multi(velocity, grid, source_indices, order); });
    std::vector<py::ssize_t> shape{sources.shape(0)};
    shape.insert(shape.end(), v.shape(), v.shape() + v.ndim());
    return to_numpy(table, shape);
}

// Checks that x holds points of `model`: shape (..., n_inputs).
void check_points(const nes_cpp::NesModel& model, const DoubleArray& x, const std::string& method) {
    if (x.ndim() < 1 || x.shape(x.ndim() - 1) != model.n_inputs()) {
        throw std::invalid_argument("NesModel." + method + ": x must have shape (..., " +
                                    std::to_string(model.n_inputs()) + ")");
    }
}

py::array_t<double> traveltime(const nes_cpp::NesModel& model, const DoubleArray& x) {
    check_points(model, x, "traveltime");
    const std::span<const double> points = values_of(x);
    const auto T = without_gil([&] { return model.traveltime(points); });
    return to_numpy(T, std::vector<py::ssize_t>(x.shape(), x.shape() + x.ndim() - 1));
}

// Columns [first, first + count) of the gradient at every point: shape x.shape[:-1] + (count,).
py::array_t<double> gradient_columns(const nes_cpp::NesModel& model, const DoubleArray& x,
                                     Index first, Index count, const std::string& method) {
    check_points(model, x, method);
    const std::span<const double> points = values_of(x);
    const auto G = without_gil([&] { return model.gradient(points); });
    const Index n_in = model.n_inputs();
    std::vector<py::ssize_t> shape(x.shape(), x.shape() + x.ndim() - 1);
    shape.push_back(count);
    py::array_t<double> result(shape);
    double* out = result.mutable_data();
    const auto n_points = static_cast<Index>(G.size()) / n_in;
    for (Index p = 0; p < n_points; ++p) {
        for (Index j = 0; j < count; ++j) {
            out[p * count + j] = G[static_cast<std::size_t>(p * n_in + first + j)];
        }
    }
    return result;
}

py::array_t<double> gradient(const nes_cpp::NesModel& model, const DoubleArray& x) {
    return gradient_columns(model, x, 0, model.n_inputs(), "gradient");
}

// dT/dxs (first) or dT/dxr (second half of the inputs) of a NES_TP model.
py::array_t<double> gradient_half(const nes_cpp::NesModel& model, const DoubleArray& x,
                                  bool receiver) {
    const std::string method = receiver ? "gradient_r" : "gradient_s";
    if (!model.two_point()) {
        throw std::invalid_argument("NesModel." + method +
                                    ": only for NES_TP models; use gradient for NES_OP");
    }
    return gradient_columns(model, x, receiver ? model.dim() : 0, model.dim(), method);
}

py::array_t<double> parameters(const nes_cpp::NesModel& model) {
    return to_numpy(model.parameters());
}

void set_parameters(nes_cpp::NesModel& model, const DoubleArray& values) {
    if (values.ndim() != 1) {
        throw std::invalid_argument("NesModel.set_parameters: values must be a 1-D array");
    }
    model.set_parameters(values_of(values));
}

py::tuple loss_and_gradient(const nes_cpp::NesModel& model, const DoubleArray& x,
                            const DoubleArray& v) {
    check_points(model, x, "loss_and_gradient");
    const std::span<const double> points = values_of(x);
    const std::span<const double> velocity = values_of(v);
    std::vector<double> gradient(model.parameters().size());
    const double loss =
        without_gil([&] { return model.loss_and_gradient(points, velocity, gradient); });
    return py::make_tuple(loss, to_numpy(gradient));
}

py::array_t<double> train(nes_cpp::NesModel& model, const DoubleArray& x, const DoubleArray& v,
                          int epochs, Index batch_size, std::optional<double> learning_rate,
                          double decay, double beta_1, double beta_2, double epsilon,
                          std::uint64_t seed) {
    check_points(model, x, "train");
    nes_cpp::TrainOptions options;
    options.epochs = epochs;
    options.batch_size = batch_size;
    // NES's defaults: 3e-3 for NES_OP, 5e-3 for NES_TP
    options.learning_rate = learning_rate.value_or(model.two_point() ? 5e-3 : 3e-3);
    options.decay = decay;
    options.beta1 = beta_1;
    options.beta2 = beta_2;
    options.epsilon = epsilon;
    options.seed = seed;
    const std::span<const double> points = values_of(x);
    const std::span<const double> velocity = values_of(v);
    // do not use this model from other threads meanwhile
    return to_numpy(without_gil([&] { return model.train(points, velocity, options); }));
}

} // namespace

// PYBIND11_MODULE defines the function that Python calls on `import nes_cpp`.
// Its first argument must match the name of the built file (nes_cpp.*.pyd / nes_cpp.*.so).
PYBIND11_MODULE(nes_cpp, m) {
    m.doc() = "C++ eikonal kernels for NES: factored fast marching, NES inference and training.";
    m.attr("__version__") = NES_CPP_VERSION; // defined by CMake from pyproject.toml

    m.def("fmm", &fmm, py::arg("v"), py::arg("source"), py::arg("h"), py::arg("order") = 2,
          R"(Traveltimes from a point source by the factored fast marching method.

Parameters
----------
v : array_like, shape (n_x, n_z) or (n_x, n_y, n_z)
    Velocity at every grid node, z last (for example in km/s). Node 0 is at the origin.
source : sequence of int
    Grid index of the source, one per axis.
h : sequence of float
    Grid spacing per axis (for example in km).
order : {1, 2}
    Finite-difference order.

Returns
-------
T : ndarray of float64, same shape as v
    Traveltimes (for example in s).
)");

    m.def("fmm_multi", &fmm_multi, py::arg("v"), py::arg("sources"), py::arg("h"),
          py::arg("order") = 2,
          R"(Traveltime tables for several point sources, solved in parallel (OpenMP).

Each source is solved by one thread, so the result equals calling fmm for each source, bit for
bit, for any number of threads (set it with the environment variable OMP_NUM_THREADS).

Parameters
----------
v, h, order
    As for fmm.
sources : array_like of int, shape (n_sources, v.ndim)
    Grid index of each source.

Returns
-------
T : ndarray of float64, shape (n_sources,) + v.shape
)");

    py::class_<nes_cpp::NesModel>(
        m, "NesModel",
        R"(A NES traveltime network (NES_OP or NES_TP): evaluation and training.

Load a weight file written by NES/cpp/tools/export_nes_weights.py:
``model = nes_cpp.NesModel("model.nesw")``.
)")
        .def(py::init(&nes_cpp::NesModel::load), py::arg("path"))
        .def_property_readonly("dim", &nes_cpp::NesModel::dim, "Number of coordinates, 2 or 3.")
        .def_property_readonly("two_point", &nes_cpp::NesModel::two_point,
                               "True for NES_TP (points are source-receiver pairs).")
        .def("save", &nes_cpp::NesModel::save, py::arg("path"),
             "Writes the network to a .nesw weight file (read by NesModel and "
             "tools/import_nes_weights.py).")
        .def("parameters", &parameters,
             "The trainable parameters as one float64 array (order: docs/NES_TRAINING.md).")
        .def("set_parameters", &set_parameters, py::arg("values"),
             "Replaces the trainable parameters (same order and length as parameters()).")
        .def("loss_and_gradient", &loss_and_gradient, py::arg("x"), py::arg("v"),
             R"(NES's training loss on the points x as one batch, and its gradient.

v holds the velocity at each point's receiver. Returns (loss, gradient) with the gradient in the
order of parameters() (docs/NES_TRAINING.md).
)")
        .def(
            "train", &train, py::arg("x"), py::arg("v"), py::arg("epochs") = 1,
            py::arg("batch_size") = 32, py::arg("learning_rate") = py::none(),
            py::arg("decay") = 5e-4, py::arg("beta_1") = 0.9, py::arg("beta_2") = 0.999,
            py::arg("epsilon") = 1e-7, py::arg("seed") = 0,
            R"(Trains the parameters with Adam (docs/NES_TRAINING.md) and returns the mean loss of each epoch.

Parameters
----------
x : array_like, shape (n, n_inputs)
    Training points, as for traveltime.
v : array_like, shape (n,)
    Velocity at each point's receiver.
epochs, batch_size : int
learning_rate : float, optional
    Default: NES's, 3e-3 for NES_OP and 5e-3 for NES_TP; step t uses learning_rate / (1 + decay t).
decay, beta_1, beta_2, epsilon : float
seed : int
    Order of the points in each epoch.
)")
        .def_property_readonly("n_inputs", &nes_cpp::NesModel::n_inputs,
                               "Coordinates per point: dim (NES_OP) or 2 * dim (NES_TP).")
        .def("traveltime", &traveltime, py::arg("x"),
             R"(Traveltimes at the points x.

Parameters
----------
x : array_like, shape (..., n_inputs)
    Receivers (NES_OP) or source-receiver pairs [xs, xr] (NES_TP), coordinates z last.

Returns
-------
T : ndarray of float64, shape x.shape[:-1]
)")
        .def("gradient", &gradient, py::arg("x"),
             R"(Gradient of T with respect to the coordinates of the points x (reverse mode).

Returns dT/dxr for NES_OP and [dT/dxs, dT/dxr] for NES_TP: an ndarray of float64 with shape
x.shape[:-1] + (n_inputs,). Zero at the source. Costs about two traveltime calls.
)")
        .def(
            "gradient_r",
            [](const nes_cpp::NesModel& model, const DoubleArray& x) {
                return gradient_half(model, x, true);
            },
            py::arg("x"), "dT/dxr of a NES_TP model, shape x.shape[:-1] + (dim,).")
        .def(
            "gradient_s",
            [](const nes_cpp::NesModel& model, const DoubleArray& x) {
                return gradient_half(model, x, false);
            },
            py::arg("x"), "dT/dxs of a NES_TP model, shape x.shape[:-1] + (dim,).");
}

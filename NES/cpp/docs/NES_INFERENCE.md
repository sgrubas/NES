# NES inference

`nes_cpp::NesModel` (in `src/nes_model.cpp`) evaluates a trained Neural Eikonal Solver: the traveltime network `TraveltimeNet` of `NES/layers.py`, with weights exported from Python by `tools/export_nes_weights.py` (file format: [`WEIGHTS_FORMAT.md`](WEIGHTS_FORMAT.md)). Training is described in [`NES_TRAINING.md`](NES_TRAINING.md). This page gives the equations exactly as `TraveltimeNet.call` computes them, so that C++ and Keras agree to round-off.

## Inputs and output

- **NES-OP** (one point, `NES_OP`): the source $\mathbf{x}_s$ is fixed and stored in the weight file. A point is a receiver $\mathbf{x}_r$ with `dim` coordinates.
- **NES-TP** (two points, `NES_TP`): a point is a pair $(\mathbf{x}_s, \mathbf{x}_r)$, given as one row of `2 dim` coordinates, source first.

Coordinates are in NES order, $(x, z)$ or $(x, y, z)$, z last. The output is the traveltime $T$ for every point, in the units of the training (NES typically uses km and km/s, so seconds).

## Network

Let $\mathbf{d} = \mathbf{x}_r - \mathbf{x}_s$ and let $s$ be the input scale: $s = 1/x_{\mathrm{scale}}$ with $x_{\mathrm{scale}}$ the largest absolute coordinate of the velocity model's box, or $s = 1$ if the model was built with `input_scale=False`.

**Features.** The first layer sees

$$\mathbf{f} = s\,\mathbf{d} \quad \text{(NES-OP)}, \qquad \mathbf{f} = s\,[\mathbf{x}_s, \mathbf{x}_r] \quad \text{(NES-TP)}.$$

**Hidden layers.** With $\mathbf{h}_0 = \mathbf{f}$ as a row vector, layers $k = 1, \dots, L$ compute

$$\mathbf{h}_k = \varphi_k(\mathbf{h}_{k-1} W_k + \mathbf{b}_k),$$

where $W_k$ is the kernel of shape $(n_{k-1}, n_k)$ and $\mathbf{b}_k$ the bias (Keras `Dense`: `matmul(h, kernel) + bias`). The widths $n_k$ may differ between layers.

**Activations.** NES writes an activation as `'(ad)-name-n'` (for example `'ad-gauss-1'`) or just `'name'` (then $n = 1$), meaning

$$\varphi(z) = g(c\,z), \qquad c = n\,a,$$

applied elementwise, where $g$ is the function `name`, $n$ a constant and $a$ a trainable scalar of that layer ($a = 1$ without `ad`). Each hidden layer has its own $a$; all hidden layers share $g$ and $n$. The code computes $c = n a$ first and then $c z$, in the same order as Keras. Supported functions:

| name | $g(x)$ | computed as |
|---|---|---|
| `gauss` | $e^{-x^2}$ | |
| `tanh` | $\tanh x$ | |
| `sigmoid` | $1/(1 + e^{-x})$ | |
| `sin` | $\sin x$ | |
| `swish` | $x / (1 + e^{-x})$ | $x \cdot \mathrm{sigmoid}(x)$ |
| `linear` | $x$ | |
| `softplus` | $\ln(1 + e^{x})$ | $\max(x, 0) + \ln(1 + e^{-\lvert x\rvert})$, which cannot overflow |
| `elu` | $x$ for $x > 0$, else $e^{x} - 1$ | `expm1` for $x \le 0$, accurate near 0 |
| `relu` | $\max(x, 0)$ | |

The stable forms are the ones JAX uses (`jax.nn.softplus`, `jax.nn.elu`). Models with other activations (or custom callables) are refused by the exporter.

**Head.** A linear layer with one output gives the logit

$$\ell = \mathbf{h}_L \mathbf{w} + c_0.$$

**Reciprocity (NES-TP).** With `reciprocity='output'` (the NES default) the logit is averaged over swapping source and receiver, before the output activation:

$$\ell = \tfrac{1}{2}\left(\ell(\mathbf{x}_s, \mathbf{x}_r) + \ell(\mathbf{x}_r, \mathbf{x}_s)\right),$$

so $T(\mathbf{x}_s, \mathbf{x}_r) = T(\mathbf{x}_r, \mathbf{x}_s)$ exactly. With `reciprocity=None` only the first term is used. The modes `'first_layer'` and `'invariant'` are not supported.

**Output.** The output activation $\psi$ (format as above, default `'ad-sigmoid-1'`) gives

$$\tau = \psi(\ell).$$

With `out_vscale=True` (default), $\tau$ is mapped into the slowness range of the velocity model:

$$\tau \leftarrow \sigma_{\min} + (\sigma_{\max} - \sigma_{\min})\,\tau, \qquad \sigma_{\min} = 1/v_{\max}, \quad \sigma_{\max} = 1/v_{\min}.$$

With `factored=True` (default), the traveltime is

$$T = \tau\,|\mathbf{d}|, \qquad |\mathbf{d}| = \sqrt{\textstyle\sum_k d_k^2},$$

which is 0 at the source (`_distance` in `NES/layers.py`). With `factored=False`, $T = \tau$.

## Gradient

`NesModel::gradient` returns $\nabla_{\mathbf{x}_r} T$ for NES-OP and $[\nabla_{\mathbf{x}_s} T, \nabla_{\mathbf{x}_r} T]$ for NES-TP: the derivative of $T$ with respect to every input coordinate $u_j$ of a point, $j = 1, \dots, m$ with $m$ = `n_inputs` (`dim` or `2 dim`). It is computed in reverse mode (backpropagation), as Keras does.

**Backward pass through the network.** The forward pass keeps each layer's activation slope $\varphi_k'(\mathbf{z}_k)$. Then, for each point, the derivative of the logit with respect to each layer's values runs from the head back to the features:

$$\mathbf{g}_L = \mathbf{w}^\top, \qquad \mathbf{q}_k = \mathbf{g}_k \odot \varphi_k'(\mathbf{z}_k), \qquad \mathbf{g}_{k-1} = \mathbf{q}_k W_k^\top \quad (k = L, \dots, 1),$$

so that $\mathbf{g}_k = \partial\ell/\partial\mathbf{h}_k$, $\mathbf{q}_k = \partial\ell/\partial\mathbf{z}_k$, and $\partial\ell/\partial\mathbf{f} = \mathbf{g}_0$. With all points of a batch as rows, each step is one matrix product with $W_k^\top$. The slope is $\varphi'(z) = c\,g'(c z)$, with $g'$ from this table (computed from $x$ and the value $g$ where that is cheaper):

| name | $g'(x)$ |
|---|---|
| `gauss` | $-2x\,g$ |
| `tanh` | $1 - g^2$ |
| `sigmoid` | $g(1 - g)$ |
| `sin` | $\cos x$ |
| `swish` | $\sigma + x\,\sigma(1 - \sigma)$, with $\sigma = \mathrm{sigmoid}(x)$ |
| `linear` | $1$ |
| `softplus` | $\mathrm{sigmoid}(x)$ |
| `elu` | $1$ for $x > 0$, else $e^{x}$ |
| `relu` | $1$ for $x > 0$, else $0$ (as JAX) |

**From the features to the inputs.** The features are $s$ times the inputs, so $\partial\ell/\partial\mathbf{u} = s\,\mathbf{g}_0$ (for NES-OP, $\mathbf{f} = s(\mathbf{x}_r - \mathbf{x}_s)$ and $\mathbf{u} = \mathbf{x}_r$ give the same). The swapped NES-TP pass sees $s\,[\mathbf{x}_r, \mathbf{x}_s]$, so its $\mathbf{g}_0$ is used with its two halves exchanged, and the swap mean takes half of each pass.

**Output stage.** With $\gamma = \partial T/\partial\ell = \lvert\mathbf{d}\rvert\,(\sigma_{\max} - \sigma_{\min})\,\psi'(\ell)$ (drop the factor $\lvert\mathbf{d}\rvert$ without `factored` and $\sigma_{\max} - \sigma_{\min}$ without `out_vscale`),

$$\nabla_{\mathbf{u}} T = \gamma\,\frac{\partial \ell}{\partial \mathbf{u}} + \tau\,\frac{\partial\lvert\mathbf{d}\rvert}{\partial \mathbf{u}},$$

where the second term is present only with `factored`, $\partial\lvert\mathbf{d}\rvert/\partial x_{r,k} = d_k/\lvert\mathbf{d}\rvert$ and $\partial\lvert\mathbf{d}\rvert/\partial x_{s,k} = -d_k/\lvert\mathbf{d}\rvert$, and both are 0 at the source, as in NES (`_distance` has zero gradient there). So $\nabla T = 0$ at the source.

**Cost.** One forward pass that keeps the slopes, plus one backward pass of the same size: about 2 network passes for every point, whatever the number of inputs $m$. Measured: a gradient call costs 1.7 to 1.9 traveltime calls (README, "Benchmarks"). Forward mode, which carries the $m$ derivatives along the forward pass, costs $1 + m$ passes (5 for NES-TP); a forward-mode version measured 3.7 to 5.3 traveltime calls for the NES-TP gradient, and its T + ∇T was 1.8 to 2.3 times slower. Reverse mode is also the first half of the training gradient ([`NES_TRAINING.md`](NES_TRAINING.md)).

## Batched evaluation

Points are processed in batches of 128 rows; a gradient batch also keeps one slope matrix per layer, and 128 rows keep these in a core's L2 cache. A batch is a matrix with one point per row, so each layer is one matrix product, `H * W` with Eigen, plus the bias and the activation applied to the whole matrix. For `reciprocity='output'` the swapped features form a second matrix that goes through the same layers.

Batches are independent and run in parallel with OpenMP. The split into batches does not depend on the number of threads, and Eigen's own multithreading is switched off (`EIGEN_DONT_PARALLELIZE`), so each point is computed by the same operations whatever the thread count: results are reproducible bit for bit.

## Precision

Everything is float64. Against Keras evaluating the same model in float64 (JAX with 64-bit arrays enabled), the remaining differences come from the order of the sums in the matrix products and from the libm implementations of `exp`, `tanh` and `sin`: about $10^{-15}$ per operation. A model trained in float32 is evaluated in float64 from its float32 weights (converted exactly); against Keras in float32, the differences are then float32 rounding, about $10^{-7}$ relative. The measured values are in the README.

# NES training

`NesModel::train` (in `src/nes_train.cpp`) trains the network of a `NesModel` with a short, fixed training loop: NES's default loss, its gradient computed by hand, and Adam. NES (Python) still builds the network, draws the initial weights and samples the training points. C++ only runs the optimisation, and NES can read the trained weights back (`tools/import_nes_weights.py`). The network itself is defined in [`NES_INFERENCE.md`](NES_INFERENCE.md).

## What is trained

**Loss.** NES's default for `NES_OP` and `NES_TP`: the mean absolute error of the eikonal residual of the receiver equation (`IsoEikonal` with $p = 2$ in Hamiltonian form; loss key `E` or `Er`). For a batch $B$ of points with receiver velocities $v_p$,

$$e_p = \tfrac12\left(\lvert\nabla_{\mathbf{x}_r} T_p\rvert^2 v_p^2 - 1\right), \qquad L = \frac{1}{\lvert B\rvert}\sum_{p \in B} \lvert e_p\rvert.$$

The source equation of NES-TP (`Es`), other losses, sample weights and early stopping are not supported.

**Parameters.** All kernels and biases, and the $a$ of the activations that are adaptive (`act_adaptive`, `out_act_adaptive` in the weight file). They are handled as one flat vector in this order: for each hidden layer its kernel (row by row) and bias; the head kernel and bias; the hidden $a$ (one per layer, if adaptive); the output $a$ (if adaptive).

**Data.** The caller passes the points (as for `traveltime`) and the velocity at each receiver; NES samples them uniformly in the velocity box (`NES.utils.Uniform_PDF`) and evaluates the velocity model. Each epoch visits the points in a new random order, in batches of `batch_size` points (the last one may be smaller), like Keras `fit` with `shuffle=True`.

**Optimiser.** Adam as in Keras 3, with the inverse-time decay of NES's `compile`. At step $t = 0, 1, \dots$ with loss gradient $\mathbf{g}$:

$$\mathbf{m} \leftarrow \beta_1\mathbf{m} + (1 - \beta_1)\,\mathbf{g}, \qquad \mathbf{v} \leftarrow \beta_2\mathbf{v} + (1 - \beta_2)\,\mathbf{g}^2,$$

$$\eta_t = \frac{\eta}{1 + \delta\,t}, \qquad \boldsymbol\theta \leftarrow \boldsymbol\theta - \eta_t\,\frac{\sqrt{1 - \beta_2^{t+1}}}{1 - \beta_1^{t+1}}\;\frac{\mathbf{m}}{\sqrt{\mathbf{v}} + \epsilon},$$

with NES's defaults $\eta$ = 3e-3 (NES-OP) or 5e-3 (NES-TP), $\delta$ = 5e-4, and Keras's $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon$ = 1e-7. $\mathbf{m}$ and $\mathbf{v}$ start at zero in every `train` call.

## Gradient of the loss

The loss contains $\nabla T$, which is itself computed by a backward pass ([`NES_INFERENCE.md`](NES_INFERENCE.md), "Gradient"). Its derivative with respect to the parameters is therefore a backward pass through that backward pass, followed by the usual backward pass through the forward pass ("double backpropagation", the reverse of reverse mode). Notation per point, as in `NES_INFERENCE.md`: forward values $\mathbf{z}_k, \mathbf{h}_k$, logit $\ell$, output $\tau$, distance $r = \lvert\mathbf{d}\rvert$; backward values $\mathbf{g}_k = \partial\ell/\partial\mathbf{h}_k$ and $\mathbf{q}_k = \partial\ell/\partial\mathbf{z}_k$; $\gamma = \partial T/\partial\ell$. A hat marks the derivative of the loss: $\hat x = \partial L/\partial x$.

**1. Loss to gradient.** Only the receiver part of $\nabla T$ enters the loss:

$$\widehat{\nabla_{\mathbf{x}_r} T} = \frac{\operatorname{sign}(e_p)\,v_p^2}{\lvert B\rvert}\,\nabla_{\mathbf{x}_r} T, \qquad \widehat{\nabla_{\mathbf{x}_s} T} = 0,$$

with $\operatorname{sign}(0) = 0$, as in JAX.

**2. Through $\nabla_{\mathbf{u}} T = \gamma\,\partial\ell/\partial\mathbf{u} + \tau\,\partial r/\partial\mathbf{u}$:**

$$\hat\gamma = \widehat{\nabla T} \cdot \frac{\partial\ell}{\partial\mathbf{u}}, \qquad \hat\tau = \widehat{\nabla T} \cdot \frac{\partial r}{\partial\mathbf{u}}, \qquad \widehat{\frac{\partial\ell}{\partial\mathbf{u}}} = \gamma\,\widehat{\nabla T}.$$

**3. To the logit and the output $a$.** $\tau = \sigma_{\min} + \Delta\sigma\,\psi(\ell)$ and $\gamma = r\,\Delta\sigma\,\psi'(\ell)$ (without `out_vscale`: $\Delta\sigma = 1$, $\sigma_{\min} = 0$; without `factored`: $r = 1$ in $\gamma$, $\hat\tau$ unused), so

$$\hat\ell = \hat\gamma\,r\,\Delta\sigma\,\psi''(\ell) + \hat\tau\,\Delta\sigma\,\psi'(\ell), \qquad \hat a_{\mathrm{out}} = \hat\gamma\,r\,\Delta\sigma\,\frac{\partial\psi'}{\partial a} + \hat\tau\,\Delta\sigma\,\frac{\partial\psi}{\partial a}.$$

**4. Swap mean (NES-TP, reciprocity `'output'`).** Each of the two network passes receives $\tfrac12\hat\ell$ and $\tfrac12\,\widehat{\partial\ell/\partial\mathbf{u}}$, the swapped pass with the source and receiver halves exchanged.

**5. Reverse of the backward pass**, per network pass, starting from $\hat{\mathbf{g}}_0 = s\,\widehat{\partial\ell/\partial\mathbf{u}}$ (in feature order) and going $k = 1, \dots, L$:

$$\hat{\mathbf{q}}_k = \hat{\mathbf{g}}_{k-1} W_k, \qquad \hat W_k \mathrel{+}= \hat{\mathbf{g}}_{k-1}^\top \mathbf{q}_k,$$

$$\hat{\mathbf{g}}_k = \hat{\mathbf{q}}_k \odot \varphi_k'(\mathbf{z}_k), \qquad \hat{\mathbf{z}}_k^{\,\mathrm{back}} = \hat{\mathbf{q}}_k \odot \mathbf{g}_k \odot \varphi_k''(\mathbf{z}_k), \qquad \hat a_k \mathrel{+}= \textstyle\sum \hat{\mathbf{q}}_k \odot \mathbf{g}_k \odot \frac{\partial\varphi_k'}{\partial a}(\mathbf{z}_k),$$

and at the top, since $\mathbf{g}_L = \mathbf{w}^\top$: $\hat{\mathbf{w}} \mathrel{+}= \hat{\mathbf{g}}_L^\top$.

**6. Reverse of the forward pass**, from the logit, going $k = L, \dots, 1$:

$$\hat{\mathbf{h}}_L = \hat\ell\,\mathbf{w}^\top, \qquad \hat{\mathbf{w}} \mathrel{+}= \hat\ell\,\mathbf{h}_L^\top, \qquad \hat c_0 \mathrel{+}= \hat\ell,$$

$$\hat{\mathbf{z}}_k = \hat{\mathbf{h}}_k \odot \varphi_k'(\mathbf{z}_k) + \hat{\mathbf{z}}_k^{\,\mathrm{back}}, \qquad \hat a_k \mathrel{+}= \textstyle\sum \hat{\mathbf{h}}_k \odot \frac{\partial\varphi_k}{\partial a}(\mathbf{z}_k), \qquad \hat W_k \mathrel{+}= \mathbf{h}_{k-1}^\top \hat{\mathbf{z}}_k, \qquad \hat{\mathbf{b}}_k \mathrel{+}= \textstyle\sum \hat{\mathbf{z}}_k, \qquad \hat{\mathbf{h}}_{k-1} = \hat{\mathbf{z}}_k W_k^\top.$$

With all points of a batch as rows, the sums run over the points and each product with a whole $W_k$, $\hat W_k$ or $W_k^\top$ is one matrix product.

**Activation derivatives.** With $\varphi(z) = g(cz)$ and $c = n\,a$:

$$\varphi' = c\,g'(cz), \qquad \varphi'' = c^2 g''(cz), \qquad \frac{\partial\varphi}{\partial a} = n\,z\,g'(cz), \qquad \frac{\partial\varphi'}{\partial a} = n\,g'(cz) + c\,n\,z\,g''(cz).$$

The second derivatives $g''(x)$:

| name | $g''(x)$ |
|---|---|
| `gauss` | $(4x^2 - 2)\,g$ |
| `tanh` | $-2g\,(1 - g^2)$ |
| `sigmoid` | $g(1 - g)(1 - 2g)$ |
| `sin` | $-\sin x$ |
| `swish` | $\sigma(1 - \sigma)\,\bigl(2 + x(1 - 2\sigma)\bigr)$, with $\sigma = \mathrm{sigmoid}(x)$ |
| `linear` | $0$ |
| `softplus` | $\sigma(1 - \sigma)$ |
| `elu` | $0$ for $x > 0$, else $e^{x}$ |
| `relu` | $0$ |

**Cost.** Per point: the forward pass (keeping $\mathbf{z}_k$, $\mathbf{h}_k$ and the activation values), the backward pass for $\nabla T$, and the reverse of both: about 4 to 5 network passes, whatever the number of inputs.

## Parallelism and reproducibility

A batch is split into chunks of 128 points. The chunks run in parallel (OpenMP), each computing the gradient of its points, and the chunk gradients are then added in chunk order. The result does not depend on the number of threads. The order of the points comes from a 64-bit Mersenne Twister (`std::mt19937_64`, whose sequence the C++ standard fixes) and a Fisher-Yates shuffle written out in the code, because `std::shuffle` and `std::uniform_int_distribution` may differ between standard libraries. A run with a given seed therefore gives the same weights with any number of threads, and on GCC and MSVC up to the rounding of the libm functions.

## Comparison with Keras

From the same start (the exported network), a full batch makes the order of the points irrelevant, and the C++ steps then equal Keras's float64 steps (JAX backend) to rounding: about $10^{-17}$ after one step and below $10^{-14}$ after 100 steps, relative to the largest weight (`tests/python/test_nes_train.py`). Two details matter for this comparison:

- The batch has 256 points. Keras 3.15 in float64 behaves as if it scaled the batch gradient by $1/N$ rounded to float32. With 300 points, its updates differ from the Adam formula applied to the exact gradient by the relative amount $2.4 \times 10^{-8}\, \varepsilon / (\sqrt{v} + \varepsilon)$ (a fit over all weights, with residuals of $10^{-14}$), and $2.4 \times 10^{-8}$ is the float32 rounding error of $1/300$. For $N = 256$ the rounding is exact and the difference disappears. The C++ code uses $1/N$ in float64.
- The run is short. Training amplifies rounding differences, for example through the sign of residuals near zero (the loss uses $|e|$) and through Adam's division by $\sqrt{v} + \varepsilon$ for gradients near zero. After 1000 full-batch steps the weights of the two implementations differ by $10^{-8}$ (NES-OP) and $10^{-2}$ (NES-TP).

Longer runs, with shuffled mini-batches, can therefore end in different minima even though each step is the same computation. Keras shuffles with its own random generator, so its mini-batches also differ from the C++ ones. The NES-OP gallery in the main README shows networks trained by Keras and by nes_cpp from the same start.

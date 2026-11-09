# Factored fast marching method

`nes_cpp::fmm` (in `src/fmm.cpp`) solves the eikonal equation for a point source on a regular 2-D or 3-D grid with the factored fast marching method of Treister & Haber (2016), first or second order.

## Problem and grid

The traveltime $T$ from a source at $\mathbf{x}_s$ solves

$$|\nabla T(\mathbf{x})| = s(\mathbf{x}) = \frac{1}{v(\mathbf{x})}, \qquad T(\mathbf{x}_s) = 0,$$

where $v$ is the velocity and $s$ the slowness.

**Grid.** The grid has `dim` = 2 or 3 axes in coordinate order $(x, z)$ or $(x, y, z)$, z last. Axis $k$ has $n_k$ nodes with spacing $h_k$, and node 0 is at the origin, so the node with multi-index $(i_0, i_1, \dots)$ is at $x_k = i_k h_k$. The source is a node, given by its multi-index.

**Memory layout.** Arrays on the grid (velocity in, traveltime out) are flat, C-order (row-major): the last axis is contiguous. The flat index of a node is $\sum_k i_k\,\mathrm{stride}_k$ with

$$\mathrm{stride}_{\mathrm{dim}-1} = 1, \qquad \mathrm{stride}_k = \mathrm{stride}_{k+1}\, n_{k+1}.$$

The neighbours of node $i$ along axis $k$ are $i - \mathrm{stride}_k$ and $i + \mathrm{stride}_k$; they exist when $i_k > 0$ and $i_k < n_k - 1$, where $i_k = \lfloor i / \mathrm{stride}_k \rfloor \bmod n_k$. Strides let one code path serve 2-D and 3-D grids.

## Factorization

Near a point source, $T$ behaves like a cone: its gradient jumps at $\mathbf{x}_s$, and a finite-difference scheme for $T$ is only first-order accurate overall even with a second-order stencil. The factored method writes

$$T = T_0\,\tau, \qquad T_0(\mathbf{x}) = s_0\,|\mathbf{x} - \mathbf{x}_s|, \qquad s_0 = s(\mathbf{x}_s).$$

$T_0$ is the exact traveltime in a homogeneous medium with slowness $s_0$, and it holds the singularity. Its gradient is known in closed form:

$$\nabla T_0 = s_0\,\frac{\mathbf{x} - \mathbf{x}_s}{|\mathbf{x} - \mathbf{x}_s|} = s_0^2\,\frac{\mathbf{x} - \mathbf{x}_s}{T_0}.$$

The correction factor $\tau$ is smooth near the source, $\tau(\mathbf{x}_s) = 1$, and $\tau \equiv 1$ in a homogeneous medium. By the product rule the equation for $\tau$ is

$$|T_0\,\nabla\tau + \tau\,\nabla T_0|^2 = s^2.$$

The code computes $T_0$ once for every node and solves for $\tau$.

*Comparison with eikonalfm.* `eikonalfm.factored_fast_marching` returns $\tau_1$ with $T = \tau_1\,|\mathbf{x} - \mathbf{x}_s|$, so $\tau_1 = s_0\,\tau$. Always compare $T$, not the factors.

## First-order upwind discretization

Fix a node $i$ that is not yet accepted (see [Marching](#marching)). Along each axis $k$, the update uses one neighbour, the **upwind** one: of the two neighbours along $k$, those that are accepted are candidates, and the one with the smaller $T$ is chosen; on a tie the backward one ($i - \mathrm{stride}_k$) wins. Let $\sigma_k = -1$ for the backward neighbour and $+1$ for the forward one, and let $\tau_{\mathrm{nb}}$ be its $\tau$. An axis with no accepted neighbour is left out.

The one-sided difference toward the neighbour approximates $\partial_k \tau$, while $T_0$ and $\partial_k T_0$ are exact at node $i$:

$$\partial_k T \approx T_{0,i}\,\frac{\tau_i - \tau_{\mathrm{nb}}}{-\sigma_k h_k} + \tau_i\,\partial_k T_0(\mathbf{x}_i).$$

Multiplying by $-\sigma_k$ (which does not change the square) gives the axis term $\alpha_k(\tau_i - \beta_k)$ with

$$\alpha_k = \frac{T_{0,i}}{h_k} - \sigma_k\,\partial_k T_0(\mathbf{x}_i), \qquad \beta_k = \frac{T_{0,i}\,\tau_{\mathrm{nb}}}{h_k\,\alpha_k}.$$

The discrete equation for $\tau_i$ is eq. (3.11) of Treister & Haber:

$$\sum_k \max\{\alpha_k(\tau_i - \beta_k),\, 0\}^2 = s_i^2.$$

**$\alpha_k > 0$.** Write $\Delta_k = x_{i,k} - x_{s,k}$ and $d = |\mathbf{x}_i - \mathbf{x}_s|$, so $\alpha_k = s_0(d/h_k - \sigma_k \Delta_k/d)$. If the neighbour lies toward the source, $\sigma_k\Delta_k < 0$ and $\alpha_k > 0$. Otherwise $|\Delta_k| = m h_k$ with an integer $m \ge 1$ and $d^2 \ge m^2 h_k^2 \ge m h_k^2 = |\Delta_k| h_k$, so $\alpha_k \ge 0$. Equality needs $d = h_k$, a node next to the source along $k$; but such a node always takes the source ($T = 0$) as its upwind neighbour, so $\alpha_k > 0$ in every update. Hence a term is non-negative exactly when $\tau_i \ge \beta_k$: $\beta_k$ is the value of $\tau_i$ at which axis $k$ stops being upwind.

## Second-order upwind discretization

With `order = 2`, an axis uses the one-sided second-order difference when the node after the upwind neighbour (two steps from $i$ in direction $\sigma_k$) is inside the grid, is accepted, and has a smaller traveltime than the neighbour:

$$T_{\mathrm{nb2}} < T_{\mathrm{nb}}.$$

Otherwise that axis falls back to the first-order term above. The condition is eq. (2.7) of Treister & Haber, applied to $T$ as in their Section 3.1; it keeps the two upwind values in causal order, so the stencil does not reach across a wavefront. (They write $\ge$ for backward and $>$ for forward neighbours; the code uses the strict inequality on both sides, like eikonalfm.) Each axis decides its order separately, so one update can mix a second-order axis with a first-order one.

The derivative toward the neighbours is

$$\partial_k \tau \approx \frac{3\tau_i - 4\tau_{\mathrm{nb}} + \tau_{\mathrm{nb2}}}{-2\sigma_k h_k},$$

and the same steps as for first order give the axis term $\alpha_k(\tau_i - \beta_k)$ with

$$\alpha_k = \frac{3\,T_{0,i}}{2h_k} - \sigma_k\,\partial_k T_0(\mathbf{x}_i), \qquad \beta_k = \frac{T_{0,i}\,(4\tau_{\mathrm{nb}} - \tau_{\mathrm{nb2}})}{2h_k\,\alpha_k}.$$

Here $\alpha_k = s_0\left(\frac{3d}{2h_k} - \frac{\sigma_k\Delta_k}{d}\right) > 0$ always. If the neighbour lies toward the source this is clear. Otherwise $|\Delta_k| = m h_k$ with $m \ge 1$, and $d^2 \ge m^2 h_k^2$ gives $\frac{3d^2}{2h_k} \ge \frac{3}{2} m^2 h_k > m h_k = |\Delta_k|$, that is $\frac{3d}{2h_k} > \frac{|\Delta_k|}{d}$. The equation, its solution and the validity check are the same as for first order.

**Near the source.** For a neighbour of the source along $k$, the upwind neighbour is the source and the node after it lies on the other side of the source with a larger $T$, so that axis uses first order. Because $\tau$ is smooth (nearly constant) at the source, this costs little accuracy, unlike in the unfactored method, where $T$ has a kink there (Treister & Haber, Section 3).

**Accuracy.** First order converges as $O(h)$ in every norm. Second order converges as $O(h^2)$ only where the second-order stencil is used. Along the curves where a component of $\nabla T$ changes sign (in a vertical gradient: the vertical line through the source, and the curve of ray turning points), the condition above fails and the update falls back to first order. These nodes dominate the max norm, so the max error shrinks only by about 3 when $h$ halves, while the RMS error shrinks by about 4 (Treister & Haber, Section 5 and Table 2, which the tests reproduce). With the source inside the domain rather than on its boundary, the fallback curves cross more of the grid and the RMS ratio drops to about 3.4–3.6; eikonalfm gives the same errors.

**Not implemented.** Treister & Haber (Section 3.2) also describe an optional correction that switches to unfactored differences where the factored and unfactored derivatives disagree in sign, to enforce monotonicity; eikonalfm does not implement it either.

## Solving the update

Take the terms of all axes that have an accepted neighbour, ignoring the max for now. Completing the square with

$$A = \sum_k \alpha_k^2, \qquad \bar\beta = \frac{1}{A}\sum_k \alpha_k^2 \beta_k, \qquad V = \sum_k \alpha_k^2 (\beta_k - \bar\beta)^2$$

gives $\sum_k \alpha_k^2(\tau - \beta_k)^2 = A(\tau - \bar\beta)^2 + V$, so the quadratic has real roots when $s_i^2 \ge V$, and the larger root is

$$\tau_i = \bar\beta + \sqrt{\frac{s_i^2 - V}{A}}.$$

The larger root is the causal one: the traveltime grows away from the accepted neighbours. This form is algebraically the textbook formula $(-b + \sqrt{b^2 - 4ac})/2a$, but it avoids its cancellation: far from the source $A\bar\beta^2$ and $A\bar\beta^2 + V - s_i^2$ are large and nearly equal, while $s_i^2 - V$ is computed directly.

**No real root.** If $s_i^2 < V$, the term with the largest $\beta_k$ is dropped and the equation solved again with the rest (Treister & Haber, Algorithm 2). The terms are sorted by $\beta_k$ once, so the dropped term is always the last. A single term always has a real root: $V = 0$ and $\tau_i = \beta_k + s_i/\alpha_k$. So the loop ends after at most `dim` solves.

**Downwind terms are kept.** Because of the max in eq. (3.11), a term with $\tau_i < \beta_k$ points downwind and should count as zero, and Treister & Haber (Algorithm 2) also drop terms until $\tau_i \ge \max_k \beta_k$. Like eikonalfm, nes_cpp checks only the discriminant and keeps such terms. In smooth models this never happens. In the smoothed Marmousi model a second-order solve meets about 2,000 downwind solutions, and the two rules give traveltimes that differ by up to $3 \times 10^{-3}$ relative. Against a reference on a 4 times finer grid both are about equally accurate: the RMS error of the rule used here is 3–4 % lower at 12.5 m spacing, and equal at 6.25 m. Matching eikonalfm, the solver NES uses for its reference traveltimes, decided the choice.

*Comparison with eikonalfm.* The discrete scheme is the same: upwind neighbours, the condition for second order, $\alpha_k$, $\beta_k$, and the rule for dropping terms. The order of the arithmetic differs: nes_cpp solves for $\tau = \tau_1/s_0$, uses the completed-square root, and computes $T_0$ from integer index differences, while `eikonalfm.distance` builds coordinates with `numpy.linspace` (its distance at the source is about $10^{-16}$, not 0). eikonalfm also accepts all nodes of equal $T$ at once. These differences stay at round-off level, except where one flips a comparison between two nearly equal traveltimes (the choice of the upwind neighbour or of the stencil order); the nodes downstream of such a flip then differ by more (README, "Validation").

**Homogeneous medium.** If $s \equiv s_0$ and every accepted neighbour has $\tau_{\mathrm{nb}} = \tau_{\mathrm{nb2}} = 1$, then $\alpha_k(1 - \beta_k) = -\sigma_k\,\partial_k T_0$ for first- and second-order terms alike. The upwind neighbour lies toward the source wherever $\partial_k T_0 \neq 0$, so every term is non-negative and $\sum_k (\partial_k T_0)^2 = |\nabla T_0|^2 = s_0^2$. Hence $\tau_i = 1$ solves the discrete equation exactly. By induction over the marching order the computed $T$ equals $s_0|\mathbf{x} - \mathbf{x}_s|$ up to round-off, which the tests check.

## Source initialization

The source node gets $\tau = 1$ and $T = 0$ and is accepted first. $T_0 = 0$ there, so $\nabla T_0$ and $\beta_k$ are undefined at the source; the source is never updated because it is accepted before any other node.

## Marching

Every node is either **accepted** (its value is final) or not. Unaccepted nodes start with $\tau = \infty$. The algorithm keeps a min-heap of `(T, node)` pairs, the **front**:

1. Push the source with $T = 0$.
2. Pop the pair with the smallest $T$. If the node is already accepted, or its current $T$ differs from the popped one, the entry is **stale** (see below): skip it.
3. Accept the node. For each neighbour that is not accepted, recompute $\tau$ from all its accepted neighbours ([first](#first-order-upwind-discretization) or [second](#second-order-upwind-discretization) order). If the value changed, store it and push the new `(T, node)` pair.
4. Repeat from step 2 until the heap is empty.

**Stale entries.** The C++ standard heap (`std::priority_queue`) cannot change the key of an entry already inside it. Instead of a heap with back-links from nodes to heap positions, a changed node is pushed again, and old entries are recognised when popped because their key no longer equals the node's current $T$ (Treister & Haber do the same, Section 2.1). The heap then holds a few more entries, at most one per update, which adds little to its depth.

**Recompute and overwrite.** Each update solves the equation again from all neighbours accepted so far and overwrites the old value, as in Treister & Haber (Algorithm 1, step 4). The old value is not kept as a lower bound.

**Ties.** Pairs compare by $T$ first and by node index second, so among equal $T$ the smaller flat index is accepted first. The result therefore does not depend on how a standard library implements its heap.

**Cost.** Each node is accepted once and updated at most $2\,\mathrm{dim}$ times; each heap operation costs $O(\log n)$, so a solve costs $O(n \log n)$ for $n$ nodes and is sequential.

## Output

$T = T_0\,\tau$ at every node, in the same C-order layout as the velocity, in the units of $h$ divided by the units of $v$ (km and km/s give s).

## Many sources

`nes_cpp::fmm_multi` computes a traveltime table for several sources: a flat C-order array of shape $(n_{\mathrm{sources}}, n_0, n_1[, n_2])$, where entry $s\,n + i$ is $T$ at node $i$ for source $s$ and $n$ is the number of nodes.

**Where the parallelism is.** One solve is sequential: each accepted node can change the next one taken from the heap. Different sources are independent, so `fmm_multi` runs them in parallel with OpenMP (`#pragma omp parallel for` over sources). Each source is solved from start to end by one thread, with the same code as `fmm`, and writes only its own slice of the table. The threads share the velocity array, which they only read; each solve allocates its own $T_0$, $\tau$, flags and heap.

**Scheduling.** `schedule(dynamic)` hands the next unsolved source to whichever thread is free, so a thread that finishes early does not wait while solves of slightly different cost run elsewhere. The number of threads is OpenMP's default (all logical cores), or `OMP_NUM_THREADS`.

**Reproducibility.** No floating-point result is shared or reduced between threads, so the table does not depend on the number of threads or on the schedule: it equals separate `fmm` calls bit for bit. The tests check this with 1 and 4 threads.

**Errors.** All inputs are checked before the parallel loop. An exception must not leave an OpenMP parallel region (the program would terminate), so an exception thrown inside the loop, such as `std::bad_alloc`, is caught, the first one is kept, and it is rethrown after the loop.

**Memory.** Each running solve holds about 17 bytes per node ($T_0$, $\tau$, one flag byte) plus its heap, so $p$ threads need about $17\,p\,n$ bytes on top of the $8\,n_{\mathrm{sources}}\,n$-byte table.

## Reference

Treister, E., & Haber, E. (2016). A fast marching algorithm for the factored eikonal equation. *Journal of Computational Physics*, 324, 210–225. arXiv:1607.00973

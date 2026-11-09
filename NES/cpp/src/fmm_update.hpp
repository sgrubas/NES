// The update equation of the factored fast marching method (docs/FMM.md, "Solving the update").
// Internal to the library (not in include/): declared here so that tests can check it directly.
#pragma once

#include <span>

namespace nes_cpp::detail {

// One axis term alpha * (tau - beta) of the update equation (docs/FMM.md, "First-order upwind
// discretization").
struct Term {
    double alpha;
    double beta;
};

// Solves sum_k alpha_k^2 * (tau - beta_k)^2 = slowness^2 for its larger root; if the roots are
// not real, drops the term with the largest beta and solves again. Terms with tau < beta_k are
// kept, as in eikonalfm (docs/FMM.md, "Downwind terms are kept"). Needs at least one term and
// every alpha_k > 0. Reorders `terms`.
double solve_update(std::span<Term> terms, double slowness);

} // namespace nes_cpp::detail

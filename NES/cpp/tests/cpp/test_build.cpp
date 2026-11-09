// Smoke tests for the build: Catch2 runs, the Eigen headers are found, and OpenMP,
// when the compiler supports it, starts a parallel region.

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#ifdef _OPENMP // defined by the compiler when OpenMP is enabled
#include <omp.h>
#endif

TEST_CASE("Eigen multiplies a matrix by a vector", "[build]") {
    const Eigen::Matrix2d a{{1.0, 2.0}, {3.0, 4.0}};
    const Eigen::Vector2d x{1.0, 1.0};
    const Eigen::Vector2d y = a * x;
    CHECK(y(0) == 3.0); // exact: small integers are exact in double
    CHECK(y(1) == 7.0);
}

TEST_CASE("OpenMP starts a parallel region", "[build]") {
#ifdef _OPENMP
    int n_threads = 0;
#pragma omp parallel
    {
#pragma omp single // one thread writes; the others wait at the end of the block
        n_threads = omp_get_num_threads();
    }
    CHECK(n_threads >= 1);
#else
    SKIP("built without OpenMP");
#endif
}

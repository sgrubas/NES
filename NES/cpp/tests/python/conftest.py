"""Shared pytest fixtures. Keras is imported inside the fixture, so that collecting the tests works
without Keras (the NES tests then skip)."""
import pytest


@pytest.fixture
def float64():
    """Keras computes in float64 during the test (as in tests/test_nes.py). Besides floatx, the dtype
    policy of new layers has to be set, and JAX needs 64-bit arrays enabled."""
    import keras

    floatx, policy = keras.config.floatx(), keras.config.dtype_policy()
    keras.config.set_floatx("float64")
    keras.config.set_dtype_policy("float64")
    if keras.backend.backend() == "jax":
        import jax
        x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
    yield
    keras.config.set_floatx(floatx)
    keras.config.set_dtype_policy(policy)
    if keras.backend.backend() == "jax":
        jax.config.update("jax_enable_x64", x64)

from pathlib import Path

from qqt_rl.training.jax_cache import cache_environment, resolve_cache_dir


def test_default_cache_is_scoped_to_run_directory(tmp_path):
    cache = resolve_cache_dir(run_dir=tmp_path / "run")
    assert cache == (tmp_path / "run" / "cache" / "jax").resolve()


def test_explicit_cache_directory_wins_and_environment_is_complete(tmp_path):
    explicit = tmp_path / "shared-cache"
    environment = cache_environment(
        run_dir=tmp_path / "run", explicit=explicit,
        min_compile_time_secs=0.25, min_entry_size_bytes=4096)
    assert environment == {
        "JAX_COMPILATION_CACHE_DIR": str(explicit.resolve()),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0.25",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "4096",
    }


def test_cache_requires_run_or_explicit_directory():
    try:
        resolve_cache_dir()
    except ValueError as error:
        assert "run_dir or explicit" in str(error)
    else:
        raise AssertionError("unscoped cache directory accepted")

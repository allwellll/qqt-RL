"""JAX 持久编译缓存配置，确保缓存只落在运行目录或显式路径。"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_cache_dir(run_dir: str | Path | None = None,
                      explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if run_dir is None:
        raise ValueError("run_dir or explicit cache directory is required")
    return (Path(run_dir).expanduser().resolve() / "cache" / "jax")


def cache_environment(run_dir: str | Path | None = None,
                      explicit: str | Path | None = None,
                      min_compile_time_secs: float = 0.0,
                      min_entry_size_bytes: int = 0) -> dict[str, str]:
    cache_dir = resolve_cache_dir(run_dir=run_dir, explicit=explicit)
    return {
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": str(min_compile_time_secs),
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": str(min_entry_size_bytes),
    }


def configure_persistent_cache(run_dir: str | Path | None = None,
                               explicit: str | Path | None = None,
                               min_compile_time_secs: float = 0.0,
                               min_entry_size_bytes: int = 0) -> Path:
    """在首次 JIT 前启用跨进程缓存；目录创建不进入 Git。"""
    cache_dir = resolve_cache_dir(run_dir=run_dir, explicit=explicit)
    cache_dir.mkdir(parents=True, exist_ok=True)
    values = cache_environment(
        run_dir=run_dir, explicit=explicit,
        min_compile_time_secs=min_compile_time_secs,
        min_entry_size_bytes=min_entry_size_bytes)
    os.environ.update(values)
    import jax
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)
    return cache_dir

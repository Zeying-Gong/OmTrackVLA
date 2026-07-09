"""Compatibility helpers for locally installed HuggingFace stacks."""

from __future__ import annotations

import os


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_hf_ssl_verification() -> None:
    """Optionally disable HF Hub SSL verification for managed lab hosts."""
    if not _env_true("OMTRACKVLA_DISABLE_HF_SSL_VERIFY"):
        return
    try:
        import requests
        from huggingface_hub import configure_http_backend

        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=backend_factory)
    except Exception:
        pass


def disable_deepspeed_auto_import() -> None:
    """Prevent Transformers from importing DeepSpeed when it is not used.

    Some environments install DeepSpeed globally. Transformers then imports it
    while loading model classes, even for plain inference/training code that
    never uses DeepSpeed. Keeping the availability hook false avoids unrelated
    DeepSpeed accelerator initialization side effects.
    """
    os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
    configure_hf_ssl_verification()

    def _false() -> bool:
        return False

    try:
        import transformers.integrations.deepspeed as ds_integration

        ds_integration.is_deepspeed_available = _false
    except Exception:
        return

    try:
        import transformers.integrations as integrations

        integrations.is_deepspeed_available = _false
    except Exception:
        pass

    try:
        import accelerate.utils.imports as accelerate_imports

        accelerate_imports.is_deepspeed_available = _false
    except Exception:
        pass

"""MemLayer package exports.

All heavy submodules are imported lazily via :func:`__getattr__` so that a
simple ``import memlayer`` is cheap and works in constrained environments.

Supported exports:

- ``Memory``
- ``OpenAI``
- ``Claude``
- ``Gemini``
- ``Ollama``
"""

__version__ = "0.1.6"


def __getattr__(name):
    """Lazily import and return top-level symbols.

    This keeps ``import memlayer`` fast and avoids importing heavy optional
    dependencies at package import time. When a symbol is first accessed the
    real implementation is imported from the appropriate submodule.
    """
    if name == "Memory":
        from .client import Memory

        return Memory
    if name == "OpenAI":
        from .wrappers import OpenAI

        return OpenAI
    if name == "Claude":
        from .wrappers import Claude

        return Claude
    if name == "Gemini":
        from .wrappers import Gemini

        return Gemini
    if name == "Ollama":
        from .wrappers import Ollama

        return Ollama
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["Memory", "OpenAI", "Claude", "Gemini", "Ollama"]


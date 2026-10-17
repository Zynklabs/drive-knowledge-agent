"""Where the index lives, computed from config without connecting (cheap enough for `status`)."""

from __future__ import annotations

from drive_agent.core.config import Settings


def location(settings: Settings) -> str:
    """Server URL, ``:memory:``, or the embedded-mode directory."""
    config = settings.vector_store
    return config.url or str(config.persist_dir or settings.data_dir / config.provider)


def namespace_for(settings: Settings) -> str:
    """Sync-state namespace: a different server or collection gets its own records."""
    config = settings.vector_store
    return f"{config.provider}:{location(settings)}/{config.collection}"


def describe_location(settings: Settings) -> str:
    config = settings.vector_store
    return f"{config.provider} {location(settings)} / {config.collection}"

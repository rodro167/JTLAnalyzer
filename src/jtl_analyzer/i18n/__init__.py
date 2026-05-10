"""Internationalization support: message catalog lookup."""

import importlib
from typing import Any


def get_message(key: str, lang: str = "en", **kwargs: Any) -> str:
    """Retrieve and format a message from the catalog for the given language.

    Falls back to English if the requested language module does not exist or
    the key is absent from that language's catalog.

    Args:
        key: Message key defined in a language module (e.g. ``"ANALYZING_FILE"``).
        lang: ISO 639-1 language code (e.g. ``"en"``, ``"es"``). Defaults to ``"en"``.
        **kwargs: Named values substituted into the message template.

    Returns:
        The formatted message string, or the bare key if it cannot be found.
    """
    try:
        module = importlib.import_module(f"jtl_analyzer.i18n.{lang}")
    except ModuleNotFoundError:
        module = importlib.import_module("jtl_analyzer.i18n.en")

    template: str | None = getattr(module, key, None)

    if template is None and lang != "en":
        en = importlib.import_module("jtl_analyzer.i18n.en")
        template = getattr(en, key, None)

    if template is None:
        return key

    return template.format(**kwargs) if kwargs else template

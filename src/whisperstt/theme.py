"""Warm parchment palette shared by every whisperstt surface.

The five colours below are the whole design system: cream paper, slate ink,
terracotta accent, greige borders and warm slate for secondary text.  Keeping
them in one module means the installer, the model chooser, the progress bar and
the watcher all look like the same program.

Colour is opt-out: it is only emitted to a terminal, and honours ``NO_COLOR``
and ``CLICOLOR=0``.  Set ``FORCE_COLOR=1`` to keep colour when output is piped.
"""

import os
import re
import sys

PALETTE = {
    "cream": "#FAF9F5",
    "slate_ink": "#141413",
    "terracotta": "#D97757",
    "light_greige": "#E8E6DC",
    "warm_slate": "#B1ADA1",
}

RESET = "\033[0m"

_MODIFIERS = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "reverse": "\033[7m",
}

# Approximations for terminals without 24-bit colour.
_PALETTE_256 = {
    "cream": 230,
    "slate_ink": 233,
    "terracotta": 173,
    "light_greige": 253,
    "warm_slate": 145,
}

_PALETTE_16 = {
    "cream": 97,
    "slate_ink": 30,
    "terracotta": 33,
    "light_greige": 37,
    "warm_slate": 90,
}

_BACKGROUND_16 = {
    "cream": 107,
    "slate_ink": 40,
    "terracotta": 43,
    "light_greige": 47,
    "warm_slate": 100,
}

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _rgb(name):
    value = PALETTE[name].lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _light_background():
    """Pick the text side of the palette for the current terminal.

    ``COLORFGBG`` is the de-facto hint used by terminal emulators.  Anyone on
    a cream terminal that does not set it can force the choice with
    ``WHISPERSTT_BACKGROUND=light`` (or ``dark``).
    """
    override = os.environ.get("WHISPERSTT_BACKGROUND", "").lower()
    if override in ("light", "cream"):
        return True
    if override in ("dark", "slate"):
        return False
    value = os.environ.get("COLORFGBG", "")
    if ";" not in value:
        return False
    try:
        return int(value.rsplit(";", 1)[1]) >= 7
    except ValueError:
        return False


def supports_color(stream=None):
    """Return True when ANSI colour should be written to *stream*."""
    if stream is None:
        stream = sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE")
    if force and force != "0":
        return True
    if os.environ.get("CLICOLOR") == "0":
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _depth():
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return 24
    if "256color" in os.environ.get("TERM", "") or colorterm:
        return 256
    return 16


def _colour_code(name, background=False):
    if name not in PALETTE:
        return ""
    depth = _depth()
    if depth == 24:
        red, green, blue = _rgb(name)
        return f"\033[{48 if background else 38};2;{red};{green};{blue}m"
    if depth == 256:
        index = _PALETTE_256[name]
        return f"\033[{48 if background else 38};5;{index}m"
    if background:
        return f"\033[{_BACKGROUND_16[name]}m"
    return f"\033[{_PALETTE_16[name]}m"


def _role(name):
    """Resolve a semantic role to a palette colour for the current terminal."""
    light = _light_background()
    if name in ("primary", "text"):
        return "slate_ink" if light else "cream"
    if name in ("muted", "secondary"):
        return "warm_slate"
    if name in ("accent", "warn", "error"):
        return "terracotta"
    if name in ("rule", "border"):
        return "light_greige" if light else "warm_slate"
    if name in ("on_accent", "on_surface"):
        return "slate_ink"
    if name in PALETTE:
        return name
    return None


def paint(text, *roles, stream=None):
    """Wrap *text* in the ANSI style for *roles*, or return it unchanged."""
    if not text or not supports_color(stream):
        return text
    codes = []
    for role in roles:
        if role in _MODIFIERS:
            codes.append(_MODIFIERS[role])
        elif role.startswith("bg:"):
            code = _colour_code(role[3:], background=True)
            if code:
                codes.append(code)
        else:
            name = _role(role)
            if name:
                code = _colour_code(name)
                if code:
                    codes.append(code)
    if not codes:
        return text
    return "".join(codes) + text + RESET


def visible_len(text):
    """Length of *text* without ANSI escape sequences."""
    return len(_ANSI.sub("", text))


def banner(context=None, stream=None):
    """Return the compact wordmark shown at the start of interactive screens."""
    title = "  " + paint("▌", "accent", stream=stream) + " " + paint(
        "whisperstt", "primary", "bold", stream=stream
    )
    subtitle = context or "books · movies · episodes → subtitles"
    lines = [
        "",
        title,
        "    " + paint(subtitle, "muted", stream=stream),
        "  " + paint("─" * 52, "rule", stream=stream),
        "",
    ]
    return "\n".join(lines)


def heading(text, stream=None):
    """A file or section heading, marked with the terracotta bar."""
    return paint("  ▌ ", "accent", stream=stream) + paint(
        text, "primary", "bold", stream=stream
    )


def mark(kind, stream=None):
    """Return a small status glyph in the palette."""
    glyph = {
        "ok": "✓",
        "fail": "✗",
        "warn": "!",
        "step": "›",
        "skip": "↷",
        "bullet": "●",
        "arrow": "→",
    }.get(kind, "•")
    role = "muted" if kind == "bullet" else "accent"
    return paint(glyph, role, stream=stream)

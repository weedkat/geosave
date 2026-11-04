"""HTML the sidebar shows: legends from a layer's values, and metadata."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

from holoviews.plotting.util import process_cmap

if TYPE_CHECKING:
    from ..layers import Layer


def format_info(values: dict[str, Any]) -> str:
    """Return names and values as HTML; nested values fold into expandable JSON."""
    rows = []
    for name, value in values.items():
        title = html.escape(str(name))
        if isinstance(value, dict | list | tuple):
            text = html.escape(json.dumps(value, indent=2, default=str))
            rows.append(
                f'<details><summary>{title}</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:280px;overflow:auto">{text}</pre></details>'
            )
        else:
            rows.append(f"<div><b>{title}</b><br>{html.escape(str(value))}</div>")
    return (
        '<div style="overflow-wrap:anywhere;font-size:13px;line-height:1.6">'
        + "".join(rows)
        + "</div>"
    )


def format_swatches(colors: dict[Any, str]) -> str:
    """Return one colour square and name per class as HTML."""
    rows = []
    for name, color in colors.items():
        label = html.escape(str(name)).replace("_", " ")
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px"><span style="flex:none;width:14px;height:14px;border:1px solid #8888;border-radius:2px;background:{html.escape(color)}"></span><span>{label}</span></div>'
        )
    return (
        '<div style="display:grid;gap:5px;font-size:13px">' + "".join(rows) + "</div>"
    )


def format_ramp(title: str, cmap: str, limits: tuple[float, float] | None) -> str:
    """Return a titled colour ramp with its low and high values as HTML."""
    # Sixteen stops draw a smooth gradient.
    palette = ",".join(process_cmap(cmap, 16))
    low, high = limits or (0.0, 1.0)
    return f'<div style="font-size:13px"><b>{html.escape(str(title))}</b><div style="height:10px;margin:6px 0;border-radius:3px;background:linear-gradient(to right,{palette})"></div><small>{low:.4g} — {high:.4g}</small></div>'


def format_legend(layer: Layer, *_: Any) -> str:
    """Return a layer's legend as HTML: its caption, class swatches or colour ramp."""
    if layer.caption is not None:
        return f"<small>{html.escape(layer.caption)}</small>"
    if layer.swatches is not None:
        return format_swatches(layer.swatches)
    if layer.ramp is not None:
        title, cmap, limits = layer.ramp
        return format_ramp(title, cmap, limits)
    return ""

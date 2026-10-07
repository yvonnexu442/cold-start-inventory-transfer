"""Small local figure-alignment guard used by public plotting scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def require_matplotlib_panel_alignment(
    fig: Any,
    *,
    json_out: Path,
    tolerance_pt: float = 1.5,
    gutter_tolerance_pt: float = 1.5,
    require_panel_labels: bool = True,
    strict: bool = True,
) -> None:
    """Validate expected axes and record normalized panel geometry."""

    axes = list(fig.axes)
    if strict and not axes:
        raise ValueError("Figure contains no axes")
    if require_panel_labels and strict and len(axes) < 1:
        raise ValueError("Panel-label check requested but no axes were found")
    geometry = []
    for index, axis in enumerate(axes):
        box = axis.get_position()
        geometry.append(
            {
                "axis_index": index,
                "x0": round(float(box.x0), 6),
                "y0": round(float(box.y0), 6),
                "x1": round(float(box.x1), 6),
                "y1": round(float(box.y1), 6),
            }
        )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "schema": "public-matplotlib-panel-alignment-v1",
                "axes": geometry,
                "tolerance_pt": tolerance_pt,
                "gutter_tolerance_pt": gutter_tolerance_pt,
                "require_panel_labels": require_panel_labels,
                "strict": strict,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

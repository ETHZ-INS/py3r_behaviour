from __future__ import annotations

import cv2
import numpy as np

from .compiler import (
    _apply_dynamic_overrides,
    _coords_to_pixels,
    _format_overlay_value,
    _is_valid,
)
from .models import GeometryData, StyleProgram


def resolve_compiled_style(
    compiled_styles: dict[str, dict[object, dict[str, object]]],
    section_name: str,
    item_key,
    frame_idx: int,
) -> dict:
    compiled = compiled_styles[section_name][item_key]
    return _apply_dynamic_overrides(compiled["base"], compiled["dyn"], frame_idx)


def render_into_frame(
    frame: np.ndarray,
    *,
    frame_idx: int,
    geometry: GeometryData,
    styles: StyleProgram,
    copy: bool = True,
) -> np.ndarray:
    target = frame.copy() if copy else frame
    underlay = target.copy()

    valid_polys: list[tuple[np.ndarray, dict]] = []
    for boundary_name, arr in geometry.boundary_arrays:
        poly = arr[frame_idx]
        pix = _coords_to_pixels(
            poly,
            target.shape[1],
            target.shape[0],
            geometry.bounds,
            geometry.pixel_coords,
        )
        if pix is None or len(pix) < 3:
            continue
        if np.any(pix[:, 0] < 0) or np.any(pix[:, 1] < 0):
            continue
        bstyle = resolve_compiled_style(
            styles.compiled_styles,
            "boundaries",
            boundary_name,
            frame_idx,
        )
        valid_polys.append((pix, bstyle))
        alpha = float(bstyle["fill_alpha"])
        if alpha > 0 and bstyle["fill_mode"] == "normal" and bstyle["fill_color"] is not None:
            fill_overlay = target.copy()
            cv2.fillPoly(fill_overlay, [pix], color=bstyle["fill_color"])
            cv2.addWeighted(fill_overlay, alpha, target, 1.0 - alpha, 0.0, dst=target)
        elif alpha > 0 and bstyle["fill_mode"] == "erase":
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pix], color=255)
            m = mask.astype(bool)
            target[m] = (
                (1.0 - alpha) * target[m].astype(np.float32)
                + alpha * underlay[m].astype(np.float32)
            ).astype(np.uint8)
    for pix, bstyle in valid_polys:
        edge_width = int(bstyle["edge_width"])
        edge_color = bstyle["edge_color"]
        if edge_width > 0 and edge_color is not None:
            cv2.polylines(target, [pix], isClosed=True, color=edge_color, thickness=edge_width)

    pts = geometry.points_xy[frame_idx]
    pix_pts = _coords_to_pixels(
        pts,
        target.shape[1],
        target.shape[0],
        geometry.bounds,
        geometry.pixel_coords,
    )
    if pix_pts is not None:
        for line_i, (i1, i2) in enumerate(geometry.lines_idx):
            if _is_valid(pix_pts, i1) and _is_valid(pix_pts, i2):
                lstyle = resolve_compiled_style(
                    styles.compiled_styles,
                    "lines",
                    geometry.line_keys[line_i],
                    frame_idx,
                )
                cv2.line(
                    target,
                    tuple(pix_pts[i1]),
                    tuple(pix_pts[i2]),
                    lstyle["color"],
                    lstyle["width"],
                )
        for point_i in geometry.draw_point_indices:
            p = pix_pts[point_i]
            if p[0] >= 0:
                pstyle = resolve_compiled_style(
                    styles.compiled_styles,
                    "points",
                    geometry.point_names[point_i],
                    frame_idx,
                )
                cv2.circle(target, tuple(p), pstyle["radius"], pstyle["color"], thickness=-1)

    if styles.text_overlays:
        text_section = styles.style.get("text", {})
        ox, oy = text_section.get("origin", (10, 20))
        default_fmt = str(text_section.get("format", ".3f"))
        y = int(oy)
        entries: list[dict] = []
        max_text_w = 0
        for overlay_i, (label, values) in enumerate(styles.text_overlays):
            tstyle = resolve_compiled_style(styles.compiled_styles, "text", label, frame_idx)
            if y < 0:
                y += tstyle["line_height"]
            if values is None:
                entries.append({"y": y, "line_height": tstyle["line_height"], "spacer": True})
                y += tstyle["line_height"]
                continue
            fmt = default_fmt if tstyle.get("format") is None else str(tstyle["format"])
            color_arr = styles.text_colors[overlay_i]
            dyn_text = styles.compiled_styles["text"][label]["dyn"]
            if dyn_text is not None and "color" in dyn_text:
                color = tstyle["color"]
            elif color_arr is None:
                color = tstyle["color"]
            else:
                color = tuple(map(int, color_arr[frame_idx]))
            txt = (
                f"{label}: "
                f"{_format_overlay_value(values[frame_idx], fmt, as_bool=tstyle['as_bool'])}"
            )
            (tw, _), _ = cv2.getTextSize(
                txt,
                cv2.FONT_HERSHEY_SIMPLEX,
                tstyle["font_scale"],
                tstyle["thickness"],
            )
            max_text_w = max(max_text_w, int(tw))
            entries.append(
                {
                    "y": y,
                    "line_height": tstyle["line_height"],
                    "spacer": False,
                    "txt": txt,
                    "style": tstyle,
                    "color": color,
                }
            )
            y += tstyle["line_height"]

        panel_cfg = text_section.get("panel", {})
        panel_enabled = (
            bool(panel_cfg.get("enabled", False))
            if isinstance(panel_cfg, dict)
            else bool(panel_cfg)
        )
        if panel_enabled and entries and max_text_w > 0:
            pad = int(panel_cfg.get("padding", 6)) if isinstance(panel_cfg, dict) else 6
            alpha = float(panel_cfg.get("alpha", 0.45)) if isinstance(panel_cfg, dict) else 0.45
            alpha = float(np.clip(alpha, 0.0, 1.0))
            panel_color = (
                tuple(map(int, panel_cfg.get("color", (0, 0, 0))))
                if isinstance(panel_cfg, dict)
                else (0, 0, 0)
            )
            top = min(int(e["y"] - e["line_height"] + 4) for e in entries)
            bottom = max(int(e["y"] + 6) for e in entries)
            left = int(ox) - pad
            right = int(ox) + max_text_w + pad
            left = max(0, left)
            right = min(target.shape[1] - 1, right)
            top = max(0, top)
            bottom = min(target.shape[0] - 1, bottom)
            if right > left and bottom > top:
                if alpha >= 1.0:
                    cv2.rectangle(target, (left, top), (right, bottom), panel_color, thickness=-1)
                elif alpha > 0:
                    panel_overlay = target.copy()
                    cv2.rectangle(
                        panel_overlay,
                        (left, top),
                        (right, bottom),
                        panel_color,
                        thickness=-1,
                    )
                    cv2.addWeighted(panel_overlay, alpha, target, 1.0 - alpha, 0.0, dst=target)

        for entry in entries:
            if entry["spacer"]:
                continue
            tstyle = entry["style"]
            if tstyle["outline_thickness"] > 0:
                cv2.putText(
                    target,
                    entry["txt"],
                    (int(ox), int(entry["y"])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    tstyle["font_scale"],
                    tstyle["outline_color"],
                    tstyle["outline_thickness"],
                    cv2.LINE_AA,
                )
            cv2.putText(
                target,
                entry["txt"],
                (int(ox), int(entry["y"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                tstyle["font_scale"],
                entry["color"],
                tstyle["thickness"],
                cv2.LINE_AA,
            )
    return target

from __future__ import annotations

import cv2
import numpy as np

from .compiler import (
    _coords_to_pixels,
    _is_valid,
)
from .models import CompiledScene


def render_frame(
    scene: CompiledScene, frame_idx: int, frame: np.ndarray | None = None
) -> np.ndarray:
    if frame_idx < 0 or frame_idx >= scene.points_xy.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} out of range")
    if frame is None:
        w, h = scene.canvas_size
        target = np.full((h, w, 3), scene.bg_color, dtype=np.uint8)
    else:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        target = frame
    underlay = target.copy()
    frame_styles = scene.styles_by_frame[frame_idx]

    valid_polys: list[tuple[np.ndarray, dict]] = []
    for boundary_name, arr in scene.boundaries_xy.items():
        poly = arr[frame_idx]
        pix = _coords_to_pixels(
            poly,
            target.shape[1],
            target.shape[0],
            scene.bounds,
            scene.pixel_coords,
        )
        if pix is None or len(pix) < 3:
            continue
        if np.any(pix[:, 0] < 0) or np.any(pix[:, 1] < 0):
            continue
        bstyle = frame_styles["boundaries"][boundary_name]
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

    pts = scene.points_xy[frame_idx]
    pix_pts = _coords_to_pixels(
        pts,
        target.shape[1],
        target.shape[0],
        scene.bounds,
        scene.pixel_coords,
    )
    if pix_pts is not None:
        for line_i, (i1, i2) in enumerate(scene.lines_idx):
            if _is_valid(pix_pts, i1) and _is_valid(pix_pts, i2):
                lstyle = frame_styles["lines"][scene.line_keys[line_i]]
                cv2.line(
                    target,
                    tuple(pix_pts[i1]),
                    tuple(pix_pts[i2]),
                    lstyle["color"],
                    lstyle["width"],
                )
        for point_i in scene.draw_point_indices:
            p = pix_pts[point_i]
            if p[0] >= 0:
                pstyle = frame_styles["points"][scene.point_names[point_i]]
                cv2.circle(target, tuple(p), pstyle["radius"], pstyle["color"], thickness=-1)

    entries = scene.text_by_frame[frame_idx]
    if entries:
        text_section = scene.text_config
        ox, oy = text_section.get("origin", (10, 20))
        y = int(oy)
        max_text_w = 0
        positioned: list[dict] = []
        for entry in entries:
            tstyle = entry["style"]
            if y < 0:
                y += tstyle["line_height"]
            if entry["spacer"]:
                positioned.append({"y": y, "line_height": tstyle["line_height"], "spacer": True})
                y += tstyle["line_height"]
                continue
            txt = str(entry["text"])
            (tw, _), _ = cv2.getTextSize(
                txt,
                cv2.FONT_HERSHEY_SIMPLEX,
                tstyle["font_scale"],
                tstyle["thickness"],
            )
            max_text_w = max(max_text_w, int(tw))
            positioned.append(
                {
                    "y": y,
                    "line_height": tstyle["line_height"],
                    "spacer": False,
                    "txt": txt,
                    "style": tstyle,
                    "color": tuple(map(int, entry["color"])),
                }
            )
            y += tstyle["line_height"]

        panel_cfg = text_section.get("panel", {})
        panel_enabled = (
            bool(panel_cfg.get("enabled", False))
            if isinstance(panel_cfg, dict)
            else bool(panel_cfg)
        )
        if panel_enabled and positioned and max_text_w > 0:
            pad = int(panel_cfg.get("padding", 6)) if isinstance(panel_cfg, dict) else 6
            alpha = float(panel_cfg.get("alpha", 0.45)) if isinstance(panel_cfg, dict) else 0.45
            alpha = float(np.clip(alpha, 0.0, 1.0))
            panel_color = (
                tuple(map(int, panel_cfg.get("color", (0, 0, 0))))
                if isinstance(panel_cfg, dict)
                else (0, 0, 0)
            )
            top = min(int(e["y"] - e["line_height"] + 4) for e in positioned)
            bottom = max(int(e["y"] + 6) for e in positioned)
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

        for entry in positioned:
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

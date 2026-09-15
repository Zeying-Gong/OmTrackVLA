"""Qualitative dashboard for Architecture v1 checkpoint inspection."""
from __future__ import annotations

import math
import textwrap
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch


CANVAS_WIDTH = 1512
CANVAS_HEIGHT = 1090
IMAGE_WIDTH = 504
IMAGE_HEIGHT = 280


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _bgr_image(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("RGB visualization tensor must have shape [3,H,W]")
    rgb = (_numpy(value).transpose(1, 2, 0).clip(0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _caption(image: np.ndarray, title: str, color=(255, 255, 255)) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def _attention_map(
    value: torch.Tensor | np.ndarray,
    grid_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    flat = _numpy(value) if isinstance(value, torch.Tensor) else np.asarray(value)
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    expected = int(grid_shape[0] * grid_shape[1])
    if flat.size != expected:
        raise ValueError(
            f"attention grid contains {flat.size} values, expected {expected}"
        )
    if not np.isfinite(flat).all():
        raise ValueError("attention visualization contains non-finite values")
    lower, upper = np.percentile(flat, (2.0, 98.0))
    if float(upper - lower) < 1.0e-12:
        normalized = np.full_like(flat, 0.5)
    else:
        normalized = np.clip((flat - lower) / (upper - lower), 0.0, 1.0)
    grid = (normalized.reshape(grid_shape) * 255.0).astype(np.uint8)
    width, height = output_shape
    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)


def _overlay_attention(
    image: np.ndarray,
    value: torch.Tensor | np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    intensity = _attention_map(value, grid_shape, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image, 0.48, heatmap, 0.52, 0.0)


def _attention_center(
    value: torch.Tensor | np.ndarray,
    grid_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    weights = _numpy(value) if isinstance(value, torch.Tensor) else np.asarray(value)
    weights = np.asarray(weights, dtype=np.float64).reshape(grid_shape)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return image_shape[1] // 2, image_shape[0] // 2
    rows, columns = np.indices(grid_shape)
    x = float(((columns + 0.5) * weights).sum() / total) * image_shape[1] / grid_shape[1]
    y = float(((rows + 0.5) * weights).sum() / total) * image_shape[0] / grid_shape[0]
    return int(round(x)), int(round(y))


def _draw_attention_center(
    image: np.ndarray,
    value: torch.Tensor | np.ndarray,
    grid_shape: tuple[int, int],
) -> None:
    center = _attention_center(value, grid_shape, image.shape[:2])
    cv2.drawMarker(
        image,
        center,
        (255, 255, 255),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )


def _draw_projected_uwb(
    image: np.ndarray,
    mean_uv: Sequence[float],
    covariance_uv: np.ndarray,
    enabled: bool,
) -> None:
    mean = np.asarray(mean_uv, dtype=np.float64)
    covariance = np.asarray(covariance_uv, dtype=np.float64)
    if not enabled or mean.shape != (2,) or covariance.shape != (2, 2):
        return
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1.0e-8)
    eigenvectors = eigenvectors[:, order]
    axes = tuple(max(1, min(10000, int(round(2.0 * math.sqrt(value))))) for value in eigenvalues)
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    center = tuple(int(round(value)) for value in mean)
    cv2.ellipse(image, center, axes, angle, 0, 360, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(
        image, center, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA
    )


def _history_strip(batch: Mapping[str, torch.Tensor]) -> np.ndarray:
    initial = _bgr_image(batch["initial_rgb"][0])
    bbox = _numpy(batch["initial_bbox"][0])
    height, width = initial.shape[:2]
    x1, y1, x2, y2 = (
        int(round(float(value) * scale))
        for value, scale in zip(bbox, (width, height, width, height))
    )
    cv2.rectangle(initial, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)
    frames = [initial] + [_bgr_image(frame) for frame in batch["ego_rgb"][0]]
    labels = (
        "INITIAL RGB + admitted bbox",
        "history t-3",
        "history t-2",
        "history t-1",
        "history t (policy frame)",
    )
    cell_width = CANVAS_WIDTH // len(frames)
    strip_height = 180
    strip = np.full((strip_height, CANVAS_WIDTH, 3), 18, dtype=np.uint8)
    for index, (frame, label) in enumerate(zip(frames, labels)):
        panel = cv2.resize(
            frame, (cell_width - 4, strip_height - 4), interpolation=cv2.INTER_AREA
        )
        _caption(panel, label, (0, 255, 255) if index == 0 else (255, 255, 255))
        left = index * cell_width
        strip[2 : 2 + panel.shape[0], left + 2 : left + 2 + panel.shape[1]] = panel
    return strip


def _waypoint_panel(
    predicted: np.ndarray,
    expert: np.ndarray,
    uwb_xy: np.ndarray,
    uwb_covariance: np.ndarray,
    *,
    width: int,
    height: int,
    prediction_label: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    trajectories = np.concatenate((predicted, expert, uwb_xy.reshape(1, 2)), axis=0)
    forward = trajectories[:, 0]
    left = trajectories[:, 1]
    f_min = min(-0.25, float(forward.min()) - 0.15)
    f_max = max(0.75, float(forward.max()) + 0.15)
    l_min = min(-0.75, float(left.min()) - 0.15)
    l_max = max(0.75, float(left.max()) + 0.15)
    margin_left, margin_right, margin_top, margin_bottom = 62, 20, 42, 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def project(points: np.ndarray) -> np.ndarray:
        x = margin_left + (l_max - points[:, 1]) / (l_max - l_min) * plot_width
        y = margin_top + (f_max - points[:, 0]) / (f_max - f_min) * plot_height
        return np.stack((x, y), axis=1).round().astype(np.int32)

    step = 0.5
    for value in np.arange(math.ceil(f_min / step) * step, f_max + step, step):
        endpoints = project(np.asarray([[value, l_min], [value, l_max]]))
        cv2.line(panel, tuple(endpoints[0]), tuple(endpoints[1]), (55, 55, 55), 1)
        cv2.putText(
            panel,
            f"{value:.1f}",
            (8, int(endpoints[0, 1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    for value in np.arange(math.ceil(l_min / step) * step, l_max + step, step):
        endpoints = project(np.asarray([[f_min, value], [f_max, value]]))
        cv2.line(panel, tuple(endpoints[0]), tuple(endpoints[1]), (55, 55, 55), 1)

    origin = project(np.zeros((1, 2), dtype=np.float32))[0]
    cv2.drawMarker(
        panel,
        tuple(origin),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        16,
        2,
        cv2.LINE_AA,
    )

    def trajectory(points: np.ndarray, color: tuple[int, int, int]) -> None:
        pixels = project(points)
        if len(pixels) > 1:
            cv2.polylines(panel, [pixels], False, color, 3, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            cv2.circle(panel, tuple(point), 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                str(index),
                tuple(point + np.asarray((5, -5))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )

    trajectory(expert, (0, 180, 255))
    trajectory(predicted, (0, 255, 0))
    angles = np.linspace(0.0, 2.0 * math.pi, 65)
    eigenvalues, eigenvectors = np.linalg.eigh(uwb_covariance)
    transform = eigenvectors @ np.diag(2.0 * np.sqrt(np.maximum(eigenvalues, 0.0)))
    ellipse = uwb_xy[None] + np.stack((np.cos(angles), np.sin(angles)), axis=1) @ transform.T
    cv2.polylines(panel, [project(ellipse)], False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(panel, tuple(project(uwb_xy.reshape(1, 2))[0]), 5, (255, 255, 0), -1)
    _caption(panel, "Local base-frame trajectory: x forward / y left")
    cv2.putText(panel, prediction_label, (80, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "EXPERT", (265, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "UWB 2-sigma", (390, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return panel


def _diagnostics_panel(
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    _caption(panel, "Audit metadata and raw predictions")
    stop_probability = float(torch.sigmoid(outputs["stop_logit"][0, 0]).cpu())
    xi = _numpy(outputs["xi_hat"][0])
    yaw = math.atan2(float(xi[2]), float(xi[3]))
    gamma = float(outputs["uwb_gamma"][0].detach().cpu())
    rho_fov = float(outputs["uwb_rho_fov"][0].detach().cpu())
    trained = str(metadata.get("policy_state", "untrained")).lower() == "trained"
    policy_label = "trained" if trained else "untrained"
    status_line = (
        f"TRAINED POLICY CHECKPOINT - STEP {int(metadata.get('checkpoint_step', 0))}"
        if trained
        else "UNTRAINED POLICY HEADS - QUALITATIVE WIRING CHECK ONLY"
    )
    prediction_description = "trained policy prediction" if trained else "random policy prediction"
    lines = [
        status_line,
        f"backend: {metadata.get('backend', 'unknown')}",
        f"episode: {metadata.get('episode', 'unknown')}",
        f"frames: init={metadata.get('initial_index')} anchor={metadata.get('anchor_index')}",
        "input: raw RGB + one initialization bbox + simulated UWB + masks",
        "perception cache: false | later bbox: false | test_locked: false",
        f"DA3 loading coverage: {100.0 * float(metadata.get('da3_coverage', 0.0)):.2f}%",
        f"waypoint SmoothL1 (diagnostic): {float(metadata.get('waypoint_loss', 0.0)):.6f}",
        f"stop probability ({policy_label}): {stop_probability:.4f}",
        f"SE(2) bottleneck raw: dx={xi[0]:+.4f} dy={xi[1]:+.4f}",
        f"SE(2) bottleneck yaw={yaw:+.4f} rad (sin={xi[2]:+.4f}, cos={xi[3]:+.4f})",
        f"UWB base xy: {_numpy(batch['uwb_xy'][0]).round(4).tolist()}",
        f"UWB image gamma={gamma:.4f} | rho_fov={rho_fov:.4f}",
        "Target heatmap: white cross=center of attention; cyan ellipse=UWB 2-sigma",
        f"Waypoint colors: green={prediction_description}; orange=expert; cyan=UWB",
    ]
    y = 50
    for line_index, line in enumerate(lines):
        color = (40, 180, 255) if line_index == 0 else (225, 225, 225)
        for wrapped in textwrap.wrap(line, width=82) or [""]:
            cv2.putText(
                panel,
                wrapped,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 21
    table_y = max(y + 4, 356)
    cv2.putText(panel, " idx | prediction [forward,left] | expert [forward,left]", (16, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (170, 170, 170), 1, cv2.LINE_AA)
    predicted = _numpy(outputs["waypoints"][0])
    expert = _numpy(batch["target_waypoints"][0])
    for index, (prediction, target) in enumerate(zip(predicted, expert)):
        row = f"  {index}  | [{prediction[0]:+7.3f},{prediction[1]:+7.3f}]       | [{target[0]:+7.3f},{target[1]:+7.3f}]"
        cv2.putText(panel, row, (16, table_y + 17 * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 215, 215), 1, cv2.LINE_AA)
    return panel


def render_pretrain_dashboard(
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    grid_shape: tuple[int, int] = (20, 36),
) -> np.ndarray:
    """Render one deterministic dashboard without consuming training-only labels as input."""

    trained = str(metadata.get("policy_state", "untrained")).lower() == "trained"
    title = (
        "Architecture v1 trained checkpoint visual inspection"
        if trained
        else "Architecture v1 pre-training visual inspection"
    )
    temporal_fusion = str(metadata.get("temporal_fusion", "gru"))
    temporal_label = "GRU" if temporal_fusion == "gru" else "single-step fusion"
    subtitle = (
        f"DA3 pretrained + trained adapters/fusion/{temporal_label}/policy heads at step {int(metadata.get('checkpoint_step', 0))}"
        if trained
        else f"DA3 is pretrained; adapters, fusion, {temporal_label} and policy heads are at initial weights"
    )
    prediction_label = "PRED (trained)" if trained else "PRED (untrained)"

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 16, dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (40, 180, 255),
        1,
        cv2.LINE_AA,
    )
    canvas[70:250] = _history_strip(batch)

    current = cv2.resize(
        _bgr_image(batch["ego_rgb"][0, -1]),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    target_attention = outputs["target_attention"][0]
    scene_attention = outputs["scene_attention"][0]
    uwb_probability = torch.softmax(outputs["uwb_patch_bias"][0], dim=0)
    panels = [
        _overlay_attention(current, target_attention, grid_shape),
        _overlay_attention(current, uwb_probability, grid_shape),
        _overlay_attention(current, scene_attention, grid_shape),
    ]
    _draw_attention_center(panels[0], target_attention, grid_shape)
    mean_uv = _numpy(outputs["uwb_mean_uv"][0])
    covariance_uv = _numpy(outputs["uwb_covariance_uv"][0])
    projectable = bool(outputs["uwb_projectable"][0].detach().cpu())
    _draw_projected_uwb(panels[0], mean_uv, covariance_uv, projectable)
    _draw_projected_uwb(panels[1], mean_uv, covariance_uv, projectable)
    _draw_attention_center(panels[2], scene_attention, grid_shape)
    captions = (
        "Target Cross-Attention + UWB bias",
        "Camera-geometry UWB Gaussian patch prior",
        "Scene Attention",
    )
    for index, (panel, caption) in enumerate(zip(panels, captions)):
        _caption(panel, caption)
        left = index * IMAGE_WIDTH
        canvas[270 : 270 + IMAGE_HEIGHT, left : left + IMAGE_WIDTH] = panel

    waypoint_width = CANVAS_WIDTH // 2
    waypoint = _waypoint_panel(
        _numpy(outputs["waypoints"][0]),
        _numpy(batch["target_waypoints"][0]),
        _numpy(batch["uwb_xy"][0]),
        _numpy(batch["uwb_covariance_xy"][0]),
        width=waypoint_width,
        height=520,
        prediction_label=prediction_label,
    )
    diagnostics = _diagnostics_panel(
        batch,
        outputs,
        metadata,
        width=CANVAS_WIDTH - waypoint_width,
        height=520,
    )
    canvas[570:1090, :waypoint_width] = waypoint
    canvas[570:1090, waypoint_width:] = diagnostics
    return canvas

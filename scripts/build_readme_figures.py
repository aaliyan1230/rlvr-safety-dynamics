#!/usr/bin/env python3
"""Build the data-grounded SVG figures embedded in the project README."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"

INK = "#172033"
MUTED = "#5f6b7a"
GRID = "#d8dee8"
PANEL = "#ffffff"
BACKGROUND = "#f7f9fc"
RISK = "#d97706"
RISK_SOFT = "#fef3c7"
CAPABILITY = "#2563eb"
MEASUREMENT = "#0f766e"
BEHAVIOR = "#7c3aed"
THRESHOLD = "#94a3b8"


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    fill: str = INK,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        'font-family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, '
        f'Segoe UI, sans-serif">{value}</text>'
    )


def line_path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}"
        for index, (x, y) in enumerate(points)
    )


def build_trajectory_figure() -> str:
    structured = load_json("artifacts/tulu_structured_trajectory_v1/metrics.json")
    cross_format = load_json("artifacts/tulu_cross_format_v1/analysis.json")

    checkpoints = structured["checkpoints"]
    capability = cross_format["capability"]["checkpoints"]
    steps = [row["step"] for row in checkpoints]
    baseline = checkpoints[0]["risk"]["estimate"]
    risk_deltas = [row["risk"]["estimate"] - baseline for row in checkpoints]
    accuracies = [row["accuracy"] for row in capability]

    width, height = 1200, 690
    left, right = 110, 1140
    plot_width = right - left
    step_max = max(steps)

    def x(step: int) -> float:
        return left + plot_width * step / step_max

    top_y, top_h = 165, 190
    bottom_y, bottom_h = 445, 155

    def risk_y(value: float) -> float:
        return top_y + top_h * (0.10 - value) / 0.20

    def capability_y(value: float) -> float:
        return bottom_y + bottom_h * (0.8 - value) / 0.6

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="trajectory-title trajectory-desc">',
        '<title id="trajectory-title">Safety-risk and capability trajectories across Tülu GRPO training</title>',
        '<desc id="trajectory-desc">The structured safety-risk change remains inside the plus or minus 0.10 practical band at all twelve checkpoints, while capability accuracy rises from 36.7 percent at baseline to 63.3 percent at step 2440.</desc>',
        f'<rect width="{width}" height="{height}" rx="24" fill="{BACKGROUND}"/>',
        text(60, 58, "What changed across the GRPO trajectory?", size=28, weight=500),
        text(
            60,
            91,
            "Behavior stayed inside the predeclared practical band while the capability panel improved.",
            size=16,
            fill=MUTED,
        ),
        text(left, 138, "Structured safety risk · change from DPO baseline", size=18, weight=500),
        text(right, 138, "12 checkpoints · 6,912 responses", size=14, fill=MUTED, anchor="end"),
        f'<rect x="{left}" y="{top_y}" width="{plot_width}" height="{top_h}" rx="12" fill="{PANEL}"/>',
        f'<rect x="{left}" y="{risk_y(0.10):.1f}" width="{plot_width}" '
        f'height="{risk_y(-0.10)-risk_y(0.10):.1f}" rx="12" fill="{RISK_SOFT}" opacity="0.72"/>',
    ]

    for tick in (-0.10, -0.05, 0.00, 0.05, 0.10):
        y = risk_y(tick)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>',
                text(left - 16, y + 5, f"{tick:+.2f}", size=13, fill=MUTED, anchor="end"),
            ]
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{risk_y(0):.1f}" x2="{right}" y2="{risk_y(0):.1f}" stroke="{INK}" stroke-width="1.5"/>',
            text(right - 8, risk_y(0.10) + 20, "+0.10 practical threshold", size=13, fill=MUTED, anchor="end"),
            text(right - 8, risk_y(-0.10) - 10, "−0.10 practical threshold", size=13, fill=MUTED, anchor="end"),
        ]
    )

    risk_points = [(x(step), risk_y(delta)) for step, delta in zip(steps, risk_deltas)]
    parts.append(
        f'<path d="{line_path(risk_points)}" fill="none" stroke="{RISK}" stroke-width="4" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for point_x, point_y in risk_points:
        parts.append(
            f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="5" fill="{PANEL}" '
            f'stroke="{RISK}" stroke-width="3"/>'
        )
    max_index = max(range(len(risk_deltas)), key=risk_deltas.__getitem__)
    parts.extend(
        [
            text(risk_points[max_index][0] - 12, risk_points[max_index][1] + 25, "+0.057 max", size=14, fill=RISK, weight=500, anchor="end"),
            text(left, 416, "Capability panel · exact-match accuracy", size=18, weight=500),
            text(right, 416, "30 authored items per checkpoint", size=14, fill=MUTED, anchor="end"),
            f'<rect x="{left}" y="{bottom_y}" width="{plot_width}" height="{bottom_h}" rx="12" fill="{PANEL}"/>',
        ]
    )

    for tick in (0.2, 0.4, 0.6, 0.8):
        y = capability_y(tick)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>',
                text(left - 16, y + 5, f"{tick:.1f}", size=13, fill=MUTED, anchor="end"),
            ]
        )

    capability_points = [(x(step), capability_y(value)) for step, value in zip(steps, accuracies)]
    parts.append(
        f'<path d="{line_path(capability_points)}" fill="none" stroke="{CAPABILITY}" stroke-width="4" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for point_x, point_y in capability_points:
        parts.append(
            f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="5" fill="{PANEL}" '
            f'stroke="{CAPABILITY}" stroke-width="3"/>'
        )
    parts.extend(
        [
            text(capability_points[0][0] + 8, capability_points[0][1] - 16, "36.7%", size=14, fill=CAPABILITY, weight=500),
            text(capability_points[-1][0], capability_points[-1][1] - 16, "63.3%", size=14, fill=CAPABILITY, weight=500, anchor="end"),
        ]
    )

    for step in (0, 320, 960, 1600, 1920, 2440):
        parts.extend(
            [
                f'<line x1="{x(step):.1f}" y1="{bottom_y + bottom_h}" x2="{x(step):.1f}" y2="{bottom_y + bottom_h + 8}" stroke="{MUTED}"/>',
                text(x(step), bottom_y + bottom_h + 30, f"{step:,}", size=13, fill=MUTED, anchor="middle"),
            ]
        )
    parts.extend(
        [
            text((left + right) / 2, 660, "GRPO training step", size=14, fill=MUTED, anchor="middle"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def build_measurement_transition_figure() -> str:
    refinement = load_json("artifacts/tulu_trajectory_refinement_v1/metrics.json")
    event = next(
        contrast
        for interval in refinement["intervals"]
        for contrast in interval["adjacent_contrasts"]
        if contrast["first_step"] == 2080 and contrast["second_step"] == 2120
    )
    rows = [
        (
            "Permutation invariance",
            event["measurement"]["permutation_invariance"]["estimate"],
            event["measurement"]["permutation_invariance"]["simultaneous_ci_low"],
            event["measurement"]["permutation_invariance"]["simultaneous_ci_high"],
            MEASUREMENT,
        ),
        (
            "Marginalized safety risk",
            event["behavior"]["estimate"],
            event["behavior"]["ci_low"],
            event["behavior"]["ci_high"],
            BEHAVIOR,
        ),
        (
            "Answer-order range",
            event["measurement"]["order_range"]["estimate"],
            event["measurement"]["order_range"]["simultaneous_ci_low"],
            event["measurement"]["order_range"]["simultaneous_ci_high"],
            MEASUREMENT,
        ),
    ]

    width, height = 1200, 490
    plot_left, plot_right = 350, 1120
    center = (plot_left + plot_right) / 2
    domain = 0.30

    def x(value: float) -> float:
        return center + (plot_right - plot_left) * value / (2 * domain)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="transition-title transition-desc">',
        '<title id="transition-title">Localized measurement drift between steps 2080 and 2120</title>',
        '<desc id="transition-desc">Permutation invariance falls by 0.135 and answer-order range rises by 0.146, both beyond the 0.10 magnitude threshold, while marginalized safety risk rises by only 0.028.</desc>',
        f'<rect width="{width}" height="{height}" rx="24" fill="{BACKGROUND}"/>',
        text(60, 58, "The clearest transition was measurement drift", size=28, weight=500),
        text(
            60,
            91,
            "Step 2,080 → 2,120 · bars show change; whiskers show 95% intervals",
            size=16,
            fill=MUTED,
        ),
    ]

    plot_top, plot_bottom = 140, 380
    for tick in (-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30):
        tick_x = x(tick)
        stroke = INK if tick == 0 else GRID
        width_value = 1.5 if tick == 0 else 1
        parts.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="{plot_top}" x2="{tick_x:.1f}" y2="{plot_bottom}" stroke="{stroke}" stroke-width="{width_value}"/>',
                text(tick_x, 414, f"{tick:+.2f}", size=13, fill=MUTED, anchor="middle"),
            ]
        )
    for threshold in (-0.10, 0.10):
        threshold_x = x(threshold)
        parts.append(
            f'<line x1="{threshold_x:.1f}" y1="{plot_top}" x2="{threshold_x:.1f}" '
            f'y2="{plot_bottom}" stroke="{THRESHOLD}" stroke-width="2" stroke-dasharray="7 7"/>'
        )
    parts.extend(
        [
            text(x(-0.10), 128, "−0.10", size=13, fill=MUTED, anchor="middle"),
            text(x(0.10), 128, "+0.10", size=13, fill=MUTED, anchor="middle"),
        ]
    )

    for index, (label, estimate, low, high, color) in enumerate(rows):
        row_y = 185 + index * 82
        estimate_x = x(estimate)
        zero_x = x(0)
        bar_x = min(zero_x, estimate_x)
        bar_width = abs(estimate_x - zero_x)
        parts.extend(
            [
                text(320, row_y + 6, label, size=16, weight=500, anchor="end"),
                f'<line x1="{x(low):.1f}" y1="{row_y}" x2="{x(high):.1f}" y2="{row_y}" stroke="{color}" stroke-width="3"/>',
                f'<line x1="{x(low):.1f}" y1="{row_y - 9}" x2="{x(low):.1f}" y2="{row_y + 9}" stroke="{color}" stroke-width="3"/>',
                f'<line x1="{x(high):.1f}" y1="{row_y - 9}" x2="{x(high):.1f}" y2="{row_y + 9}" stroke="{color}" stroke-width="3"/>',
                f'<rect x="{bar_x:.1f}" y="{row_y - 15}" width="{bar_width:.1f}" height="30" rx="6" fill="{color}" opacity="0.88"/>',
                f'<circle cx="{estimate_x:.1f}" cy="{row_y}" r="5" fill="{PANEL}" stroke="{color}" stroke-width="3"/>',
            ]
        )
        label_anchor = "end" if estimate < 0 else "start"
        label_x = estimate_x - 12 if estimate < 0 else estimate_x + 12
        parts.append(
            text(label_x, row_y + 6, f"{estimate:+.3f}", size=15, fill=color, weight=500, anchor=label_anchor)
        )

    parts.extend(
        [
            text(center, 458, "Change in metric", size=14, fill=MUTED, anchor="middle"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "trajectory-overview.svg": build_trajectory_figure(),
        "measurement-transition.svg": build_measurement_transition_figure(),
    }
    for name, content in figures.items():
        output = FIGURE_DIR / name
        output.write_text(content, encoding="utf-8")
        print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

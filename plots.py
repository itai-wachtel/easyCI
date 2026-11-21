"""The charts of easyCI. Each function returns a Plotly figure.

    runs_plot        per-seed mode: every run as a dot
    accuracy_plot    per-example mode: each model's accuracy with its interval
    difference_plot  both modes: the confidence interval of the difference
"""

import plotly.graph_objects as go

# Colours for light and dark mode. Model A is always blue, model B always orange.
COLOURS = {
    "light": {
        "a": "#2a78d6",
        "b": "#eb6834",
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "a": "#3987e5",
        "b": "#d95926",
        "surface": "#0e1117",
        "ink": "#f0efec",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def _style(fig, colours, height):
    """Settings shared by both charts: quiet grid, no background, small margins."""
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=20, t=20, b=60),  # room for the axis labels
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colours["ink"]),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=colours["grid"], linecolor=colours["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=colours["grid"], linecolor=colours["axis"], zeroline=False)
    return fig


def runs_plot(a, b, name_a, name_b, paired, dark=False, decimals=2):
    """One dot per run, one column per model, with a short line at each mean.

    In paired mode a thin grey line joins the two runs that shared a seed.
    """
    colours = COLOURS["dark" if dark else "light"]
    fig = go.Figure()

    if paired:
        for seed, (score_a, score_b) in enumerate(zip(a, b), start=1):
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[score_a, score_b],
                    mode="lines",
                    line=dict(color=colours["muted"], width=1),
                    hoverinfo="skip",
                )
            )

    for position, (scores, name, key) in enumerate([(a, name_a, "a"), (b, name_b, "b")]):
        mean = sum(scores) / len(scores)
        # The mean: a short horizontal line with its value written next to it.
        fig.add_trace(
            go.Scatter(
                x=[position - 0.18, position + 0.18],
                y=[mean, mean],
                mode="lines",
                line=dict(color=colours[key], width=2),
                hoverinfo="skip",
            )
        )
        side = -0.22 if position == 0 else 0.22
        fig.add_annotation(
            x=position + side,
            y=mean,
            text=f"mean {mean:.{decimals}f}",
            showarrow=False,
            xanchor="right" if position == 0 else "left",
            font=dict(color=colours["ink"], size=13),
        )
        # The runs: one dot each, with a ring so overlapping dots stay visible.
        fig.add_trace(
            go.Scatter(
                x=[position] * len(scores),
                y=scores,
                mode="markers",
                marker=dict(
                    color=colours[key],
                    size=11,
                    line=dict(color=colours["surface"], width=2),
                ),
                customdata=list(range(1, len(scores) + 1)),
                hovertemplate=f"{name}, run %{{customdata}}: %{{y}}<extra></extra>",
            )
        )

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=[name_a, name_b],
        range=[-0.8, 1.8],
        showgrid=False,
        tickfont=dict(size=14),
    )
    fig.update_yaxes(title_text="Score")
    return _style(fig, colours, height=380)


def accuracy_plot(result, name_a, name_b, dark=False):
    """Each model's accuracy as a dot, with its Wilson interval as a vertical bar."""
    colours = COLOURS["dark" if dark else "light"]
    percent = round(result.confidence * 100)
    fig = go.Figure()

    models = [
        (result.accuracy_a, result.interval_a, name_a, "a"),
        (result.accuracy_b, result.interval_b, name_b, "b"),
    ]
    for position, (accuracy, (low, high), name, key) in enumerate(models):
        fig.add_trace(
            go.Scatter(
                x=[position, position],
                y=[low, high],
                mode="lines",
                line=dict(color=colours[key], width=3),
                hovertemplate=(
                    f"{name}, {percent}% interval: {low:.1f}% to {high:.1f}%<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[position],
                y=[accuracy],
                mode="markers",
                marker=dict(
                    color=colours[key], size=13, line=dict(color=colours["surface"], width=2)
                ),
                hovertemplate=f"{name}: {accuracy:.1f}%<extra></extra>",
            )
        )
        fig.add_annotation(
            x=position + 0.08,
            y=accuracy,
            text=f"{accuracy:.1f}%",
            showarrow=False,
            xanchor="left",
            font=dict(color=colours["ink"], size=13),
        )

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=[name_a, name_b],
        range=[-0.8, 1.8],
        showgrid=False,
        tickfont=dict(size=14),
    )
    # Leave some room above and below the two bars.
    lowest = min(result.interval_a[0], result.interval_b[0])
    highest = max(result.interval_a[1], result.interval_b[1])
    room = (highest - lowest) * 0.1 or 1
    fig.update_yaxes(title_text="Accuracy (%)", range=[lowest - room, highest + room])
    return _style(fig, colours, height=340)


def difference_plot(result, name_a, name_b, dark=False, unit="", decimals=2):
    """The confidence interval of the difference as a bar, with a line at zero.

    If the bar does not touch the zero line, "no difference" is not plausible.
    `unit` is added to the axis title, for example " (percentage points)".
    """
    colours = COLOURS["dark" if dark else "light"]
    percent = round(result.confidence * 100)
    fig = go.Figure()

    # The difference belongs to neither model, so it is drawn in the text
    # colour rather than in model A's blue or model B's orange.
    fig.add_trace(
        go.Scatter(
            x=[result.ci_low, result.ci_high],
            y=[0, 0],
            mode="lines",
            line=dict(color=colours["ink"], width=4),
            hovertemplate=(
                f"{percent}% confidence interval: "
                f"{result.ci_low:.{decimals}f} to {result.ci_high:.{decimals}f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.difference],
            y=[0],
            mode="markers",
            marker=dict(color=colours["ink"], size=13, line=dict(color=colours["surface"], width=2)),
            hovertemplate=(
                f"Observed difference: {result.difference:.{decimals}f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color=colours["muted"], width=1.5, dash="dash"))
    fig.add_annotation(
        x=0, y=1, yref="paper", text="no difference", showarrow=False,
        yanchor="bottom", font=dict(color=colours["muted"], size=12),
    )

    # Always keep zero in view, with some room on both sides.
    left = min(result.ci_low, 0)
    right = max(result.ci_high, 0)
    room = (right - left) * 0.15 or 1
    fig.update_xaxes(
        title_text=f"{name_b} minus {name_a}{unit}", range=[left - room, right + room]
    )
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig = _style(fig, colours, height=200)
    fig.update_layout(margin=dict(t=30))
    return fig

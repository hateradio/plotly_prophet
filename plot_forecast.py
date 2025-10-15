# -*- coding: utf-8 -*-
"""
Simple, test‑friendly Plotly helper for Prophet forecasts with fully
configurable colour palette.

The module keeps the public API tiny – one function plus optional keyword
arguments – while removing dead code, unnecessary imports and long
duplicated logic.  All traces are built in a vectorised way and added in
bulk – no recursion, no “look‑around” for extra traces.

Typical usage:

    import pandas as pd
    from prophet import Prophet
    from plot_forecast import plot_forecast

    m = Prophet()
    m.fit(df)
    fcst = m.predict(df[['ds']])
    fig = plot_forecast(m, fcst, colors={"forecast": "#2a9d8f"})
    fig.show()
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional, Union

# --------------------------------------------------------------------------- #
#  Default colour palette (kept for backward compatibility)
# --------------------------------------------------------------------------- #
DEFAULT_PALETTE: Dict[str, Union[str, float]] = {
    "forecast":    "#0D47A1",          # Prophet’s mean line
    "observed":    "#FF6F00",          # Scatter markers
    "cap":         "#000000",          # Dashed capacity lines
    "uncertainty": "rgba(0, 114, 178, 0.2)",  # Confidence band fill
}

# --------------------------------------------------------------------------- #
#  Helper functions – each < 20 lines, accept a palette dict
# --------------------------------------------------------------------------- #
def _add_forecast(fig: go.Figure, fcst: pd.DataFrame, palette: Dict[str, str]) -> None:
    fig.add_trace(
        go.Scatter(
            x=fcst["ds"],
            y=fcst["yhat"],
            mode="lines",
            line=dict(color=palette["forecast"], width=2),
            name="Forecast",
        )
    )


def _add_observed(fig: go.Figure, m, palette: Dict[str, str]) -> None:
    if hasattr(m, "history"):
        fig.add_trace(
            go.Scatter(
                x=m.history["ds"],
                y=m.history["y"],
                mode="markers",
                marker=dict(color=palette["observed"], size=4, opacity=1),
                name="Observed data points",
            )
        )


def _add_capacity(fig: go.Figure, m, fcst: pd.DataFrame, palette: Dict[str, str]) -> None:
    if "cap" in fcst.columns:
        fig.add_trace(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["cap"],
                mode="lines",
                line=dict(dash="dash", color=palette["cap"]),
                name="Maximum capacity",
            )
        )
    if getattr(m, "logistic_floor", False) and "floor" in fcst.columns:
        fig.add_trace(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["floor"],
                mode="lines",
                line=dict(dash="dash", color=palette["cap"]),
                name="Minimum capacity",
            )
        )


def _add_uncertainty(fig: go.Figure, m, fcst: pd.DataFrame, palette: Dict[str, str]) -> None:
    if getattr(m, "uncertainty_samples", 0) > 0:
        # Upper bound – placeholder for the lower trace (no visible line)
        fig.add_trace(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Lower bound – fill to the previous (upper) trace
        fig.add_trace(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=palette["uncertainty"],
                hoverinfo="skip",
                name="Uncertainty interval",
            )
        )

# --------------------------------------------------------------------------- #
#  Public API – single function you call
# --------------------------------------------------------------------------- #
def plot_forecast(
    m,
    fcst: pd.DataFrame,
    *,
    fig: Optional[go.Figure] = None,
    uncertainty: bool = True,
    plot_cap: bool = True,
    xlabel: str = "ds",
    ylabel: str = "y",
    width: int = 800,
    height: int = 600,
    include_legend: bool = False,
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Plot a Prophet forecast in a clean, vectorised way with an optional
    custom colour palette.

    Parameters
    ----------
    m : prophet.forecaster.Prophet
        Trained Prophet model – only its ``history`` and attributes are used.
    fcst : pd.DataFrame
        DataFrame returned by ``m.predict``.
    fig : plotly.graph_objects.Figure | None, optional
        Existing figure to add to.  If ``None`` a new figure is created.
    uncertainty : bool, default=True
        Plot confidence intervals when the model was trained with samples.
    plot_cap : bool, default=True
        Draw capacity / floor lines when available.
    xlabel, ylabel : str
        Axis labels.
    width, height : int
        Figure dimensions in pixels.
    include_legend : bool
        Show a legend (default is hidden to keep the plot clean).
    colors : dict[str, str] | None, optional
        Custom hex/rgba values for any of the keys:
        ``forecast``, ``observed``, ``cap``, ``uncertainty``.
        They override the default palette.

    Returns
    -------
    plotly.graph_objects.Figure
        The fully configured figure.
    """
    # Ensure we work with a copy of the forecast
    fcst = fcst.copy()

    # Resolve palette: default values + user overrides
    palette = DEFAULT_PALETTE.copy()
    if colors:
        palette.update(colors)

    # Create or reuse the figure
    fig = fig or go.Figure()

    # Build and add traces
    _add_forecast(fig, fcst, palette)
    _add_observed(fig, m, palette)

    if plot_cap:
        _add_capacity(fig, m, fcst, palette)

    if uncertainty:
        _add_uncertainty(fig, m, fcst, palette)

    # Layout – keep it simple and consistent
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text="", x=0.5),
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        hovermode="x unified",
    )

    # Legend handling
    if include_legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    else:
        fig.update_layout(showlegend=False)

    return fig

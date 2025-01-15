import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from typing import List
from pylit.backend.core.experiment import Experiment
from pylit.global_settings import (
    FLOAT_DTYPE,
    COLOR_F,
    COLOR_F_SHADED,
    COLOR_D,
    COLOR_D_SHADED,
    COLOR_S,
    COLOR_S_SHADED,
    ARRAY,
    DPI,
    FIGSIZE,
)

# --- Default Model Plotting --- #


def plot_default_model(exp: Experiment) -> go.Figure:
    _matplot_default_model(exp)
    return _plotly_default_model(exp)


def _matplot_default_model(exp: Experiment) -> None:
    # Create a new figure and axis
    _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot D
    ax.plot(exp.prep.omega, exp.prep.D, color=COLOR_D, label="D(ω)")

    # Plot the expected value of D
    ax.axvline(x=exp.prep.expD, color=COLOR_D, linestyle="--", label="𝔼(ω)")

    # Adjust layout
    ax.set_title("ω-Space")
    ax.set_xlabel("ω")
    ax.set_ylabel("D(ω)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Store matplot
    plt.savefig(
        os.path.join(exp.plots_default_model_png_directory, "default-model.png")
    )


def _plotly_default_model(exp: Experiment) -> go.Figure:
    # Create a plotly figure
    fig = go.Figure()

    # Plot D on the right subplot
    fig.add_trace(
        go.Scatter(
            x=exp.prep.omega,
            y=exp.prep.D,
            mode="lines",
            name=f"D(ω)",
            line=dict(color=COLOR_D),
        ),
    )

    # Plot the expected value of D as a red line
    fig.add_trace(
        go.Scatter(
            x=[exp.prep.expD, exp.prep.expD],
            y=[0, np.max(exp.prep.D)],
            mode="lines",
            name="𝔼(ω)",
            line=dict(color=COLOR_D, dash="dash"),
        ),
    )

    # Calculate the bounds for the standard deviation area
    omega_lower_bound = exp.prep.expD - exp.prep.stdD
    omega_upper_bound = exp.prep.expD + exp.prep.stdD

    # Get all omega values between the lower and upper bounds
    mask = (exp.prep.omega >= omega_lower_bound) & (exp.prep.omega <= omega_upper_bound)
    omega_within_bounds = exp.prep.omega[mask]
    D_within_bounds = exp.prep.D[mask]

    # Create the shaded area for the standard deviation
    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [[omega_lower_bound], omega_within_bounds, [omega_upper_bound]]
            ),
            y=np.concatenate([[0], D_within_bounds, [0]]),
            fill="toself",
            fillcolor=COLOR_D_SHADED,
            line=dict(color="rgba(0, 0, 0, 0)"),
            name="Standard Deviation",
            showlegend=True,
        ),
    )

    # Update layout for figure
    fig.update_layout(
        title="",
        xaxis=dict(title="ω"),
        yaxis=dict(title="ω-space"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Store the plotly-plot
    fig.write_html(
        os.path.join(exp.plots_default_model_html_directory, "default-model.html")
    )

    return fig


def plot_forward_default_model(exp: Experiment) -> go.Figure:
    F_upper, F_lower = None, None
    if exp.prep.noiseF is not None and len(exp.prep.noiseF) > 0:
        eps_upper = np.amax(exp.prep.noiseF, axis=0)
        eps_lower = np.amin(exp.prep.noiseF, axis=0)
        F_upper = exp.prep.F + eps_upper
        F_lower = exp.prep.F + eps_lower

    _matplot_forward_default_model(exp, F_upper, F_lower)
    return _plotly_forward_default_model(exp, F_upper, F_lower)


def _matplot_forward_default_model(
    exp: Experiment, F_upper: ARRAY, F_lower: ARRAY
) -> None:
    # Create a new figure and axis
    _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot F
    if F_upper is not None and F_lower is not None:
        ax.fill_between(
            exp.prep.tau, F_lower, F_upper, color=COLOR_F, label="F+ε(τ)", alpha=0.5
        )
    ax.plot(exp.prep.tau, exp.prep.F, color=COLOR_F, label="F(τ)")

    # Plot L(D)
    ax.plot(
        exp.prep.tau, exp.prep.forwardD, color=COLOR_D, linestyle="--", label="L[D](τ)"
    )

    # Adjust layout
    ax.set_title("τ-Space")
    ax.set_xlabel("τ")
    ax.set_ylabel("F(τ)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Store matplot
    plt.savefig(
        os.path.join(exp.plots_default_model_png_directory, "forward-default-model.png")
    )


def _plotly_forward_default_model(
    exp: Experiment, F_upper: ARRAY, F_lower: ARRAY
) -> go.Figure:
    # Create a plotly figure
    fig = go.Figure()

    # Add the shaded area for F±ε(τ)
    if F_upper is not None and F_lower is not None:
        fig.add_trace(
            go.Scatter(
                x=exp.prep.tau,
                y=F_upper,
                mode="lines",
                name="F+ε₊(τ)",
                line=dict(width=0.5, color=COLOR_F),
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=exp.prep.tau,
                y=F_lower,
                fill="tonexty",
                mode="lines",
                name="F+ε₋(τ)",
                line=dict(width=0.5, color=COLOR_F),
                fillcolor=COLOR_F_SHADED,
            ),
        )

    # Plot F on the left subplot
    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=exp.prep.F,
            mode="lines",
            name=f"F(τ)",
            line=dict(
                color=COLOR_F,
            ),
        ),
    )

    # Plot L(D) on the left subplot
    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=exp.prep.forwardD,
            mode="lines",
            name=f"L[D](τ)",
            line=dict(
                color=COLOR_D,
                dash="dash",
            ),
        ),
    )

    # Update layout for the figure
    fig.update_layout(
        title="",
        xaxis=dict(title="τ"),
        yaxis=dict(title="τ-space"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Store the plotly-plot
    fig.write_html(
        os.path.join(
            exp.plots_default_model_html_directory, "forward-default-model.html"
        )
    )

    return fig


def plot_forward_default_model_error(exp: Experiment) -> go.Figure:
    _matplot_forward_default_model_error(exp)
    return _plotly_forward_default_model_error(exp)


def _matplot_forward_default_model_error(exp: Experiment) -> None:
    # Create a new figure and axis
    _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot forwardDAbsError
    ax.plot(
        exp.prep.tau, exp.prep.forwardDAbsError, color=COLOR_D, label="|L[D] - F|(τ)"
    )

    # Adjust layout
    ax.set_title("")
    ax.set_xlabel("τ")
    ax.set_ylabel("τ-Space")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Store matplot
    plt.savefig(
        os.path.join(
            exp.plots_default_model_png_directory, "forward-default-model-error.png"
        )
    )


def _plotly_forward_default_model_error(exp: Experiment) -> go.Figure:
    # Create a plotly figure
    fig = go.Figure()

    # Plot forwardDAbsError
    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=exp.prep.forwardDAbsError,
            mode="lines",
            name=f"|L[D] - F|(τ)",
            line=dict(color=COLOR_D),
        ),
    )

    # Update layout for the figure
    fig.update_layout(
        title="",
        xaxis=dict(title="τ"),
        yaxis=dict(title="τ-space", tickformat=".2e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Store the plotly-plot
    fig.write_html(
        os.path.join(
            exp.plots_default_model_html_directory, "forward-default-model-error.html"
        )
    )

    return fig


# --- Model Plotting --- #


def plot_results(
    exp: Experiment,
    coefficients: bool,
    model: bool,
    forward_model: bool,
    forward_model_error: bool,
):
    if coefficients:
        plot_coeffs(exp)
    if model:
        plot_model(exp)
    if forward_model:
        plot_forward_model(exp)
    if forward_model_error:
        plot_forward_model_error(exp)


def plot_coeffs(exp: Experiment) -> go.Figure:
    if exp.output.coefficients is None:
        return None
    _matplot_coeffs(exp)
    return _plotly_coeffs(exp)


def _matplot_coeffs(exp: Experiment) -> None:
    # Create a matplot figure
    fig, ax = plt.subplots()

    # Display the coefficients array as an image
    cax = ax.imshow(exp.output.coefficients, aspect="auto", cmap="viridis")

    # Add a color bar
    fig.colorbar(cax)

    # Add labels and title
    ax.set_title("")
    ax.set_xlabel("Coefficient Index")
    ax.set_ylabel("Model Index")

    # Adjust layout
    plt.tight_layout()

    # Store matplot-plot
    plt.savefig(os.path.join(exp.plots_model_png_directory, "coefficients.png"))


def _plotly_coeffs(exp: Experiment) -> go.Figure:
    # Create a plotly heatmap
    heatmap = go.Heatmap(z=exp.output.coefficients, colorscale="Viridis")

    # Create a plotly figure
    fig = go.Figure(data=[heatmap])

    # Add labels and title
    fig.update_layout(
        title="",
        xaxis_title="Coefficient Index",
        yaxis_title="Model Index",
    )

    # Save the plot as an HTML file
    fig.write_html(os.path.join(exp.plots_model_html_directory, "coefficients.html"))

    return fig


def plot_model(exp: Experiment) -> go.Figure:
    S = exp.output.S

    if exp.prep.D is None or S is None:
        return None

    min_S = np.max(S, axis=0)
    max_S = np.min(S, axis=0)
    avg_S = np.mean(S, axis=0)
    avg_expS = np.mean(exp.output.expS)

    _matplot_model(exp, max_S, min_S, avg_S)
    return _plotly_model(exp, max_S, min_S, avg_S, avg_expS)


def _matplot_model(exp: Experiment, max_S: ARRAY, min_S: ARRAY, avg_S: ARRAY) -> None:
    # Create a matplot figure
    _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Plot S
    ax.plot(exp.prep.omega, exp.prep.D, label="D(ω)", color=COLOR_D)

    # Plot Models
    ax.plot(exp.prep.omega, min_S, label="min S(ω)", color=COLOR_S)
    ax.plot(exp.prep.omega, max_S, label="max S(ω)", color=COLOR_S)
    ax.plot(exp.prep.omega, avg_S, label="avg S(ω)", color=COLOR_S)

    # Fill between min and max values
    ax.fill_between(exp.prep.omega, min_S, max_S, color=COLOR_S, alpha=0.3)

    # Adjust layout
    ax.set_title("")
    ax.set_xlabel("ω")
    ax.set_ylabel("ω-Space")
    ax.legend(loc="upper right")
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(os.path.join(exp.plots_model_png_directory, "model.png"))


def _plotly_model(
    exp: Experiment, max_S: ARRAY, min_S: ARRAY, avg_S: ARRAY, avg_expS: FLOAT_DTYPE
) -> go.Figure:
    # Create a plotly figure
    fig = go.Figure()

    # Plot S
    fig.add_trace(
        go.Scatter(
            x=exp.prep.omega,
            y=exp.prep.D,
            mode="lines",
            name=f"D(ω)",
            line=dict(color=COLOR_D),
        )
    )

    # Plot the expected value of D
    fig.add_trace(
        go.Scatter(
            x=[exp.prep.expD, exp.prep.expD],
            y=[0, np.max(exp.prep.D)],
            mode="lines",
            name="𝔼(ω)",
            line=dict(color=COLOR_D, dash="dash"),
            visible="legendonly",
        ),
    )

    # Plot Models
    fig.add_trace(
        go.Scatter(
            x=exp.prep.omega,
            y=min_S,
            mode="lines",
            name=f"min S(ω)",
            line=dict(width=0.5, color=COLOR_S),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp.prep.omega,
            y=max_S,
            fill="tonexty",
            mode="lines",
            name=f"max S(ω)",
            line=dict(width=0.5, color=COLOR_S),
            fillcolor=COLOR_S_SHADED,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp.prep.omega,
            y=avg_S,
            mode="lines",
            name=f"avg S(ω)",
            line=dict(color=COLOR_S),
        )
    )

    # Plot the average expected value of S
    fig.add_trace(
        go.Scatter(
            x=[avg_expS, avg_expS],
            y=[0, np.max(avg_S)],
            mode="lines",
            name="avg 𝔼(ω)",
            line=dict(color=COLOR_S, dash="dash"),
            visible="legendonly",
        ),
    )

    # Update layout for figure
    fig.update_layout(
        title="",
        xaxis=dict(title="ω"),
        yaxis=dict(title="ω-Space"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Store the plotly
    fig.write_html(os.path.join(exp.plots_model_html_directory, "model.html"))

    return fig


def plot_forward_model(exp: Experiment) -> go.Figure:
    forward_S = exp.output.forwardS

    if exp.prep.F is None or forward_S is None:
        return None

    max_forward_S = np.max(forward_S, axis=0)
    min_forward_S = np.min(forward_S, axis=0)
    avg_forward_S = np.mean(forward_S, axis=0)

    _matplot_forward_model(exp, max_forward_S, min_forward_S, avg_forward_S)
    return _plotly_forward_model(exp, max_forward_S, min_forward_S, avg_forward_S)


def _matplot_forward_model(
    exp: Experiment, max_forward_S: ARRAY, min_forward_S: ARRAY, avg_forward_S: ARRAY
) -> None:
    # Create a matplot figure
    _, ax = plt.subplots()

    # Plot F
    ax.plot(exp.prep.tau, exp.prep.F, label="F(τ)", color=COLOR_F)

    # Plot Forward Models
    ax.plot(exp.prep.tau, min_forward_S, label="min L[S](τ)", color=COLOR_S)
    ax.plot(exp.prep.tau, max_forward_S, label="max L[S](τ)", color=COLOR_S)
    ax.plot(exp.prep.tau, avg_forward_S, label="avg L[S](τ)", color=COLOR_S)

    # Fill between min and max values
    ax.fill_between(
        exp.prep.tau, min_forward_S, max_forward_S, color=COLOR_S, alpha=0.3
    )

    # Adjust layout
    ax.set_title("")
    ax.set_xlabel("τ")
    ax.set_ylabel("τ-Space")
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Save matplot
    plt.savefig(os.path.join(exp.plots_model_png_directory, "forward-model.png"))


def _plotly_forward_model(
    exp: Experiment, max_forward_S: ARRAY, min_forward_S: ARRAY, avg_forward_S: ARRAY
) -> go.Figure:
    # Create a plotly figure
    fig = go.Figure()

    # Plot F
    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=exp.prep.F,
            mode="lines",
            name=f"F(τ)",
        )
    )

    # Plot Forward Models

    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=min_forward_S,
            mode="lines",
            name=f"min L[S](τ)",
            line=dict(color=COLOR_S),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=max_forward_S,
            fill="tonexty",
            mode="lines",
            name=f"max L[S](τ)",
            line=dict(color=COLOR_S),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=avg_forward_S,
            mode="lines",
            name=f"avg L[S](τ)",
            line=dict(color=COLOR_S),
        )
    )

    fig.update_layout(
        title="",
        xaxis=dict(title="τ"),
        yaxis=dict(title="τ-Space"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.write_html(os.path.join(exp.plots_model_html_directory, "forward-model.html"))

    return fig


def plot_forward_model_error(exp: Experiment) -> go.Figure:
    forward_S_abs_error = exp.output.forwardSAbsError

    if exp.prep.F is None or forward_S_abs_error is None:
        return None

    max_forward_S_abs_error = np.max(forward_S_abs_error, axis=0)
    min_forward_S_abs_error = np.min(forward_S_abs_error, axis=0)
    avg_forward_S_abs_error = np.mean(forward_S_abs_error, axis=0)

    _matplot_forward_model_error(
        exp, max_forward_S_abs_error, min_forward_S_abs_error, avg_forward_S_abs_error
    )
    return _plotly_forward_model_error(
        exp, max_forward_S_abs_error, min_forward_S_abs_error, avg_forward_S_abs_error
    )


def _matplot_forward_model_error(
    exp: Experiment,
    max_forward_S_abs_error: ARRAY,
    min_forward_S_abs_error: ARRAY,
    avg_forward_S_abs_error: ARRAY,
) -> None:
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()

    # # Plot Forward Model Error
    ax.plot(
        exp.prep.tau, min_forward_S_abs_error, label="min |L[S]-F|(τ)", color=COLOR_S
    )
    ax.plot(
        exp.prep.tau, max_forward_S_abs_error, label="max |L[S]-F|(τ)", color=COLOR_S
    )
    ax.plot(
        exp.prep.tau, avg_forward_S_abs_error, label="avg |L[S]-F|(τ)", color=COLOR_S
    )

    # Fill between min and max values
    ax.fill_between(
        exp.prep.tau,
        min_forward_S_abs_error,
        max_forward_S_abs_error,
        color=COLOR_S,
        alpha=0.3,
    )

    # Adjust layout
    ax.set_title("")
    ax.set_xlabel("τ")
    ax.set_ylabel("τ-Space")
    ax.legend(loc="upper right")
    plt.tight_layout()

    # Save matplot
    plt.savefig(os.path.join(exp.plots_model_png_directory, "forward-model-error.png"))


def _plotly_forward_model_error(
    exp: Experiment,
    min_forward_S_abs_error: ARRAY,
    max_forward_S_abs_error: ARRAY,
    avg_forward_S_abs_error: ARRAY,
) -> go.Figure:
    # Create a Plotly figure
    fig = go.Figure()

    # Plot Forward Models Error
    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=min_forward_S_abs_error,
            mode="lines",
            name=f"min |L[S]-F|(τ)",
            line=dict(color=COLOR_S),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=max_forward_S_abs_error,
            fill="tonexty",
            mode="lines",
            name=f"max |L[S]-F|(τ)",
            line=dict(color=COLOR_S),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exp.prep.tau,
            y=avg_forward_S_abs_error,
            mode="lines",
            name=f"avg |L[S]-F|(τ)",
            line=dict(color=COLOR_S),
        )
    )

    fig.update_layout(
        title="",
        xaxis=dict(title="τ"),
        yaxis=dict(title="τ-Space"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.write_html(
        os.path.join(exp.plots_model_html_directory, "forward-model-error.html")
    )

    return fig

import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pylit.backend import models, methods, optimize
from pylit.global_settings import ARRAY, FLOAT_DTYPE
from pylit.backend.core import (
    DataLoader,
    Configuration,
    Preparation,
    Output,
    noise_iid,
    noise_conv,
)
from pylit.backend.core.utils import (
    extend_on_negative_axis,
    extend_S,
    exp_std,
    save_to_json,
    load_from_json,
)


class Experiment:

    # TODO
    # These are the attributes that are not part of the config itself but are added by the experiment
    _model_attrs = {"_coeffs", "_grid_points", "_reg_mat"}
    _method_attrs = {"E", "S", "omegas"}

    def __init__(self, name: str, workspace: str = "/", make: bool = False):
        self.name = name
        self.directory = os.path.join(workspace, name)
        self.path_F = os.path.join(self.directory, "F.csv")
        self.path_S = os.path.join(self.directory, "S.csv")
        self.config_path = os.path.join(self.directory, "config.json")
        self.prep_path = os.path.join(self.directory, "prep.json")
        self.output_path = os.path.join(self.directory, "output.json")
        self.plots_dir = os.path.join(self.directory, "plots")

        self._check_directories(make)
        self._check_json_files(make)

        if os.path.isfile(self.output_path):
            self._init_model()
            self._assign_coeffs()
            self._benchmark()
        else:
            self.model = None
            self.method = None
            self.output = None

    @property
    def exported(self):
        return (
            self.prep is not None
            and (
                self.prep.tau is not None
                and self.prep.F is not None
                and self.prep.omega is not None
                and self.prep.S is not None
            )
            and (
                self.prep.tau.shape == self.prep.F.shape
                and self.prep.omega.shape == self.prep.S.shape
            )
            and (self.prep.tau.size > 0 and self.prep.omega.size > 0)
        )

    @property
    def prepared(self):
        return (
            self.prep is not None
            and (
                self.prep.modifiedF is not None
                and self.prep.modifiedOmega is not None
                and self.prep.modifiedS is not None
            )
            and (
                self.prep.modifiedF.size > 0
                and self.prep.modifiedOmega.size > 0
                and self.prep.modifiedS.size > 0
            )
        )

    def _check_directories(self, make: bool):
        if os.path.isdir(self.directory):
            pass
        elif make:
            os.makedirs(self.directory)
        else:
            raise NotADirectoryError(
                f"The directory '{self.directory}' does not exist. Please set 'make' to True."
            )

        # Create plots directory
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

    def _check_json_files(self, make: bool):
        # Check and load config.json
        if os.path.isfile(self.config_path):
            self.config = load_from_json(Configuration, self.config_path)
        elif make:
            self.config = Configuration()
        else:
            raise FileNotFoundError(
                f"The configuration file '{self.config_path}' does not exist. Please set 'make' to True."
            )

        # Check and load prep.json
        if os.path.isfile(self.prep_path):
            self.prep = load_from_json(Preparation, self.prep_path)
        else:
            self.prep = Preparation()

        # Check and load output.json
        if os.path.isfile(self.output_path):
            self.output = load_from_json(Output, self.output_path)
        else:
            self.output = Output()

    def import_F(self):
        # Fetch tau, F from the data file
        dl = DataLoader(self.path_F)
        dl.fetch()
        if dl.data is None:
            raise ValueError("Data file for 'F' is empty.")
        if len(dl.data.shape) != 2:
            raise ValueError("Data file for 'F' must have exactly 2 columns.")
        self.prep.tau, self.prep.F = dl.data.T  # 0 ... tau, 1 ... F
        dl.clear()

    def import_S(self):
        # Fetch omega, S from the data file
        dl = DataLoader(self.path_S)
        dl.fetch()
        if dl.data is None:
            raise ValueError("Data file for 'S' is empty.")
        if len(dl.data.shape) != 2:
            raise ValueError("Data file for 'S' must have exactly 2 columns.")
        self.prep.omega, self.prep.S = dl.data.T  #  0 ... omega, 1 ... S
        dl.clear()

    def prepare(self):
        self._scale_and_noise_F()
        self.prep.tauMin, self.prep.tauMax = np.min(self.prep.tau), np.max(
            self.prep.tau
        )
        self._extend_and_scale()
        self.prep.modifiedOmegaMin, self.prep.modifiedOmegaMax = np.min(
            self.prep.modifiedOmega
        ), np.max(self.prep.modifiedOmega)
        self.prep.expS, self.prep.stdS = exp_std(
            self.prep.modifiedOmega, self.prep.modifiedS
        )
        # Save preparation to JSON
        save_to_json(self.prep, self.prep_path)
        # Also save config (so far) to Json
        save_to_json(self.config, self.config_path)

    def _scale_and_noise_F(self):
        modifiedF = np.copy(self.prep.F)
        if self.config.scaleMaxF:
            modifiedF /= np.max(modifiedF)
        if self.config.noiseActive and self.config.noiseConvActive:
            noise = getattr(noise_iid, self.config.noiseName)(
                *list(self.config.noiseParams.values())
            )(
                np.zeros_like(modifiedF)
            )  # Create noise
            noise = getattr(noise_conv, self.config.noiseConvName)(
                *list(self.config.noiseConvParams.values())
            )(
                noise
            )  # Apply convolution
            modifiedF += noise  # Add noise
        if self.config.noiseActive and not self.config.noiseConvActive:
            # print("Noise and Convolution")
            # print("=====================")
            # print("Noise: ", self.config.noiseName)
            # print("Noise Params: ", self.config.noiseParams)
            modifiedF = getattr(noise_iid, self.config.noiseName)(
                *list(self.config.noiseParams.values())
            )(
                modifiedF
            )  # Noisy F
        self.prep.modifiedF = modifiedF

    def _extend_and_scale(self):
        modifiedOmega = extend_on_negative_axis(self.prep.omega)
        modifiedS = np.copy(self.prep.S)
        if self.config.PosS:
            modifiedS = np.maximum(0, modifiedS)
        if self.config.ExtS:
            modifiedS = extend_S(modifiedS, self.prep.omega, self.prep.tauMax)
        if self.config.trapzS:
            modifiedS /= np.trapz(modifiedS, modifiedOmega)
        self.prep.modifiedOmega = modifiedOmega
        self.prep.modifiedS = modifiedS

    def plot_prep(self):
        self._matplot_prep()
        return self._plotly_prep()

    def _matplot_prep(self):
        # Set figure size and create subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), sharey=False, sharex=False
        )

        # Plot F on the left subplot
        ax1.set_title("$\\tau$-Space")
        ax1.plot(
            self.prep.tau,
            self.prep.modifiedF,
            label=f"$F(\\tau)$",
        )
        ax1.set_xlabel("$\\tau$")
        ax1.set_ylabel("$F(\\tau)$")
        ax1.legend()
        ax1.grid(True)

        # Plot S on the right subplot
        ax2.set_title("$\\omega$-Space")
        ax2.plot(
            self.prep.modifiedOmega,
            self.prep.modifiedS,
            label=f"$S(\\omega)$",
        )
        ax2.set_xlabel("$\\omega$")
        ax2.set_ylabel("$S(\\omega)$")
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(-0.75, 0.75)

        # Adjust layout
        plt.tight_layout()

        # Store matplot-plot
        plt.savefig(os.path.join(self.plots_dir, "prep.png"))

    def _plotly_prep(self):
        # Create a Plotly figure
        fig1 = go.Figure()
        fig2 = go.Figure()
        # Plot F on the left subplot
        fig1.add_trace(
            go.Scatter(
                x=self.prep.tau,
                y=self.prep.modifiedF,
                mode="lines",
                name=f"F(τ)",
            )
        )
        # Plot S on the right subplot
        fig2.add_trace(
            go.Scatter(
                x=self.prep.modifiedOmega,
                y=self.prep.modifiedS,
                mode="lines",
                name=f"S(ω)",
            )
        )
        # Update layout
        fig1.update_layout(
            title="τ-Space",
            xaxis=dict(title="τ"),
            yaxis=dict(title="F(τ)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig2.update_layout(
            title="ω-Space",
            xaxis=dict(title="ω"),
            yaxis=dict(title="S(ω)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        # Store plotly-plots
        fig1.write_html(os.path.join(self.plots_dir, "prep_F.html"))
        fig2.write_html(os.path.join(self.plots_dir, "prep_S.html"))

        return [fig1, fig2]

    def fit(self):
        self._init_model()
        self._add_method_params()
        self._init_method()
        self._optimize()
        self._assign_coeffs()
        self._evaluate()

    def _init_model(self):
        modelParams = {
            key: value
            for key, value in self.config.modelParams.items()
            if key not in self._model_attrs  # TODO _model_attrs is deprecated
        }

        model = getattr(models, self.config.modelName)(**modelParams)

        model = getattr(models.scaling, self.config.scalingName)(
            lrm=model, **self.config.scalingParams
        )

        self.model = model

    def _add_method_params(self):
        if self.model is None:
            raise ValueError(
                "The model must be initiated first. Please use the init_model method"
            )
        # Add Evaluation Matrix
        if "E" in self.config.methodParams:
            self.config.methodParams["E"] = self.model(
                self.prep.modifiedOmega, matrix=True
            )
        # Add Model Matrix
        if "S" in self.config.methodParams:
            self.config.methodParams["S"] = self.prep.modifiedS
        # Add omegas
        if "omegas" in self.config.methodParams:
            self.config.methodParams["omegas"] = self.prep.modifiedOmega

    def _init_method(self):
        methodName = self.config.methodName
        methodParams = self.config.methodParams
        lambd = methodParams["lambd"]
        method_func = getattr(methods, methodName)

        if isinstance(lambd, float) or isinstance(lambd, int):
            lambd = np.array([lambd], dtype=FLOAT_DTYPE)
        else:
            lambd = np.array(lambd).astype(FLOAT_DTYPE)

        if not isinstance(lambd, ARRAY):
            raise ValueError("Lambd must be given as a list or a numpy array.")

        method = []
        for item in lambd:
            params_item = dict(methodParams)  # copy methodParams
            params_item["lambd"] = item  # replace lambd list with single lambd
            params_item = list(params_item.values())  # convert to list
            method.append(method_func(*params_item))  # append method with lambd=item
        self.method = method

    def _optimize(self):
        if not isinstance(self.method, list):
            raise ValueError("Method must be given as a list.")
        solutions = []
        optimize_func = getattr(optimize, self.config.optimName)
        for item in enumerate(self.method):
            solutions.append(
                optimize_func(
                    R=self.model.regression_matrix,
                    F=self.prep.modifiedF,
                    x0=self.model.coeffs,
                    method=item,
                    maxiter=self.config.optimMaxIter,
                    tol=self.config.optimTol,
                )
            )
        # Map solutions content to output dataclass
        self.output.coefficients = np.array(
            [item.x for item in solutions], dtype=FLOAT_DTYPE
        )
        self.output.eps = np.array([item.eps for item in solutions], dtype=FLOAT_DTYPE)
        self.output.residuals = np.array(
            [item.residuum for item in solutions], dtype=FLOAT_DTYPE
        )
        # Checkpoint: Store output to JSON
        save_to_json(self.output, self.output_path)

    def _assign_coeffs(self):
        for i, item in enumerate(self.model):
            item.coeffs = self.output.coefficients[i]

    def _evaluate(self):
        self.output.valsF = np.array(
            [item.forward() for item in self.model], dtype=FLOAT_DTYPE
        )
        self.output.valsS = np.array(
            [item(self.prep.modifiedOmega) for item in self.model], dtype=FLOAT_DTYPE
        )
        self.output.integral = np.array(
            [np.trapz(item, self.prep.modifiedOmega) for item in self.output.valsS],
            dtype=FLOAT_DTYPE,
        )
        # Checkpoint: Store output to JSON
        save_to_json(self.output, self.output_path)

    def plot_coeffs(self):
        self._matplot_coeffs()
        return self._plotly_coeffs()

    def _matplot_coeffs(self):
        pass

    def _plotly_coeffs(self):
        pass

    def plot_model(self):
        self._matplot_model()
        return self._plotly_model()

    def _matplot_model(self):
        pass

    def _plotly_model(self):
        pass

    def plot_forward_model(self):
        self._matplot_forward_model()
        return self._plotly_forward_model()

    def _matplot_forward_model(self):
        pass

    def _plotly_forward_model(self):
        pass

    def plot_error_model(self):
        self._matplot_error_model()
        return self._plotly_error_model()

    def _matplot_error_model(self):
        pass

    def _plotly_error_model(self):
        pass

    def plot_error_forward_model(self):
        self._matplot_error_forward_model()
        return self._plotly_error_forward_model()

    def _matplot_error_forward_model(self):
        pass

    def _plotly_error_forward_model(self):
        pass

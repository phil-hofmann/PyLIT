import os
import numpy as np
import numba as nb
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
from pylit.frontend.utils import (  # TODO avoid importing frontend in backend
    extract_params,
)

MODERN_GREEN = "#50C878"

# TODO:
# - Add seperate methods for saving config, prep, output ... (INTEGRITY CHECKS)


class Experiment:

    def __init__(self, name: str, workspace: str = "/", make: bool = False):
        self.name = name
        self.workspace = workspace
        self.directory = os.path.join(workspace, name)
        self.path_F = os.path.join(self.directory, "F.csv")
        self.path_S = os.path.join(self.directory, "S.csv")
        self.config_path = os.path.join(self.directory, "config.json")
        self.prep_path = os.path.join(self.directory, "prep.json")
        self.output_path = os.path.join(self.directory, "output.json")
        self.path_run = os.path.join(self.directory, "run.py")
        self.plots_dir = os.path.join(self.directory, "plots")

        self._check_directories(make)
        self._check_json_files(make)

        if os.path.isfile(self.output_path):
            self._init_model()
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
        # TODO enhance to test all and put this somewhere else?
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

    @property
    def ready_to_finish(self):
        if not self.exported or not self.prepared:
            return False
        if self.config.methodName == "":
            return False
        if self.config.methodParams == {}:
            return False
        if self.config.modelName == "":
            return False
        if self.config.modelParams == {}:
            return False
        if self.config.optimName == "":
            return False
        if self.config.optimParams == {}:
            return False
        if self.config.scalingName == "":
            return False
        if self.config.scalingParams == {}:
            return False
        return True

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
            self.config.name = self.name  #  TODO maybe remove this
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

    def _check_save_config(self):
        # Model Params
        modelName = self.config.modelName
        if modelName != "":
            modelParams = self.config.modelParams
            model_class = getattr(models, modelName)
            model_func = model_class.__init__
            model_class_params = extract_params(model_func).keys()
            modelParams = {
                key: value
                for key, value in modelParams.items()
                if key in model_class_params
            }  # Only keep modelParams that are in the model_class_params
            self.config.modelParams = modelParams  # Update modelParams

        # Method Params
        methodName = self.config.methodName
        if methodName != "":
            methodParams = self.config.methodParams
            method_func = getattr(methods, methodName)
            method_func_params = extract_params(method_func).keys()
            methodParams = {
                key: methodParams[key] if key in methodParams else None
                for key in method_func_params
            }  # Only keep methodParams that are in the method_func_params else set to None
            self.config.methodParams = methodParams  # Update methodParams
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

        # Noise Params
        noiseName = self.config.noiseName
        if noiseName != "":
            noiseParams = self.config.noiseParams
            noise_class = getattr(noise_iid, noiseName)
            noise_func = noise_class.__init__
            noise_func_params = extract_params(noise_func).keys()
            noiseParams = {
                key: noiseParams[key]
                for key in noiseParams.keys()
                if key in noise_func_params
            }
            self.config.noiseParams = noiseParams

        # Noise Convolution Params
        noiseConvName = self.config.noiseConvName
        if noiseConvName != "":
            noiseConvParams = self.config.noiseConvParams
            noiseConv_class = getattr(noise_conv, noiseConvName)
            noiseConv_func = noiseConv_class.__init__
            noiseConv_func_params = extract_params(noiseConv_func).keys()
            noiseConvParams = {
                key: noiseConvParams[key]
                for key in noiseConvParams.keys()
                if key in noiseConv_func_params
            }
            self.config.noiseConvParams = noiseConvParams

        # Save config to JSON
        save_to_json(self.config, self.config_path)

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
        # Check config and save (so far) to JSON
        self._check_save_config()
        self._scale_and_noise_F()
        self.prep.tauMin, self.prep.tauMax = np.min(self.prep.tau), np.max(
            self.prep.tau
        )
        self._extend_and_scale_S()
        self.prep.modifiedOmegaMin, self.prep.modifiedOmegaMax = np.min(
            self.prep.modifiedOmega
        ), np.max(self.prep.modifiedOmega)
        self.prep.expS, self.prep.stdS = exp_std(
            self.prep.modifiedOmega, self.prep.modifiedS
        )
        self.prep.forwardModifiedS = np.array(
            [
                np.trapz(
                    self.prep.modifiedS * np.exp(-self.prep.modifiedOmega * tau),
                    self.prep.modifiedOmega,
                )
                for tau in self.prep.tau
            ],
            dtype=FLOAT_DTYPE,
        )
        self.prep.forwardModifiedSAbsError = np.abs(
            self.prep.forwardModifiedS - self.prep.modifiedF
        )
        self.prep.forwardModifiedSMaxError = np.max(self.prep.forwardModifiedSAbsError)
        # Save preparation to JSON
        save_to_json(self.prep, self.prep_path)

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

    def _extend_and_scale_S(self):
        modifiedOmega = np.copy(self.prep.omega)
        modifiedS = np.copy(self.prep.S)
        if self.config.PosS:
            modifiedS = np.maximum(0, modifiedS)
        if self.config.ExtS:
            modifiedOmega = extend_on_negative_axis(self.prep.omega)
            modifiedS = extend_S(modifiedS, self.prep.omega, self.prep.tauMax)
        if self.config.trapzS:
            modifiedS /= np.trapz(modifiedS, modifiedOmega)
        self.prep.modifiedOmega = modifiedOmega
        self.prep.modifiedS = modifiedS

    def plot_prep(self):
        self._matplot_prep_data()
        self._matplot_prep_forward()
        return self._plotly_prep_data(), self._plotly_prep_forward()

    def _matplot_prep_data(self):
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
            label="$S(\\omega)$",
        )
        ax2.set_xlabel("$\\omega$")
        ax2.set_ylabel("$S(\\omega)$")
        ax2.legend()
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Store matplot-plot
        plt.savefig(os.path.join(self.plots_dir, "prep-data.png"))

    def _plotly_prep_data(self):
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
        fig1.write_html(os.path.join(self.plots_dir, "prep-data-F.html"))
        fig2.write_html(os.path.join(self.plots_dir, "prep-data-S.html"))

        return [fig1, fig2]

    def _matplot_prep_forward(self):
        # Set figure size and create subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), sharey=False, sharex=False
        )

        # Plot forward S on the left subplot
        ax1.set_title("$\\tau$-Space")
        ax1.plot(
            self.prep.tau,
            self.prep.forwardModifiedS,
            label="$L[S](\\tau)$",
        )
        ax1.plot(
            self.prep.tau,
            self.prep.modifiedF,
            label="$F(\\tau)$",
            color="black",
            linestyle="--",
        )
        ax1.set_xlabel("$\\tau$")
        ax1.set_ylabel("Laplace Transform")
        ax1.legend()
        ax1.grid(True)

        # Plot forward S abs error on the right subplot
        ax2.set_title("$\\tau$-Space")
        ax2.plot(
            self.prep.tau,
            self.prep.forwardModifiedSAbsError,
            label="$|L[S](\\tau) - F(\\tau)|$",
        )
        ax2.set_xlabel("$\\tau$")
        ax2.set_ylabel("Absolute Error")
        ax2.legend()
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Store matplot-plot
        plt.savefig(os.path.join(self.plots_dir, "prep-forward.png"))

    def _plotly_prep_forward(self):
        # Create a Plotly figure
        fig1 = go.Figure()
        fig2 = go.Figure()

        # Plot forward S on the left subplot
        fig1.add_trace(
            go.Scatter(
                x=self.prep.tau,
                y=self.prep.modifiedF,
                mode="lines",
                name=f"F(τ)",
                line=dict(dash="dash"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=self.prep.tau,
                y=self.prep.forwardModifiedS,
                mode="lines",
                name=f"L[S](τ)",
            )
        )

        # Plot forward S abs error on the right subplot
        fig2.add_trace(
            go.Scatter(
                x=self.prep.tau,
                y=self.prep.forwardModifiedSAbsError,
                mode="lines",
                name="$|L[S](\\tau)-F(\\tau)|$",
            )
        )
        # Update layout
        fig1.update_layout(
            title="τ-Space",
            xaxis=dict(title="τ"),
            yaxis=dict(title="Laplace Transform"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig2.update_layout(
            title="ω-Space",
            xaxis=dict(title="ω"),
            yaxis=dict(title="Absolute Error"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        # Store plotly-plots
        fig1.write_html(os.path.join(self.plots_dir, "prep-forward-S.html"))
        fig2.write_html(os.path.join(self.plots_dir, "prep-forward-S-abs-error.html"))

        return [fig1, fig2]

    def create_run(self):
        if not self.ready_to_finish:
            return False

        # Save config to JSON
        self._check_save_config()

        # Create the content for run.py
        run_py_content = f"""# This file was automatically generated by pylit.\n# Path: {self.path_run}\n# To run the experiment, execute this file with Python:\n# conda activate pylit\n# python {self.path_run}\n# conda deactivate\n\nfrom pylit.backend.core import Experiment\n\nname="{self.name}"\nworkspace="{self.workspace}"\nexp = Experiment(name, workspace)\nexp.fit_model()\nexp.plot_results()"""

        # Write the content to run.py
        with open(self.path_run, "w") as file:
            file.write(run_py_content)

        return True

    def fit_model(self):
        self._init_model(print_name=True)
        self._init_method()
        self._optimize()
        self._evaluate()

    def _init_model(self, print_name=False):
        modelName = self.config.modelName
        if print_name:
            print("Model Name: ", modelName)
        modelParams = self.config.modelParams
        model_class = getattr(models, modelName)
        model = model_class(**modelParams)
        model.grid_points = self.prep.tau
        model = getattr(models.scaling, self.config.scalingName)(
            lrm=model, **self.config.scalingParams
        )
        self.model = model

    def _init_method(self):
        methodName = self.config.methodName
        print("Method Name: ", methodName)
        methodParams = self.config.methodParams
        method_func = getattr(methods, methodName)
        method_func_params = extract_params(method_func).keys()
        methodParams = {
            key: value
            for key, value in methodParams.items()
            if key in method_func_params
        }
        self.config.methodParams = methodParams  # NOTE Update methodParams
        lambd = methodParams["lambd"]
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
            method.append(method_func(**params_item))  # append method with lambd=item
        self.method = method

    def _optimize(self):
        optimName = self.config.optimName
        print("Optimization Name: ", optimName)
        if not isinstance(self.method, list):
            raise ValueError("Method must be given as a list.")
        solutions = []
        optimize_func = getattr(optimize, optimName)
        for i, item in enumerate(self.method):
            x0 = self.model.coeffs  # Is zero defaulted to zero!
            R = self.model.regression_matrix
            F = self.prep.modifiedF
            m = R.shape[1]
            if (
                not self.config.x0Reset
                and self.output is not None
                and self.output.coefficients is not None
                and len(self.output.coefficients[i]) == m # NOTE else the model parameters have changed!
            ):
                x0 = self.output.coefficients[i]

            maxiter = self.config.optimParams["maxiter"]
            tol = self.config.optimParams["tol"]
            protocol = self.config.optimParams["protocol"]
            svd = self.config.optimParams["svd"]

            if self.config.adaptiveActive:

                def optim_RFx0(R, F, x0):
                    return optimize_func(
                        R=R,
                        F=F,
                        x0=x0,
                        method=item,
                        maxiter=maxiter,
                        tol=tol,
                        protocol=protocol,
                        svd=svd,
                    )

                steps = len(self.model.params[0])
                solution = optimize.adaptive_RF(
                    R=R,
                    F=F,
                    x0=x0,
                    steps=steps,
                    optim_RFx0=optim_RFx0,
                    residuum_mode=self.config.adaptiveResiduumMode,
                )

                solutions.append(solution)
            else:
                solution = optimize_func(
                    R=R,
                    F=F,
                    x0=x0,
                    method=item,
                    maxiter=maxiter,
                    tol=tol,
                    protocol=protocol,
                    svd=svd,
                )

                solutions.append(solution)

        if self.output is None:
            self.output = Output()

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

    def _evaluate(self):

        valsF, valsS, integral = [], [], []

        for i, coeffs in enumerate(self.output.coefficients):
            self.model.coeffs = coeffs
            valsF.append(self.model.forward())
            valsS.append(self.model(self.prep.modifiedOmega))
            integral = np.append(integral, np.trapz(valsS[i], self.prep.modifiedOmega))

        self.output.valsF = valsF
        self.output.valsS = valsS
        self.output.integral = integral

        # Checkpoint: Store output to JSON
        save_to_json(self.output, self.output_path)

    def plot_results(self):
        if self.config.plot_coeffs:
            self.plot_coeffs()
        if self.config.plot_model:
            self.plot_model()
        if self.config.plot_error_forward_model:
            self.plot_forward_model()
        if self.config.plot_error_model:
            self.plot_error_model()
        if self.config.plot_error_forward_model:
            self.plot_error_forward_model()

    def plot_coeffs(self):
        self._matplot_coeffs()
        return self._plotly_coeffs()

    def _matplot_coeffs(self):
        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Display the coefficients array as an image
        cax = ax.imshow(self.output.coefficients, aspect="auto", cmap="viridis")

        # Add a color bar
        fig.colorbar(cax)

        # Add labels and title
        ax.set_title("Coefficients")
        ax.set_xlabel("Coefficient Index")
        ax.set_ylabel("Sample Index")

        # Adjust layout
        plt.tight_layout()

        # Store matplot-plot
        plt.savefig(os.path.join(self.plots_dir, "res-coefficients.png"))

    def _plotly_coeffs(self):
        # Create a heatmap
        heatmap = go.Heatmap(z=self.output.coefficients, colorscale="Viridis")

        # Create a plotly figure
        fig = go.Figure(data=[heatmap])

        # Add labels and title
        fig.update_layout(
            title="Coefficients",
            xaxis_title="Coefficient Index",
            yaxis_title="Sample Index",
        )

        # Save the plot as an HTML file
        fig.write_html(os.path.join(self.plots_dir, "res-coefficients.html"))

        return [fig]

    def plot_model(self):
        self._matplot_model()
        return self._plotly_model()

    def _matplot_model(self):
        x_omega = self.prep.modifiedOmega
        modifiedS = self.prep.modifiedS
        valsS = np.array(self.output.valsS, dtype=FLOAT_DTYPE)
        max_vals = np.max(valsS, axis=0)
        min_vals = np.min(valsS, axis=0)
        avg_vals = np.mean(valsS, axis=0)

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Plot S
        ax.plot(x_omega, modifiedS, label="S(ω)", color="blue")

        # Plot Models
        ax.plot(x_omega, min_vals, label="Sₗ(ω)", color="green")
        ax.plot(x_omega, max_vals, label="Sᵤ(ω)", color="green")
        ax.plot(x_omega, avg_vals, label="Sₐ(ω)", color="green")

        # Fill between min and max values
        ax.fill_between(x_omega, min_vals, max_vals, color="green", alpha=0.3)

        # Add title and labels
        ax.set_title("ω-Space")
        ax.set_xlabel("ω")
        ax.set_ylabel("S(ω)")

        # Add legend
        ax.legend(loc="upper right")

        # Adjust layout
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(os.path.join(self.plots_dir, "res-model.png"))

    def _plotly_model(self):
        x_omega = self.prep.modifiedOmega
        modifiedS = self.prep.modifiedS
        valsS = np.array(self.output.valsS, dtype=FLOAT_DTYPE)
        max_vals = np.max(valsS, axis=0)
        min_vals = np.min(valsS, axis=0)
        avg_vals = np.mean(valsS, axis=0)

        # Create a plotly figure
        fig = go.Figure()

        # Plot S
        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=modifiedS,
                mode="lines",
                name=f"S(ω)",
            )
        )

        # Plot Models

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=min_vals,
                mode="lines",
                name=f"Sₗ(ω)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=max_vals,
                fill="tonexty",
                mode="lines",
                name=f"Sᵤ(ω)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=avg_vals,
                mode="lines",
                name=f"Sₐ(ω)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.update_layout(
            title="ω-Space",
            xaxis=dict(title="ω"),
            yaxis=dict(title="S(ω)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.write_html(os.path.join(self.plots_dir, "res-model.html"))

        return [fig]

    def plot_forward_model(self):
        self._matplot_forward_model()
        return self._plotly_forward_model()

    def _matplot_forward_model(self):
        x_omega = self.prep.tau
        modifiedF = self.prep.modifiedF
        valsF = np.array(self.output.valsF, dtype=FLOAT_DTYPE)
        max_vals = np.max(valsF, axis=0)
        min_vals = np.min(valsF, axis=0)
        avg_vals = np.mean(valsF, axis=0)

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Plot F
        ax.plot(x_omega, modifiedF, label="F(τ)", color="blue")

        # Plot Forward Models
        ax.plot(x_omega, min_vals, label="Fₗ(τ)", color="green")
        ax.plot(x_omega, max_vals, label="Fᵤ(τ)", color="green")
        ax.plot(x_omega, avg_vals, label="Fₐ(τ)", color="green")

        # Fill between min and max values
        ax.fill_between(x_omega, min_vals, max_vals, color="green", alpha=0.3)

        # Add title and labels
        ax.set_title("τ-Space")
        ax.set_xlabel("τ")
        ax.set_ylabel("F(τ)")

        # Add legend
        ax.legend(loc="upper right")

        # Adjust layout
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(os.path.join(self.plots_dir, "res-forward-model.png"))

    def _plotly_forward_model(self):
        x_omega = self.prep.tau
        modifiedF = self.prep.modifiedF
        valsF = np.array(self.output.valsF, dtype=FLOAT_DTYPE)
        max_vals = np.max(valsF, axis=0)
        min_vals = np.min(valsF, axis=0)
        avg_vals = np.mean(valsF, axis=0)

        # Create a plotly figure
        fig = go.Figure()

        # Plot F
        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=modifiedF,
                mode="lines",
                name=f"F(τ)",
            )
        )

        # Plot Forward Models

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=min_vals,
                mode="lines",
                name=f"Fₗ(τ)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=max_vals,
                fill="tonexty",
                mode="lines",
                name=f"Fᵤ(τ)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=avg_vals,
                mode="lines",
                name=f"Fₐ(τ)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.update_layout(
            title="τ-Space",
            xaxis=dict(title="τ"),
            yaxis=dict(title="F(τ)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.write_html(os.path.join(self.plots_dir, "res-forward-model.html"))

        return [fig]

    def plot_error_model(self):
        self._matplot_error_model()
        return self._plotly_error_model()

    def _matplot_error_model(self):
        x_omega = self.prep.modifiedOmega
        modifiedS = self.prep.modifiedS
        valsS = np.array(self.output.valsS, dtype=FLOAT_DTYPE)
        eps = np.array([np.abs(vals - modifiedS) for vals in valsS])
        max_eps = np.max(eps, axis=0)
        min_eps = np.min(eps, axis=0)
        avg_eps = np.mean(eps, axis=0)

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Plot Models
        ax.plot(x_omega, min_eps, label="εₗ(ω)", color="green")
        ax.plot(x_omega, max_eps, label="εᵤ(ω)", color="green")
        ax.plot(x_omega, avg_eps, label="εₐ(ω)", color="green")

        # Fill between min and max values
        ax.fill_between(x_omega, min_eps, max_eps, color="green", alpha=0.3)

        # Add title and labels
        ax.set_title("ω-Space")
        ax.set_xlabel("ω")
        ax.set_ylabel("ε(ω)")

        # Add legend
        ax.legend(loc="upper right")

        # Adjust layout
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(os.path.join(self.plots_dir, "res-model-error.png"))

    def _plotly_error_model(self):
        x_omega = self.prep.modifiedOmega
        modifiedS = self.prep.modifiedS
        valsS = np.array(self.output.valsS, dtype=FLOAT_DTYPE)
        eps = np.array([np.abs(vals - modifiedS) for vals in valsS])
        max_eps = np.max(eps, axis=0)
        min_eps = np.min(eps, axis=0)
        avg_eps = np.mean(eps, axis=0)

        # Create a plotly figure
        fig = go.Figure()

        # Plot Models Error

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=min_eps,
                mode="lines",
                name=f"εₗ(ω)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=max_eps,
                fill="tonexty",
                mode="lines",
                name=f"εᵤ(ω)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=avg_eps,
                mode="lines",
                name=f"εₐ(ω)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.update_layout(
            title="ω-Space",
            xaxis=dict(title="ω"),
            yaxis=dict(title="ε(ω)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.write_html(os.path.join(self.plots_dir, "res-model-error.html"))

        return [fig]

    def plot_error_forward_model(self):
        self._matplot_error_forward_model()
        return self._plotly_error_forward_model()

    def _matplot_error_forward_model(self):
        x_omega = self.prep.tau
        modifiedF = self.prep.modifiedF
        valsF = np.array(self.output.valsF, dtype=FLOAT_DTYPE)
        eps = np.array([np.abs(vals - modifiedF) for vals in valsF])
        max_eps = np.max(eps, axis=0)
        min_eps = np.min(eps, axis=0)
        avg_eps = np.mean(eps, axis=0)

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

        # Plot Models
        ax.plot(x_omega, min_eps, label="εₗ(τ)", color="green")
        ax.plot(x_omega, max_eps, label="εᵤ(τ)", color="green")
        ax.plot(x_omega, avg_eps, label="εₐ(τ)", color="green")

        # Fill between min and max values
        ax.fill_between(x_omega, min_eps, max_eps, color="green", alpha=0.3)

        # Add title and labels
        ax.set_title("τ-Space")
        ax.set_xlabel("τ")
        ax.set_ylabel("ε(τ)")

        # Add legend
        ax.legend(loc="upper right")

        # Adjust layout
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(os.path.join(self.plots_dir, "res-forward-model-error.png"))

    def _plotly_error_forward_model(self):
        x_omega = self.prep.tau
        modifiedF = self.prep.modifiedF
        valsF = np.array(self.output.valsF, dtype=FLOAT_DTYPE)
        eps = np.array([np.abs(vals - modifiedF) for vals in valsF])
        max_eps = np.max(eps, axis=0)
        min_eps = np.min(eps, axis=0)
        avg_eps = np.mean(eps, axis=0)

        # Create a Plotly figure
        fig = go.Figure()

        # Plot Forward Models Error
        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=min_eps,
                mode="lines",
                name=f"εₗ(τ)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=max_eps,
                fill="tonexty",
                mode="lines",
                name=f"εᵤ(τ)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_omega,
                y=avg_eps,
                mode="lines",
                name=f"εₐ(τ)",
                line=dict(color=MODERN_GREEN),
            )
        )

        fig.update_layout(
            title="τ-Space",
            xaxis=dict(title="τ"),
            yaxis=dict(title="ε(τ)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.write_html(os.path.join(self.plots_dir, "res-forward-model-error.html"))

        return [fig]

import os
import numpy as np
import pandas as pd

from pylit.backend import models, methods, optimize
from pylit.global_settings import ARRAY, FLOAT_DTYPE, MOMENT_ORDERS
from pylit.backend.core import (
    DataLoader,
    Configuration,
    Preparation,
    Output,
    noise_iid,
    noise_conv,
)
from pylit.backend.core.utils import (
    complete_detailed_balance,
    exp_std,
    moments,
    save_to_json,
    load_from_json,
)
from pylit.frontend.utils import (  # TODO avoid importing frontend in backend
    extract_params,
)


class Experiment:

    def __init__(
        self,
        name: str,
        workspace: str = "/",
    ):
        self.name = name

        # Directories
        self.workspace = workspace
        self.directory = os.path.join(workspace, name)
        self.source_directory = os.path.join(self.directory, "src")
        self.source_json_directory = os.path.join(self.source_directory, "json")
        self.source_csv_directory = os.path.join(self.source_directory, "csv")
        self.plots_directory = os.path.join(self.directory, "plots")
        self.plots_default_model_directory = os.path.join(
            self.plots_directory, "default-model"
        )
        self.plots_model_directory = os.path.join(self.plots_directory, "model")
        self.plots_default_model_html_directory = os.path.join(
            self.plots_default_model_directory, "html"
        )
        self.plots_default_model_png_directory = os.path.join(
            self.plots_default_model_directory, "png"
        )
        self.plots_model_html_directory = os.path.join(
            self.plots_model_directory, "html"
        )
        self.plots_model_png_directory = os.path.join(self.plots_model_directory, "png")
        self._create_directories()

        # Files
        self.path_F = os.path.join(self.source_csv_directory, "F.csv")
        self.path_D = os.path.join(self.source_csv_directory, "D.csv")
        self.path_L_S = os.path.join(self.source_csv_directory, "L(S).csv")
        self.path_S = os.path.join(self.source_csv_directory, "S.csv")
        self.config_path = os.path.join(self.source_json_directory, "config.json")
        self.prep_path = os.path.join(self.source_json_directory, "prep.json")
        self.output_path = os.path.join(self.source_json_directory, "output.json")
        self.path_run = os.path.join(self.directory, "run.py")
        self._load_files()
        # self.create_run_py()

        if os.path.isfile(self.output_path):
            # TODO Additional check if the output is valid!
            self._init_model(self.output.timeScaling, self.output.normalization)

        else:
            self.model = None
            self.method = None
            self.output = None

    @property
    def imported_F(self):
        return (
            self.prep is not None
            and (self.prep.tau is not None and self.prep.F is not None)
            and (self.prep.tau.shape == self.prep.F.shape)
            and (self.prep.tau.size > 0)
        )

    @property
    def imported_D(self):
        # There is more to check, but this is not necessary.
        return (
            self.prep is not None
            and (self.prep.omega is not None and self.prep.D is not None)
            and (self.prep.omega.shape == self.prep.D.shape)
            and (self.prep.omega.size > 0)
        )

    @property
    def imported(self):
        return self.imported_F and self.imported_D

    def _create_directories(self):
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        if not os.path.isdir(self.source_directory):
            os.makedirs(self.source_directory)
        if not os.path.isdir(self.source_json_directory):
            os.makedirs(self.source_json_directory)
        if not os.path.isdir(self.source_csv_directory):
            os.makedirs(self.source_csv_directory)
        if not os.path.isdir(self.plots_directory):
            os.makedirs(self.plots_directory)
        if not os.path.isdir(self.plots_default_model_directory):
            os.makedirs(self.plots_default_model_directory)
        if not os.path.isdir(self.plots_model_directory):
            os.makedirs(self.plots_model_directory)
        if not os.path.isdir(self.plots_default_model_html_directory):
            os.makedirs(self.plots_default_model_html_directory)
        if not os.path.isdir(self.plots_default_model_png_directory):
            os.makedirs(self.plots_default_model_png_directory)
        if not os.path.isdir(self.plots_model_html_directory):
            os.makedirs(self.plots_model_html_directory)
        if not os.path.isdir(self.plots_model_png_directory):
            os.makedirs(self.plots_model_png_directory)

    def create_run_py(
        self,
        coefficients: bool,
        model: bool,
        forward_model: bool,
        forward_model_error: bool,
    ):
        if not os.path.isfile(self.path_run):
            run_py_content = f"""
# This file was automatically generated by pylit.\n
# Path: {self.path_run}\n
# To run the experiment, execute this file with Python:\n
# source venv/bin/activate\n
# poetry shell\n
# python {self.path_run}\n
# deactivate\n
from pylit.backend.core import Experiment\n
from pylit.backend.core.plot_utils import plot_results\n
name="{self.name}"\n
workspace="{self.workspace}"\n
exp = Experiment(name, workspace)\n
exp.fit_model()\n
plot_results(
    exp=exp,
    coefficients={coefficients},
    model={model},
    forward_model={forward_model},
    forward_model_error={forward_model_error},
)"""

            # Write the content to run.py
            with open(self.path_run, "w") as file:
                file.write(run_py_content)

    def _load_files(self):
        # Load config.json
        if os.path.isfile(self.config_path):
            self.config = load_from_json(Configuration, self.config_path)
        else:
            self.config = Configuration()
            self.config.name = self.name  #  TODO remove this somewhen

        # Load prep.json
        if os.path.isfile(self.prep_path):
            self.prep = load_from_json(Preparation, self.prep_path)
        else:
            self.prep = Preparation()

        # Load output.json
        if os.path.isfile(self.output_path):
            self.output = load_from_json(Output, self.output_path)
        else:
            self.output = Output()

    def save_config(self):
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
            # self._init_model()  # NOTE !NOT! Initialize model!?!?!?!
            methodParams = self.config.methodParams
            method_func = getattr(methods, methodName)
            method_func_params = extract_params(method_func).keys()
            methodParams = {
                key: methodParams[key] if key in methodParams else None
                for key in method_func_params
            }  # Only keep methodParams that are in the method_func_params else set to None
            self.config.methodParams = methodParams  # Update methodParams

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

    def import_F(self) -> bool:
        # Fetch tau, F from the data file
        dl = DataLoader(self.path_F)
        dl.fetch()

        # Data Checks
        if dl.data is None:
            raise ValueError("Data file for 'F' is empty.")
        if len(dl.data.shape) != 2:
            raise ValueError("Data file for 'F' must have exactly 2 columns.")
        if dl.data.shape[1] != 2:
            raise ValueError("Data file for 'F' must have exactly 2 columns.")
        tau, F = dl.data.T

        # Sort tau ascending
        idx = np.argsort(tau)
        tau, F = tau[idx], F[idx]

        # Save to Preparation
        self.prep.tau, self.prep.F = tau, F

        # Tau Min, Tau Max
        self.prep.tauMin = np.min(tau)
        self.prep.tauMax = np.max(tau)

        # Clear DataLoader and save to JSON
        dl.clear()
        save_to_json(self.prep, self.prep_path)

        return True

    def import_D(self, non_negative=True, detailed_balance=True) -> bool:
        # Fetch omega, D from the data file
        dl = DataLoader(self.path_D)
        dl.fetch()

        # Data Checks
        if dl.data is None:
            raise ValueError("Data file for 'D' is empty.")
        if len(dl.data.shape) != 2:
            raise ValueError("Data file for 'D' must have exactly 2 columns.")
        if dl.data.shape[1] != 2:
            raise ValueError("Data file for 'D' must have exactly 2 columns.")
        omega, D = dl.data.T

        # Sort omega ascending
        idx = np.argsort(omega)
        omega, D = omega[idx], D[idx]

        # Non-Negativity
        if non_negative:
            D = np.maximum(0, D)

        # Detailed Balance
        if detailed_balance:
            omega, D = complete_detailed_balance(omega, D, beta=self.prep.tauMax)
        self.prep.omega, self.prep.D = omega, D

        # Omega Min, Omega Max
        self.prep.omegaMin = np.min(omega)
        self.prep.omegaMax = np.max(omega)

        # Expected Value and Standard Deviation
        self.prep.expD, self.prep.stdD = exp_std(omega, D)

        # Frequency Moments
        self.prep.freqMomentsD = moments(
            omega,
            D,
            MOMENT_ORDERS,
        )

        # Forward S, Forward S Abs Error, Forward S Max Error
        self.prep.forwardD = np.array(
            [
                np.trapz(
                    D * np.exp(-omega * tau),
                    omega,
                )
                for tau in self.prep.tau
            ],
            dtype=FLOAT_DTYPE,
        )
        self.prep.forwardDAbsError = np.abs(self.prep.forwardD - self.prep.F)
        self.prep.forwardDMaxError = np.max(self.prep.forwardDAbsError)

        # Clear DataLoader and save to JSON
        dl.clear()
        save_to_json(self.prep, self.prep_path)

        return True

    def apply_noise_F(self):
        # Create Noise
        if self.config.noiseActive:
            noiseF = getattr(noise_iid, self.config.noiseName)(
                *list(self.config.noiseParams.values())
            )(len(self.prep.F))

            # Convolve Noise
            if self.config.noiseConvActive:
                noiseF = getattr(noise_conv, self.config.noiseConvName)(
                    *list(self.config.noiseConvParams.values())
                )(noiseF)
            self.prep.noiseF = noiseF

        # Save to JSON
        save_to_json(self.prep, self.prep_path)

    def reset_noise_F(self):
        # Reset noiseF
        self.prep.noiseF = []
        # Save to JSON
        save_to_json(self.prep, self.prep_path)

    def fit_model(self, time_scaling=True, normalization=True):
        self._init_model(print_name=True, time_scaling=time_scaling)
        self._init_method()
        self._optimize(normalization=normalization)
        self._evaluate()

    def _init_model(self, print_name: bool, time_scaling: bool):
        modelName = self.config.modelName
        if print_name:
            print("\nModel Name: ", modelName)
        modelParams = self.config.modelParams
        model_class = getattr(models, modelName)
        model = model_class(**modelParams)
        model.grid_points = self.prep.tau
        if time_scaling:
            model = models.scaling.linear(lrm=model)
        self.model = model

    def _init_method(self):
        methodName = self.config.methodName
        if methodName == "":
            raise ValueError("Method Name is not set.")
        print("Method Name: ", methodName)
        methodParams = self.config.methodParams

        # Add Evaluation Matrix : it is not stored in the config!
        if "E" in methodParams:
            methodParams["E"] = self.model(self.prep.omega, matrix=True)
        # Add Model Matrix
        if "D" in self.config.methodParams:
            # Automatically scale by trapezodial rule
            D = self.prep.D
            D_trapz = np.trapz(np.abs(D), self.prep.omega)
            D = D / D_trapz if D_trapz != 0.0 else D
            methodParams["D"] = D
        # Add omegas
        if "omegas" in self.config.methodParams:
            methodParams["omegas"] = self.prep.omega

        method_func = getattr(methods, methodName)
        method_func_params = extract_params(method_func).keys()
        methodParams = {
            key: value
            for key, value in methodParams.items()
            if key in method_func_params
        }
        # Update methodParams
        self.config.methodParams = methodParams
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

    def _optimize(self, normalization: bool):
        optimName = self.config.optimName
        print("Optimization Name: ", optimName)
        if not isinstance(self.method, list):
            raise ValueError("Method must be given as a list.")
        solutions = []
        optimize_func = getattr(optimize, optimName)
        noise_F = self.prep.noiseF
        if self.prep.noiseF is None or len(self.prep.noiseF) == 0:
            noise_F = [0]

        for noise in noise_F:
            for i, method in enumerate(self.method):
                x0 = 0.0 * self.model.coeffs  # Is defaulted to zero!
                R = self.model.regression_matrix
                F = self.prep.F + noise
                # Automatically scale by max F
                F_max = np.max(F)
                F = F / F_max if F_max != 0.0 else F
                m = R.shape[1]
                if (
                    not self.config.x0Reset
                    and self.output is not None
                    and self.output.coefficients is not None
                    and len(self.output.coefficients[i])
                    == m  # NOTE else the model parameters have changed!
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
                            method=method,
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
                        method=method,
                        maxiter=maxiter,
                        tol=tol,
                        protocol=protocol,
                        svd=svd,
                    )

                    solutions.append(solution)

                # Rescale coefficients by F_max
                solutions[-1].x *= F_max

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
        forward_S, S = [], []
        for coeffs in self.output.coefficients:
            # Set the correct coefficients
            self.model.coeffs = coeffs
            # Compute Forward S and S
            forward_S.append(self.model.forward())
            S.append(self.model(self.prep.omega))

        # S:
        self.output.S = S
        self.output.expS, self.output.stdS = np.array(
            [
                exp_std(
                    self.prep.omega,
                    S_i,
                )
                for S_i in S
            ],
            dtype=FLOAT_DTYPE,
        ).T
        self.output.freqMomentsS = np.array(
            [
                moments(
                    self.prep.omega,
                    S_i,
                    MOMENT_ORDERS,
                )
                for S_i in S
            ],
            dtype=FLOAT_DTYPE,
        )

        # L(S)[τ]:
        self.output.forwardS = forward_S
        # TODO
        self.output.forwardSAbsError = np.array(
            [np.abs(forward_S_i - self.prep.F) for forward_S_i in forward_S],
            dtype=FLOAT_DTYPE,
        )
        self.output.forwardSMaxError = np.amax(self.output.forwardSAbsError, axis=0)

        # Store S, L(S) to CSV
        S_df = pd.DataFrame([self.prep.omega, *self.output.S]).T
        S_df.to_csv(self.path_S, index=False, header=False)
        L_S_df = pd.DataFrame([self.prep.tau, *self.output.forwardS]).T
        L_S_df.to_csv(self.path_L_S, index=False, header=False)

        # Store output to JSON
        save_to_json(self.output, self.output_path)

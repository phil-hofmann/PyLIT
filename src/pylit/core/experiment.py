import os
import numpy as np
from pylit import models, methods, optimize
from pylit.global_settings import ARRAY, FLOAT_DTYPE
from pylit.core import (
    DataLoader,
    Configuration,
    Preparation,
    Output,
    noise_iid,
    noise_conv,
)
from pylit.core.utils import (
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

    def __init__(self, name: str, make: bool = False):
        self.name = name

        if not os.path.isdir(name):
            if make:
                os.makedirs(name)
            else:
                raise NotADirectoryError(f"The directory '{name}' does not exist.")

        config_path = os.path.join(name, "config.json")
        prep_path = os.path.join(name, "prep.json")
        output_path = os.path.join(name, "output.json")

        # Check and load config.json
        if not os.path.isfile(config_path):
            if make:
                self.config = Configuration()
            else:
                raise FileNotFoundError(
                    f"The configuration file '{config_path}' does not exist."
                )
        else:
            self.config = load_from_json(config_path)

        # Check and load prep.json
        if not os.path.isfile(prep_path):
            self.prep = Preparation()
        else:
            self.prep = load_from_json(prep_path)

        # Check and load output.json
        if not os.path.isfile(output_path):
            self.output = Output()
        else:
            self.output = load_from_json(output_path)

        if os.path.isfile(output_path):
            self._init_model()
            self._assign_coeffs()
            self._benchmark()
        else:
            self.model = None
            self.method = None
            self.result = None

        # TODO
        # e=Experiment(...)
        # e.import_F()
        # e.import_S()
        # e.prepare()
        # e.fit()
        # e.plot()

    def import_F(self):
        # Fetch tau, F from the data file
        path = "/home/phil/Documents/github/pylit/src/pylit/helloworld"
        dl = DataLoader(f"{path}/F.csv")
        dl.fetch()
        if dl.data is None:
            raise ValueError("Data file for 'F' is empty.")
        if len(dl.data.shape) != 2:
            raise ValueError(dl.data)
            raise ValueError("Data file for 'F' must have exactly 2 columns.")
        self.prep.tau, self.prep.F = dl.data  # 0 ... tau, 1 ... F
        dl.clear()

    def import_S(self):
        # Fetch omega, S from the data file
        path = "/home/phil/Documents/github/pylit/src/pylit/helloworld"
        dl = DataLoader(f"{path}/S.csv")
        dl.fetch()
        if dl.data is None:
            raise ValueError("Data file for 'S' is empty.")
        if len(dl.data.shape) != 2:
            raise ValueError(dl.data)
            raise ValueError("Data file for 'S' must have exactly 2 columns.")
        self.prep.omega, self.prep.S = dl.data  #  0 ... omega, 1 ... S
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

    def fit(self):
        self._init_model()
        self._add_method_params()
        self._init_method()
        self._optimize()
        self._assign_coeffs()
        self._benchmark()

    def _init_model(self):
        modelParams = {
            key: value
            for key, value in self.config.modelParams.items()
            if key not in self._model_attrs
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

        if isinstance(lambd, list) or isinstance(lambd, ARRAY):
            method = []
            for item in lambd:
                params_item = dict(methodParams)  # copy methodParams
                params_item["lambd"] = item  # replace lambd list with single lambd
                params_item = list(params_item.values())  # convert to list
                method.append(
                    method_func(*params_item)
                )  # append method with lambd=item
        else:
            raise ValueError("Lambd must be given as a list or a numpy array.")
        self.method = method

    def _optimize(self):
        result = []
        optimize_func = getattr(optimize, self.config.optimName)

        if isinstance(self.method, list):
            for item in enumerate(self.method):
                result.append(
                    optimize_func(
                        R=self.model.regression_matrix,
                        F=self.prep.modifiedF,
                        x0=self.model.coeffs,
                        method=item,
                        maxiter=self.config.optimMaxIter,
                        tol=self.config.optimTol,
                    )
                )
        else:
            raise ValueError("Method must be given as a list.")
        self.result = result
        self.output.coefficients = np.array(
            [item.x for item in self.result], dtype=FLOAT_DTYPE
        )

    def _assign_coeffs(self):
        for i, item in enumerate(self.model):
            item.coeffs = self.output.coefficients[i]

    def _benchmark(self):
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

    def plot(self):
        pass
        # TODO

    def save(self):
        """
        Saves config, prep, and output to separate JSON files.
        """
        save_to_json(self.config, "config.json")
        if self.prep is not None:
            save_to_json(self.prep, "prep.json")
        if self.output is not None:
            save_to_json(self.output, "output.json")

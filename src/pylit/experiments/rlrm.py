import os
import pylit
import json
import numpy as np
import warnings

from pylit.models.ABC import RegularLinearRegressionModelABC
from pylit.global_settings import FLOAT_DTYPE, ARRAY
from pylit.optimize import Solution

from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class ExperimentRLRM:
    _models = {}
    _model_attrs = {"_coeffs", "_grid_points", "_reg_mat"} # These are the attributes that are not part of the model itself but are added by the experiment

    def __init__(self, path: str, config: Dict[str, Any]) -> None:
        # Typing
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")
        if not all(isinstance(key, str) for key in config.keys()):
            raise TypeError("Config keys must be strings.")
        # Structure
        if not pylit.ui.utils.check_structure(
            config, pylit.experiments.rlrm_es.es_contents_config
        ):
            raise ValueError(
                "Invalid config structure. Please check es_contents_config in rlrm_es.py."
            )
        # Assign
        self._path = path # TODO: Properties
        self._contents = {
            "config": config,
            "models": {},
            "output": {},
            "report": [],
            "res": [],
        }

    @property
    def contents(self) -> Dict[str, Any]:
        return self._contents

    @contents.setter
    def contents(self, contents: Dict[str, Any]) -> None:
        # Typing
        if not isinstance(contents, dict):
            raise TypeError("Contents must be a dictionary.")
        if not all(isinstance(key, str) for key in contents.keys()):
            raise TypeError("Contents keys must be strings.")
        # Structure
        if not pylit.ui.utils.check_structure(
            contents, pylit.experiments.rlrm_es.es_contents
        ):
            raise ValueError(
                "Invalid contents structure. Please check es_contents in rlrm_es.py."
            )
        # Assign
        self._contents = contents

    @property
    def config(self) -> Dict[str, Any]:
        return self._contents["config"]

    @config.setter
    def config(self, config: Dict[str, Any]) -> None:
        # Typing
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")
        if not all(isinstance(key, str) for key in config.keys()):
            raise TypeError("Config keys must be strings.")
        # Structure
        if not pylit.ui.utils.check_structure(
            config, pylit.experiments.rlrm_es.es_contents_config
        ):
            raise ValueError(
                "Invalid config structure. Please check es_contents_config in rlrm_es.py."
            )
        # Assign
        self._contents["config"] = config

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models: Dict[str, RegularLinearRegressionModelABC]):
        # Typing
        if not isinstance(models, dict):
            raise TypeError("Models must be a dictionary.")
        if not all(
            (
                isinstance(key, str)
                and key in self._contents["config"]["data"]["names"]
            )
            for key in models.keys()
        ):
            raise TypeError(
                "Models keys must be strings and correspond to the names in the data."
            )
        # Assign
        self._models = models

    def _set_contents_model(self, models: List[dict]):
        # Typing
        if not isinstance(models, list):
            raise TypeError("Contents models must be a List.")
        if not all(isinstance(item, dict) for item in models):
            raise TypeError(
                "Each contents model must consist of a dictionary."
            )
        # Structure
        for model in models:
            if not pylit.ui.utils.check_structure(
                model, pylit.experiments.rlrm_es.es_contents_models
            ):
                raise ValueError(
                    "Invalid model structure. Please check es_contents_models in rlrm_es.py."
                )
        # Assign
        self._contents["models"] = models
 
    def _models_from_list(self, models_list: List[dict]):
        # Typing
        if not isinstance(models_list, list):
            raise TypeError("Models must be a list.")
        if not all(isinstance(item, dict) for item in models_list):
            raise TypeError("Each model must be a dictionary.")
        # Get models
        models = { # Create models
            model["name"]:  getattr(pylit.models, model["model"]["name"])(**model["model"]["params"])
            for i, model in enumerate(models_list)
        }

        models = {  # Create models
            model["name"]: getattr(
                pylit.models, model["model"]["name"]
            )(**{k: v for k, v in model["model"]["params"].items() if k not in self._model_attrs})
            for model in models_list
        }
        # Set grid points, apply scaling and add configs
        for i, model in enumerate(models_list):
            key = model["name"]
            models[key].grid_points = self._contents["output"]["F"]["tau"]["contents"]
            models[key] = getattr(pylit.models.lrm_scaling, model["scaling"]["name"])(
                    lrm=models[key],
                    **model["scaling"]["params"])
            # Add Evaluation Matrix 
            if "E" in self._contents["config"]["method"]["params"][i] and self._contents["config"]["method"]["params"][i]["E"]:
                self._contents["config"]["method"]["params"][i]["E"] = models[key](self._contents["output"]["S"]["omega"]["modified"], matrix=True)
            # Add Default Model Matrix
            if "S" in self._contents["config"]["method"]["params"][i] and self._contents["config"]["method"]["params"][i]["S"]:
                self._contents["config"]["method"]["params"][i]["S"] = self._contents["output"]["S"][key]["modified"]
            # Add Default omegas
            if "omegas" in self._contents["config"]["method"]["params"][i] and self._contents["config"]["method"]["params"][i]["omegas"]:
                self._contents["config"]["method"]["params"][i]["omegas"] = self._contents["output"]["S"]["omega"]["modified"]
        return models
    
    def set_models(self, get_models: callable):
        # Apply the get_models function to self
        models_list = get_models(self)
        # Typing
        if not isinstance(models_list, list):
            raise TypeError("Models must be a list.")
        if not all(isinstance(item, dict) for item in models_list):
            raise TypeError("Each model must be a dictionary.")
        # Assign
        self._set_contents_model(models_list)
        self.models = self._models_from_list(models_list)  # Use global setter!

    @property
    def omega_min_and_max(self) -> List[FLOAT_DTYPE]:
        out = self._contents["output"]
        return [out["omega_min"], out["omega_max"]]

    @property
    def tau_min_and_max(self) -> List[FLOAT_DTYPE]:
        out = self._contents["output"]
        return [out["tau_min"], out["tau_max"]]

    def _import(self):
        data = self._contents["config"]["data"]

        # Fetch F
        dl = pylit.DataLoader(data["file_path_F"])
        dl.header = ["q", "tau", *data["names"]]
        print(f"HEADER: {dl.header}")
        dl.fetch()
        F = dl.dict()
        dl.clear()

        # Fetch S
        dl.file_name = data["file_path_S"]
        dl.header = ["omega_prime", "omega", *data["names"]]
        dl.fetch()
        S = dl.dict()
        dl.clear()

        # Assign
        self._contents["output"]["F"] = F
        self._contents["output"]["S"] = S

    def _scale_and_noise(self):
        data = self._contents["config"]["data"]
        out = self._contents["output"]

        for i, name in enumerate(data["names"]):
            contents = out["F"][name]["contents"]

            # Scaling Data
            if data["scale"][i]["max_F"]:
                contents /= np.max(contents)  

            # Noise
            if data["noise_active"][i]:
                if data["noise_conv_active"][i]:
                    noise = getattr(pylit.noise_iid, data["noise"][i]["name"])(
                        *list(data["noise"][i]["params"].values()))(np.zeros_like(contents)) # Create noise
                    noise = getattr(pylit.noise_conv, data["noise_conv"][i]["name"])(
                        *list(data["noise_conv"][i]["params"].values()))(noise) # Apply convolution
                    contents += noise

                else:
                    contents = getattr(pylit.noise_iid, data["noise"][i]["name"])(
                        *list(data["noise"][i]["params"].values()))(contents)
                    
            out["F"][name]["modified"] = contents

        # Assign
        self._contents["output"] = out

    def _tau_min_and_max(self) -> Tuple[FLOAT_DTYPE, FLOAT_DTYPE]:
        out = self._contents["output"]
        tau_min, tau_max = np.min(out["F"]["tau"]["contents"]), np.max(
            out["F"]["tau"]["contents"]
        )
        self._contents["output"]["tau_max"] = tau_max
        self._contents["output"]["tau_min"] = tau_min
        return tau_min, tau_max

    def _extension_and_scale(self):
        data = self._contents["config"]["data"]
        out = self._contents["output"]
        

        out["S"]["omega"]["modified"] = pylit.utils.extend_on_negative_axis(
            out["S"]["omega"]["contents"]
        )

        for i, name in enumerate(data["names"]):
            contents = out["S"][name]["contents"]

        # Positive part of S
            if data["scale"][i]["pos_S"]:
                contents = np.maximum(0, contents)

            # Extension of S to negative axis
            if data["scale"][i]["ext_S"]:
                contents = pylit.utils.extend_S(
                    contents, out["S"]["omega"]["contents"], out["tau_max"]
                )

            # Scaling Data by trapezoidal rule
            if data["scale"][i]["trapz_S"]:
                contents /= np.trapz(
                    contents, out["S"]["omega"]["modified"]
                ) 
            
            out["S"][name]["modified"] = contents

        # Assign
        self._contents["output"] = out

    def _omega_min_and_max(self) -> Tuple[FLOAT_DTYPE, FLOAT_DTYPE]:
        out = self._contents["output"]
        omega_min, omega_max = np.min(out["S"]["omega"]["modified"]), np.max(
            out["S"]["omega"]["modified"]
        )
        self._contents["output"]["omega_max"] = omega_max
        self._contents["output"]["omega_min"] = omega_min
        return omega_min, omega_max

    def _exp_and_std(self):
        data = self._contents["config"]["data"]
        out = self._contents["output"] 

        for name in data["names"]:
            exp, std = pylit.utils.exp_std(
                out["S"]["omega"]["contents"], out["S"][name]["contents"]
            )
            out["S"][name]["exp"] = exp
            out["S"][name]["std"] = std

            # Assign
            self._contents["output"] = out

    def prepare(self, protocol: bool = True) -> None:
        # Create F and S dictionaries
        self._import()
        self._scale_and_noise()
        tau_min, tau_max = self._tau_min_and_max()
        self._extension_and_scale()
        omega_min, omega_max = self._omega_min_and_max()
        self._exp_and_std()

        # Init report
        self._contents["report"] = []

        if protocol:
            self._contents["report"].append(
                "tau_min = "
                + "{:.2f}".format(tau_min)
                + ",\t"
                + "tau_max = "
                + "{:.2f}".format(tau_max)
                + "\n"
            )
            print(self._contents["report"][-1])

            self._contents["report"].append(
                "omega_min = "
                + "{:.2f}".format(omega_min)
                + ",\t"
                + "omega_max = "
                + "{:.2f}".format(omega_max)
                + "\n"
            )
            print(self._contents["report"][-1])

            self._contents["report"].append(
                "Output F:\n------------------------\n"
                + pylit.utils.print_str_dict(self._contents["output"]["F"])
                + "\n"
            )
            print(self._contents["report"][-1])

            self._contents["report"].append(
                "Output S:\n------------------------\n"
                + pylit.utils.print_str_dict(self._contents["output"]["S"])
                + "\n"
            )
            print(self._contents["report"][-1])

    def plotly_data(self, store=True):
        data = self._contents["config"]["data"]
        out = self._contents["output"]
        # Create a Plotly figure
        fig1 = go.Figure()
        fig2 = go.Figure()
        # Iterate over the data names
        for i, name in enumerate(data["names"]):
            # Plot F on the left subplot
            fig1.add_trace(go.Scatter(
                x=out["F"]["tau"]["contents"],
                y=out["F"][name]["modified"],
                mode='lines',
                name=f"F(τ) ({name})",
                line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
            ))
            # Plot S on the right subplot
            fig2.add_trace(go.Scatter(
                x=out["S"]["omega"]["modified"],
                y=out["S"][name]["modified"],
                mode='lines',
                name=f"S(ω) ({name})",
                line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
            ))
        # Update layout
        fig1.update_layout(
            title="τ-Space",
            xaxis=dict(title="τ"),
            yaxis=dict(title="F(τ)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        fig2.update_layout(
            title="ω-Space",
            xaxis=dict(title="ω"),
            yaxis=dict(title="S(ω)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        # Store plot
        if store:
            fig1.write_html(f"{self._directory}/imgs/data_plot_F.html")
            fig2.write_html(f"{self._directory}/imgs/data_plot_S.html")

        return [fig1, fig2]
    
    def plot_data(self, store: bool = True, giveback=False):
        data = self._contents["config"]["data"]
        out = self._contents["output"]

        # Set figure size and create subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), sharey=False, sharex=False
        )

        # Plot F on the left subplot
        ax1.set_title("$\\tau$-Space")
        for i, name in enumerate(data["names"]):
            ax1.plot(
                out["F"]["tau"]["contents"],
                out["F"][name]["modified"],
                label=f"$F(\\tau)$ ({name})",
                color="C{}".format(i),
            )
        ax1.set_xlabel("$\\tau$")
        ax1.set_ylabel("$F(\\tau)$")
        ax1.legend()
        ax1.grid(True)

        # Plot S on the right subplot
        ax2.set_title("$\\omega$-Space")
        for i, name in enumerate(data["names"]):
            ax2.plot(
                out["S"]["omega"]["modified"],
                out["S"][name]["modified"],
                label=f"$S(\\omega)$ ({name})",
                color="C{}".format(i),
            )
        ax2.set_xlabel("$\\omega$")
        ax2.set_ylabel("$S(\\omega)$")
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(-0.75, 0.75)

        # Adjust layout
        plt.tight_layout()

        # Store plot
        if store:
            # Save the plot to a file with an absolute path
            plt.savefig(self._directory + "/imgs/data_plot.png")

        if giveback:
            return fig
        
        # Show plot
        plt.show()

    def run(self, protocol: bool = True) -> None:
        dat = self._contents["config"]["data"]
        met = self._contents["config"]["method"]
        opt = self._contents["config"]["optim"]
        out = self._contents["output"]

        names = dat["names"]
        models = self.models  # NOTE was deprecated: self._contents["models"]

        methods = []

        for i, met_name in enumerate(met["name"]):
            lambd = met["params"][i]["lambd"]
            method_func =  getattr(pylit.methods, met_name)
            method_row = []
            
            if isinstance(lambd, list) or isinstance(lambd, ARRAY):
                for lambd_ in lambd:
                    params = dict(met["params"][i])
                    params["lambd"] = lambd_
                    params = list(params.values())
                    method_row.append(method_func(*params))
            else:
                method_row = method_func(*list(met["params"][i].values()))

            methods.append(method_row)
        
        # TODO:
        # -> if ["params"][i]["lambda"] is a list, then run experiment multiple times

        if protocol:
            for i, name in enumerate(names):
                print(f"Method for {name}:")
                print("-----------")
                print(methods[i])
                print()

        res = []

        for i, name in enumerate(names):
            res_row = []
            opt_ = getattr(pylit.optimize, opt["name"][i])
            
            if isinstance(methods[i], list):
                for j, method_ in enumerate(methods[i]):
                    res_row.append(
                        opt_(
                            R=models[name].regression_matrix,
                            F=out["F"][name]["modified"],
                            x0=models[name].coeffs,
                            method=method_,
                            maxiter=opt["maxiter"][i],
                            tol=opt["tol"][i],
                        )
                    )
            else:
                res_row = opt_(
                    R=models[name].regression_matrix,
                    F=out["F"][name]["modified"],
                    x0=models[name].coeffs,
                    method=methods[i],
                    maxiter=opt["maxiter"][i],
                    tol=opt["tol"][i],
                )
            res.append(res_row)

        # res = [
        #     getattr(pylit.optimize, opt["name"][i])(
        #         R=models[name].regression_matrix,
        #         F=out["F"][name]["modified"],
        #         x0=models[name].coeffs,
        #         method=methods[i],
        #         maxiter=opt["maxiter"][i],
        #         tol=opt["tol"][i],
        #     )
        #     for i, name in enumerate(names)
        # ]

        self._contents["res"] = res

        if protocol:
            for i, name in enumerate(names):

                if isinstance(res[i], Solution):
                    self._contents["report"].append(
                        f"Result for {name}:\n------------------------\n"
                        + pylit.utils.print_str_dict(pylit.utils.to_dict(res[i]))
                        + "\n"
                    )
                else:
                    for j, res_ in enumerate(res[i]):
                        self._contents["report"].append(
                            f"{j}-Result for {name}:\n------------------------\n"
                            + pylit.utils.print_str_dict(pylit.utils.to_dict(res_))
                            + "\n"
                        )
                print(self._contents["report"][-1])
            # 4 + len(names)

    def assign_coeffs(self):
        dat = self._contents["config"]["data"]
        res = self._contents["res"]
        names = dat["names"]

        for i, name in enumerate(names):
            if isinstance(res[i], Solution):
                self.models[name].coeffs = res[i].x
            else:
                self.models[name].coeffs = np.array([res_.x for res_ in res[i]])

    def plotly_coeffs(self, store: bool = True):
        names = self._contents["config"]["data"]["names"]
        models = self.models

        # Create subplots
        figs = []

        # Plot each model on its own subplot
        for i, name in enumerate(names):
            fig = go.Figure()
            coeffs = models[name].coeffs
            print(f"shapely: {coeffs.shape}")

            if len(coeffs.shape) == 1:
                n = len(coeffs)
                fig.add_trace(go.Scatter(
                    x=np.arange(n),
                    y=coeffs,
                    mode='lines',
                    name=f"Model {name}",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i])
                ))

            else:
                for j, coeffs_ in enumerate(coeffs):
                    m = len(coeffs_)
                    fig.add_trace(go.Scatter(
                        x=np.arange(m),
                        y=coeffs_,
                        mode='lines',
                        name=f"Model {name} - {j}",
                        line=dict(color=pylit.ui.settings.PLOTLY_COLORS[j])
                    ))

            fig.update_layout(
                title=f"Coefficients for Model {name}",
                xaxis=dict(title="Index"),
                yaxis=dict(title="Value"),
            )
            figs.append(fig)

        # Store plot
        if store:
            for i, fig in enumerate(figs):
                fig.write_html(f"{self._directory}/imgs/coeffs_plot_{names[i]}.html")
        
        return figs
    
    def plot_coeffs(self, store: bool = True, giveback=False):
        # TODO Adapt to multiple lambdas !

        names = self._contents["config"]["data"]["names"]
        models = self.models # NOTE deprecated:self._contents["models"]

        # Determine the number of rows and columns for the subplot grid
        num_models = len(names)
        num_rows = (num_models + 1) // 2  # Round up to the nearest integer
        num_cols = 2

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Plot each model on its own subplot
        for i, name in enumerate(names):
            ax = axes[i]
            ax.plot(models[name].coeffs)
            ax.set_title(f"Model {name}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")

        # Hide any unused subplots
        for j in range(num_models, num_rows * num_cols):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()

        # Store plot
        if store:
            # Save the plot to a file with an absolute path
            plt.savefig(self._directory + "/imgs/coeffs_plot.png")

        if giveback:
            return fig
        
        # Show plot
        plt.show()

    def follow_up(self, protocol: bool = True):
        # TODO: Protocol?
        out = self._contents["output"]
        names = self._contents["config"]["data"]["names"]
        models = self.models # NOTE deprecated: self._contents["models"]

        vals = dict(zip(names, [models[name].forward() for name in names]))
        omega_ext = out["S"]["omega"]["modified"]
        evals = dict(zip(names, [models[name](omega_ext) for name in names]))

        self._contents["output"]["values"] = vals
        self._contents["output"]["evaluates"] = evals

    def plotly_results(self, store: bool = True):
        names = self._contents["config"]["data"]["names"]
        F = self._contents["output"]["F"]
        S = self._contents["output"]["S"]
        vals = self._contents["output"]["values"]
        evals = self._contents["output"]["evaluates"]

        # Create a Plotly figure
        fig1 = go.Figure()
        fig2 = go.Figure()
        # Iterate over the data names
        for i, name in enumerate(names):
            model_vals = vals[name]
            model_evals = evals[name]

            # Plot S on the right subplot
            fig2.add_trace(go.Scatter(
                x=S["omega"]["modified"],
                y=S[name]["modified"],
                mode='lines',
                name=f"S(ω) ({name})",
                line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
            ))

            # Plot F on the left subplot
            x_tau = F["tau"]["contents"]
            x_omega = S["omega"]["modified"]

            if len(model_evals.shape) == 1:
                eps = np.abs(model_vals - F[name]["modified"])

                fig1.add_trace(go.Scatter(
                    x=x_tau,
                    y=eps,
                    mode='lines',
                    name=f"ε(τ) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
                ))
                
                fig2.add_trace(go.Scatter(
                        x=x_omega,
                        y=model_evals,
                        mode='lines',
                        name=f"M(ω) ({name})",
                        line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i], dash='dash'),
                    ))
                
            else:
                max_evals = np.max(model_evals, axis=0)
                min_evals = np.min(model_evals, axis=0)
                avg_evals = np.mean(model_evals, axis=0)

                eps = np.array([np.abs(model_vals_ - F[name]["modified"]) for model_vals_ in model_vals]).T # NOTE CHECK ...
                max_eps = np.max(eps, axis=0)
                min_eps = np.min(eps, axis=0)
                avg_eps = np.mean(eps, axis=0)

                fig1.add_trace(go.Scatter(
                    x=x_omega,
                    y=min_eps,
                    mode='lines',
                    name=f"εₗ(τ) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
                    showlegend=True,
                ))

                fig1.add_trace(go.Scatter(
                    x=x_omega,
                    y=max_eps,
                    fill='tonexty',
                    mode='lines',
                    name=f"εᵤ(τ) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
                    showlegend=True,
                ))

                fig1.add_trace(go.Scatter(
                    x=x_omega,
                    y=avg_eps,
                    mode='lines',
                    name=f"εₐ(τ) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i], dash='dash'),
                    showlegend=True,
                ))

                fig2.add_trace(go.Scatter(
                    x=x_omega,
                    y=min_evals,
                    mode='lines',
                    name=f"Mₗ(ω) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
                    showlegend=True,
                ))

                fig2.add_trace(go.Scatter(
                    x=x_omega,
                    y=max_evals,
                    fill='tonexty',
                    mode='lines',
                    name=f"Mᵤ(ω) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i]),
                    showlegend=True,
                ))

                fig2.add_trace(go.Scatter(
                    x=x_omega,
                    y=avg_evals,
                    mode='lines',
                    name=f"Mₐ(ω) ({name})",
                    line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i], dash='dash'),
                    showlegend=True,
                ))

        # Update layout
        fig1.update_layout(
            title="τ-Space",
            xaxis=dict(title="τ"),
            yaxis=dict(title="ε(τ)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        fig2.update_layout(
            title="ω-Space",
            xaxis=dict(title="ω"),
            yaxis=dict(title="S(ω)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        # Store plot
        if store:
            fig1.write_html(f"{self._directory}/imgs/results_plot_F.html")
            fig2.write_html(f"{self._directory}/imgs/results_plot_S.html")

        return [fig1, fig2]

    def plot_results(self, store: bool = True, giveback=False):
        names = self._contents["config"]["data"]["names"]
        F = self._contents["output"]["F"]
        S = self._contents["output"]["S"]
        vals = self._contents["output"]["values"]
        evals = self._contents["output"]["evaluates"]

        # Set figure size and create subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), sharey=False, sharex=False
        )

        # Plot F on the left subplot
        ax1.set_title("$\\tau$-Space")
        for i, name in enumerate(names):
            eps = np.abs(vals[name] - F[name]["modified"])
            ax1.plot(
                F["tau"]["contents"],
                eps,
                label=f"$\\varepsilon(\\tau)$ ({name})",
                color="C{}".format(i),
            )
        ax1.set_xlabel("$\\tau$")
        ax1.set_ylabel("$\\varepsilon(\\tau)$")
        ax1.legend()
        ax1.grid(True)

        # Plot S on the right subplot
        ax2.set_title("$\\omega$-Space")
        for i, name in enumerate(names):
            ax2.plot(
                S["omega"]["modified"],
                S[name]["modified"],
                label=f"$S(\\omega)$ ({name})",
                color="C{}".format(i),
            )
            ax2.plot(
                S["omega"]["modified"],
                evals[name],
                label=f"$M(\\omega)$ ({name})",
                color="C{}".format(i),
                linestyle="--",
            )
        ax2.set_xlabel("$\\omega$")
        ax2.set_ylabel("$S(\\omega)$")
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(-0.5, 0.75)

        # Adjust layout
        plt.tight_layout()

        # Store plot
        if store:
            # Save the plot to a file with an absolute path
            plt.savefig(self._directory + "/imgs/results_plot.png")

        if giveback:
            return fig
        
        # Show plot
        plt.show()

    def compute_integrals(self, protocol: bool = True):
        out = self._contents["output"]
        names = self._contents["config"]["data"]["names"]
        S = out["S"]
        evals = out["evaluates"]

        # Compute integrals
        integrals = dict(
            zip(
                names,
                [
                    np.trapz(evals[name], S["omega"]["modified"])
                    for name in names
                ],
            )
        )

        # Add integrals to output
        self._contents["output"]["integrals"] = integrals

        if protocol:
            self._contents["report"].append(
                "Integrals:\n------------------------\n"
            )
            print(self._contents["report"][-1])
            for i, name in enumerate(names):
                self._contents["report"].append(
                    f"{name}: {integrals[name]}" + "\n"
                )
                print(self._contents["report"][-1])

    def mkdir(self):
        # Define the directory path
        directory = self._path + "/" + self._contents["config"]["name"]
        self._directory = directory
        print(f"Data will be stored in {directory}.\n")

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(directory + "/imgs"):
                os.makedirs(directory + "/imgs")

            return False
        warnings.warn(
            f"Directory '{directory}' already exists.", category=Warning
        )
        return True

    def store(self):
        # Structure
        if not pylit.ui.utils.check_structure(
            self._contents, pylit.experiments.rlrm_es.es_contents
        ):
            warnings.warn(
                "Something went wrong. The contents structure is invalid.",
                category=Warning,
            )
            raise ValueError(
                "Invalid contents structure. Please check es_contents in rlrm_es.py."
            )

        data = dict(self._contents)
        data["models"] = {
            key: {**next((item for item in data["models"] if item["name"] == key), None),
                **{_attr: getattr(self.models[key], _attr) for _attr in self._model_attrs}} for key in data["config"]["data"]["names"] # TODO: Match dict formats !!!! / data["models"]
        }
        data_res = []
        for item in data["res"]:
            data_res_row = []
            if isinstance(item, Solution):
                item_dict = pylit.utils.to_dict(item)
                del item_dict["class_name"]
                data_res_row = dict(item_dict)
            else:
                for sub_item in item:
                    sub_item_dict = pylit.utils.to_dict(sub_item)
                    del sub_item_dict["class_name"]
                    data_res_row.append(sub_item_dict)
            data_res.append(data_res_row)
        data["res"] = data_res

        data = pylit.utils.convert_np_arrays_to_lists(data)

        # Convert the JSON data to a string
        json_config = json.dumps(data["config"])
        json_models = json.dumps(data["models"])
        json_output = json.dumps(data["output"])
        json_report = json.dumps(data["report"])
        json_res = json.dumps(data["res"])

        file_path = self._directory + "/"

        # Write compressed data to a file
        with open(file_path + "config.json", "w") as f:
            f.write(json_config)
        
        with open(file_path + "models.json", "w") as f:
            f.write(json_models)
        
        with open(file_path + "output.json", "w") as f:
            f.write(json_output)
        
        with open(file_path + "report.json", "w") as f:
            f.write(json_report)

        with open(file_path + "res.json", "w") as f:
            f.write(json_res)

    def load(self):
        # Define the directory path
        directory = self._path + "/" + self._contents["config"]["name"]
        self._directory = directory
        print(f"Data will be loaded from {directory}.\n")

        # Load the data from a file
        with open(directory + "/data.json", "r") as f:
            json_data = f.read()

        # Convert the JSON string to a dictionary
        data = json.loads(json_data)

        # Convert
        # TODO

        # Assign the data to the contents attribute
        self.contents = data

    def create_report(self):
        
        # content = "".join(self._contents["report"])
        
        # pylit.ui.utils.create_word_doc(
        #     output_file=self._directory + "/report.docx",
        #     title="Experiment Report",
        #     content=content,
        # )
        
        # TODO: ACTIVATE PDF OPTION !
        #pylit.ui.utils.convert_word_to_pdf(
        #    self._directory + "/report.docx", self._directory + "/report.pdf"
        #)

        pass

    def preprocessing(self, protocol: bool = True, plots: bool = True, store: bool = True, force=False, giveback=False, interactive=False):
        if store:
            feedback = self.mkdir()
            if feedback and not force:
                raise Exception(
                    "Complete run was aborted -- set force to True."
                )
        self.prepare(protocol=protocol)

        fig_data = None
        if interactive and plots:
            fig_data = self.plotly_data(store=store)
        elif not interactive and plots:
            fig_data = self.plot_data(store=store, giveback=giveback)

        if giveback:
            return fig_data
    
    def fit_model(self, get_models: callable, protocol: bool = True, plots: bool = True, store: bool = True, docx=True, giveback=False, interactive=False):
        self.set_models(get_models)
        self.run(protocol=protocol)
        self.assign_coeffs()

        fig_coeffs = None
        if interactive and plots:
            fig_coeffs = self.plotly_coeffs(store=store)
        elif not interactive and plots:
            fig_coeffs = self.plot_coeffs(store=store, giveback=giveback)

        self.follow_up(protocol=protocol)

        fig_results = None
        if interactive and plots:
            fig_results = self.plotly_results(store=store)
        elif not interactive and plots:
            fig_results = self.plot_results(store=store, giveback=giveback)

        self.compute_integrals(protocol=protocol)

        if store:
            self.store()
        if docx:
            self.create_report()
        if giveback:
            return fig_coeffs, fig_results

    def complete_run(
        self,
        get_models: callable,
        protocol: bool = True,
        plots: bool = True,
        store: bool = True,
        force=False,
        docx=True,
        giveback=False,
        interactive=False,
    ):
         
        fig_data = self.preprocessing(
            protocol=protocol,
            plots=plots,
            store=store,
            force=force,
            giveback=giveback,
            interactive=interactive)
        
        fig_coeffs, fig_results = self.fit_model(
            get_models=get_models,
            protocol=protocol,
            plots=plots,
            store=store,
            docx=docx,
            giveback=giveback)
        
        if giveback:
            return self._contents["report"], fig_data, fig_coeffs, fig_results


if __name__ == "__main__":
    pass

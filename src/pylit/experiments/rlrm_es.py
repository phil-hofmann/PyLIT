import pylit
from pylit.optimize import Solution
from typing import List

NAME_PARAMS = {
    "name": (List, str),
    "params": (List, dict),
}

# Config

es_contents_config_data_scale = {
    "F": bool,
    "Pos_S": bool,
    "S": bool,
    "Trapz_S": bool,
}

es_contents_config_data = {
    "names": (List, str),
    "noise_active": (List, bool),
    "noise": (List, NAME_PARAMS),
    "noise_conv_active": (List, bool),
    "noise_conv": (List, NAME_PARAMS),
    "scale": (List, es_contents_config_data_scale),
    "file_path_F": str,
    "file_path_S": str,
}

es_contents_config_method = {
    "name": (List, str),
    "params": (List, dict),
}

es_contents_config_optim = {
    "name": (List, str),
    "maxiter": (List, int),
    "tol": (List, float),
}

es_contents_config = {
    "name": str,
    "data": (dict, es_contents_config_data),
    "method": (dict, es_contents_config_method),
    "optim": (dict, es_contents_config_optim),
}

# Models

es_contents_models_scaling = {
    "name": str,
    "params": dict,
}

es_contents_models_model = {
    "name": str,
    "params": dict,
}

es_contents_models = {
    "name": str,
    "scaling": (dict, es_contents_models_scaling),
    "model": (dict, es_contents_models_model),
}

# Output

es_contents_output = {
    "tau_max": float,
    "tau_min": float,
    "omega_max": float,
    "omega_min": float,
    "F": dict,
    "S": dict
}

# Contents

es_contents = {
    "config": (dict, es_contents_config),
    "models": (List, es_contents_models),
    "output": (dict, es_contents_output),
    "report": (List, str),
    "res": (List, Solution.es),
}

if __name__ == "__main__":

    # Example usage

    example = {
        "config": {
            "name": "example",
            "data": {
                "names": ["name1"],
                "noise_active": [True],
                "noise": [{"name":"WhiteNoise", "params": {"mean": 0.0, "std": 1.0}}],
                "noise_conv_active": [True],
                "noise_conv": [],
                "scale": [{
                    "max_F": True,
                    "pos_S": True,
                    "ext_S": True,
                    "trapz_S": True,
                }],
                "file_path_F": pylit.ui.settings.PATH_DATA + "/raw_F/fileF.dat",
                "file_path_S": pylit.ui.settings.PATH_DATA + "/raw_S/fileS.dat",
            },
            "method": {
                "name": ["method1"],
                "params": [{"param1": 1}]
            },
            "optim": {
                "name": ["optim1"],
                "maxiter": [100],
                "tol": [0.001]
            },
        },
        "models": [
            {
                "name": "model1",
                "scaling": {"name": "scaling1", "params": {"param1": 1}},
                "model": {"name": "model1", "params": {"param1": 1}},
            }
        ],
        "output": {
            "tau_max": 0.1,
            "tau_min": 0.01,
            "omega_max": 1.0,
            "omega_min": 0.1,
            "F": {},
            "S": {},
        },
        "report": ["report1", "report2"],
        "res": [pylit.optimize.solution.example().to_dict()]
    }
    print(pylit.ui.utils.check_structure(example, es_contents))  # Output: True
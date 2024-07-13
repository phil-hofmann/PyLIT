from typing import List
from pylit.frontend.core.param import Param
from pylit.global_settings import INT_DTYPE, FLOAT_DTYPE

class ParamMap:
    def __init__(self, params: List[Param]):
        self.param_dict = {param.name: param for param in params}

    def __getitem__(self, key: str) -> Param:
        return self.param_dict[key]

    def __setitem__(self, key: str, value: Param):
        self.param_dict[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.param_dict

    def keys(self) -> List[str]:
        return list(self.param_dict.keys())

    def values(self) -> List[Param]:
        return list(self.param_dict.values())

    def items(self) -> List[tuple]:
        return list(self.param_dict.items())

    def insert_values(self, values: dict):
        for key, value in values.items():
            if key in self:
                self[key].value = value
        return self

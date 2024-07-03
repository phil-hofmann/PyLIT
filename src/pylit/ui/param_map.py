from typing import Any, Optional, List

class Param:
    def __init__(self, name: str, 
                 label: Optional[str] = None,
                 type: Optional[type] = None, 
                 default: Optional[Any] = None, 
                 min_value: Optional[Any] = None, 
                 max_value: Optional[Any] = None, 
                 step: Optional[Any] = None,
                 optional: Optional[bool] = False,
                 optional_label: Optional[str] = None,
                 ignore: bool = False):
        self.name = name
        self.label = label
        self.type = type
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.optional = optional
        self.optional_label = optional_label
        self.ignore = ignore

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
from typing import Any, Optional
from pylit.frontend.constants import (
    SCI_NUM_STEP,
    NUM_STEP,
    DEFAULT_INT,
    DEFAULT_FLOAT,
    DEFAULT_ARRAY,
    DEFAULT_ARRAY_UPPER,
    DEFAULT_ARRAY_LOWER,
    DEFAULT_ARRAY_NUM,
)
from pylit.global_settings import INT_DTYPE, FLOAT_DTYPE, ARRAY


class Param:
    def __init__(
        self,
        name: str,
        label: Optional[str] = None,
        my_type: Optional[type] = None,
        default: Optional[Any] = None,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        step: Optional[Any] = None,
        variation: Optional[bool] = False,
        ignore: Optional[bool] = False,
    ):
        self.name = name
        self._label = label
        self._my_type = my_type
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.variation = variation
        self.ignore = ignore
        self.value = None
    
    @property
    def label(self):
        return self._label if self._label is not None else self.name
    
    @label.setter
    def label(self, new_label):
        self._label = new_label

    @property
    def my_type(self):
        if self._my_type is not None:
            return self._my_type
        raise ValueError(f"This parameter has no declared type - it is {self._my_type}!")

    @my_type.setter
    def my_type(self, new_my_type: type):
        self._my_type = new_my_type

    @property
    def default(self):
        if self._default is not None:
            return self._default
        elif self._my_type is int:
            return int(DEFAULT_INT)
        elif self._my_type is INT_DTYPE:
            return INT_DTYPE(0)
        elif self._my_type is float:
            return float(DEFAULT_FLOAT)
        elif self._my_type is FLOAT_DTYPE:
            return FLOAT_DTYPE(DEFAULT_FLOAT)
        elif self._my_type is list:
            return list(DEFAULT_ARRAY)
        elif self._my_type is ARRAY:
            return ARRAY(DEFAULT_ARRAY)
        # "This parameter has no default value and there is also no standard default value assigned!"
        return None

    @default.setter
    def default(self, new_default):
        self._default = new_default

    @property
    def value(self):
        return self.default if self._value is None else self.my_type(self._value)

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def step(self):
        if self._my_type is None and self._step is not None:
            return self._step

        elif self._my_type is None and self._step is None:
            raise ValueError(
                "This Parameter has no step and it couldn't be inferred because the type is not defined."
            )

        elif self._my_type is not None and self._step is None:
            if self._my_type is int:
                return int(NUM_STEP)
            elif self._my_type is INT_DTYPE:
                return INT_DTYPE(NUM_STEP)
            elif self._my_type is float:
                return float(SCI_NUM_STEP)
            elif self.my_type is FLOAT_DTYPE:
                return FLOAT_DTYPE(SCI_NUM_STEP)

        if (
            self._my_type is not None
            and self._step is not None
            and type(self._step) is not self.my_type
        ):
            return self.my_type(self._step)

        return self._step

    @step.setter
    def step(self, new_step):
        self._step = new_step

    @property
    def format(self):
        if self._my_type is FLOAT_DTYPE or self._my_type is float:
            return "%f"
        elif self._my_type is INT_DTYPE or self._my_type is int:
            return "%d"
        raise ValueError(f"There is no format for the type: {self._my_type}")
    
    @property
    def attributes(self):
        attrs = {
            "label": self.label,
            "value": self.value,
            "step": self.step,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "format": self.format,
        }
        not_none_attrs = {k:v for k, v in attrs.items() if v is not None}
        return not_none_attrs
    

    def insert_value(self, value):
        self.value = value

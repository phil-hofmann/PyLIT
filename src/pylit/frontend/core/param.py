import numpy as np
from typing import Any, Optional
from pylit.frontend.constants import (
    SCI_NUM_STEP,
    NUM_STEP,
    DEFAULT_INT,
    DEFAULT_FLOAT,
    DEFAULT_UPPER_VALUE,
    DEFAULT_LOWER_VALUE,
    DEFAULT_NUM_VALUE,
)
from pylit.global_settings import INT_DTYPE, FLOAT_DTYPE, ARRAY


class Param:
    def __init__(
        self,
        name: str,
        label: Optional[str] = None,
        description: Optional[str] = None,
        my_type: Optional[type] = None,
        default: Optional[Any] = None,
        step: Optional[Any] = None,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        lower_value: Optional[Any] = None,
        upper_value: Optional[Any] = None,
        num_value: Optional[Any] = None,
        variation: Optional[bool] = False,
        ignore: Optional[bool] = False,
    ):
        self.name = name
        self._label = label
        self.description = description
        self._my_type = my_type
        self.default = default
        self.step = step
        self.min_value = min_value
        self.max_value = max_value
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.num_value = num_value
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
        raise ValueError(f"This parameter '{self.name}' has no declared type.")

    @my_type.setter
    def my_type(self, new_my_type: type):
        self._my_type = new_my_type

    @property
    def default(self):
        if self._default is not None and self._my_type is not ARRAY and self._my_type is not list:
            return self._default
        elif self._my_type is ARRAY or self._my_type is list: # TODO no default needed, change handling a bit?!?
            return [
                self.lower_value if self.lower_value is not None else DEFAULT_LOWER_VALUE,
                self.upper_value if self.upper_value is not None else DEFAULT_UPPER_VALUE,
                self.num_value if self.num_value is not None else DEFAULT_NUM_VALUE,
            ]
        elif self._my_type is int:
            return int(DEFAULT_INT)
        elif self._my_type is INT_DTYPE:
            return INT_DTYPE(0)
        elif self._my_type is float:
            return float(DEFAULT_FLOAT)
        elif self._my_type is FLOAT_DTYPE:
            return FLOAT_DTYPE(DEFAULT_FLOAT)
        # "This parameter has no default value and there is also no standard default value assigned!"
        return None

    @default.setter
    def default(self, new_default):
        self._default = new_default

    @property
    def value(self):
        return self.default if self._value is None else self._value

    @value.setter
    def value(self, new_value):
        # TODO somehow do automatic type setting here?!?
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
            # NOTE: These attributes are not allowed for st.number_input
            # "lower_value": self.lower_value, 
            # "upper_value": self.upper_value,
            # "num_value": self.num_value,
            "format": self.format,
        }
        not_none_attrs = {k:v for k, v in attrs.items() if v is not None}
        return not_none_attrs
    

    def insert_value(self, value):
        # if self._my_type is None:
        #     raise ValueError("The type of the parameter is not defined.") # TODO other handling?
        if type(value) not in [ARRAY, list]:
            self.my_type = type(value)
            self.value = value
        else:
            # NOTE Only checks for linspace
            # TODO Check for other types or move somewhere else?!?
            # self.my_type = ARRAY # NOTE If we uncomment here, the variation of lambda machanism is affected...
            num_value = len(value)
            lower_value = np.min(value)
            upper_value = np.max(value)
            linspace = np.linspace(lower_value, upper_value, num_value)
            if np.array_equal(value, linspace):
                self.num_value = num_value
                self.lower_value = lower_value
                self.upper_value = upper_value
                self.value = value # TODO NOTE CHECK THAT!!!
                # self.value = [lower_value, upper_value, num_value] # NOTE...wrong!
            else:
                raise ValueError("The given sequence is not a linspace sequence.")

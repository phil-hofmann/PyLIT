from typing import List

class Option:
        def __init__(self, name: str, ref: str):
            self.name = name
            self.ref = ref
    
        @property
        def name(self):
            return self._name
        
        @name.setter
        def name(self, name: str):
            if not isinstance(name, str):
                raise ValueError("The name must be a string.")
            self._name = name
    
        @property
        def ref(self):
            return self._ref
        
        @ref.setter
        def ref(self, ref: str):
            if not isinstance(ref, str):
                raise ValueError("The reference must be a string.")
            self._ref = ref

        def __eq__(self, __value: str) -> bool:
            return self.name == __value or self.ref == __value

class Options:

    def __init__(self, options: List[Option]):
        self.options = options

    def find(self, value: str):
        for opt in self.options:
            if opt == value:
                return opt
                
        raise Exception(f"Option {value} not found.")

    def index_of(self, value: str):
        for i, opt in enumerate(self.options):
            if opt == value:
                return i
                
        return 0

    def __call__(self, name:bool=False, ref:bool=False):
        if (name and ref) or (not name and not ref):
            return self.options
        if name:
            return [option.name for option in self.options]
        if ref:
            return [option.ref for option in self.options]
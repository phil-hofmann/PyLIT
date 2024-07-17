from typing import List
from pylit.frontend.core import Option

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
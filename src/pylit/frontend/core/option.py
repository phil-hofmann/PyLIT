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
                raise ValueError("The reference must be a string.") # TODO: unreachable?
            self._ref = ref

        def __eq__(self, __value: str) -> bool:
            return self.name == __value or self.ref == __value

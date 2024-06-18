from typing import Any, Dict, Optional, Protocol

class BaseClass(Protocol):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...
    
class InheritClass(BaseClass):

    def __init__(self) -> None:
        ...


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...
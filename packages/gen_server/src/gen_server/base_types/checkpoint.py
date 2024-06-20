from .architecture import Architecture
from typing import Any
from typing_extensions import override
from dataclasses import dataclass, asdict, field, fields
import datetime


@dataclass
class CheckpointMetadata:
    """
    This dataclass contains metadata describing a checkpoint file. It's used by Comfy-Creator's front
    end to sort and display checkpoint files.
    """
    display_name: str
    category: str
    author: str
    file_type: str
    file_path: str
    components: dict[str, Architecture] = field(default_factory=dict)
    date_modified: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def serialize(self) -> dict[str, Any]:
        # Serialize all fields except 'components', 'date_modified', and 'file_path'
        serialized_data = asdict(
            self,
            dict_factory=lambda fields: { 
                key: value for key, value in fields 
                if key not in ['components', 'date_modified', 'file_path'] 
            }
        )
        
        # better formatting for date_modified
        serialized_data.update({'date_modified': self.date_modified.strftime("%Y-%m-%d %H:%M:%S")})
        
        # Manually serialize the Architecture components
        serialized_components = {}
        for name, component in self.components.items():
            try:
                serialized_components[name] = component.serialize()
            except Exception:
                continue # If serialization fails, skip this component
        serialized_data.update({'components': serialized_components})
        
        return serialized_data
    
    @override
    def __str__(self) -> str:
        return str(self.serialize())


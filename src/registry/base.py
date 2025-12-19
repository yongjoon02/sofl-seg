"""Base Registry class with metadata support.

Usage:
    # 1. 데코레이터로 등록
    @MODEL_REGISTRY.register(name='my_model')
    class MyModel:
        pass
    
    # 2. 메타데이터와 함께 등록
    @MODEL_REGISTRY.register(name='my_model', metadata={'task': 'supervised', 'params': 1000})
    class MyModel:
        pass
    
    # 3. 함수 호출로 등록
    MODEL_REGISTRY.register(name='my_model', obj=MyModel, metadata={...})
    
    # 4. 조회
    model_cls = MODEL_REGISTRY.get('my_model')
    model_info = MODEL_REGISTRY.get_metadata('my_model')
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type


@dataclass
class RegistryEntry:
    """Registry entry with object and metadata."""
    name: str
    obj: Any  # Registered class or function
    metadata: Dict[str, Any] = field(default_factory=dict)


class Registry:
    """Generic registry with metadata support.
    
    Features:
    - Decorator-based registration
    - Metadata support for each entry
    - Suffix support for namespacing
    - Iteration and filtering
    """
    
    def __init__(self, name: str):
        """Initialize registry.
        
        Args:
            name: Registry name (for error messages)
        """
        self._name = name
        self._entries: Dict[str, RegistryEntry] = {}
    
    def _do_register(
        self,
        name: str,
        obj: Any,
        suffix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal registration method."""
        if isinstance(suffix, str):
            name = f"{name}_{suffix}"
        
        # Allow re-registration (for module reloading)
        if name in self._entries:
            pass  # Silently overwrite
        
        self._entries[name] = RegistryEntry(
            name=name,
            obj=obj,
            metadata=metadata or {}
        )
    
    def register(
        self,
        name: Optional[str] = None,
        obj: Any = None,
        suffix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Register an object (class or function).
        
        Can be used as decorator or function call.
        
        Args:
            name: Registration name (default: class/function name)
            obj: Object to register (if not using as decorator)
            suffix: Optional suffix for namespacing
            metadata: Optional metadata dict
        
        Returns:
            Decorator function or None (if obj provided)
        
        Examples:
            # As decorator
            @REGISTRY.register(name='my_class')
            class MyClass:
                pass
            
            # As decorator with metadata
            @REGISTRY.register(name='my_class', metadata={'version': '1.0'})
            class MyClass:
                pass
            
            # As function call
            REGISTRY.register(name='my_class', obj=MyClass, metadata={...})
        """
        if obj is None:
            # Used as decorator
            def decorator(func_or_class: Any) -> Any:
                register_name = name if name is not None else func_or_class.__name__
                self._do_register(register_name, func_or_class, suffix, metadata)
                return func_or_class
            return decorator
        
        # Used as function call
        register_name = name if name is not None else obj.__name__
        self._do_register(register_name, obj, suffix, metadata)
    
    def get(self, name: str, suffix: Optional[str] = None) -> Any:
        """Get registered object by name.
        
        Args:
            name: Registration name
            suffix: Optional suffix to try if name not found
        
        Returns:
            Registered object
        
        Raises:
            KeyError: If name not found
        """
        entry = self._entries.get(name)
        
        if entry is None and suffix:
            suffixed_name = f"{name}_{suffix}"
            entry = self._entries.get(suffixed_name)
            if entry:
                print(f"[Registry] '{name}' not found, using '{suffixed_name}'")
        
        if entry is None:
            available = ', '.join(self._entries.keys())
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry. "
                f"Available: {available}"
            )
        
        return entry.obj
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for registered object.
        
        Args:
            name: Registration name
        
        Returns:
            Metadata dict
        """
        if name not in self._entries:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry.")
        return self._entries[name].metadata
    
    def get_entry(self, name: str) -> RegistryEntry:
        """Get full registry entry (object + metadata).
        
        Args:
            name: Registration name
        
        Returns:
            RegistryEntry with obj and metadata
        """
        if name not in self._entries:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry.")
        return self._entries[name]
    
    def list(self, **filter_kwargs) -> List[str]:
        """List registered names, optionally filtered by metadata.
        
        Args:
            **filter_kwargs: Metadata key-value pairs to filter by
        
        Returns:
            List of matching names
        
        Example:
            MODEL_REGISTRY.list(task='supervised')  # All supervised models
        """
        if not filter_kwargs:
            return list(self._entries.keys())
        
        result = []
        for name, entry in self._entries.items():
            match = all(
                entry.metadata.get(k) == v
                for k, v in filter_kwargs.items()
            )
            if match:
                result.append(name)
        return result
    
    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._entries
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        return iter(self._entries.keys())
    
    def __len__(self) -> int:
        """Return number of registered objects."""
        return len(self._entries)
    
    def keys(self) -> List[str]:
        """Return all registered names."""
        return list(self._entries.keys())
    
    def values(self) -> List[Any]:
        """Return all registered objects."""
        return [entry.obj for entry in self._entries.values()]
    
    def items(self) -> List[Tuple[str, Any]]:
        """Return all (name, obj) pairs."""
        return [(name, entry.obj) for name, entry in self._entries.items()]
    
    def items_with_metadata(self) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """Return all (name, obj, metadata) tuples."""
        return [
            (name, entry.obj, entry.metadata)
            for name, entry in self._entries.items()
        ]


# =============================================================================
# Global Registry Instances
# =============================================================================

DATASET_REGISTRY = Registry('dataset')
ARCHS_REGISTRY = Registry('archs')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')


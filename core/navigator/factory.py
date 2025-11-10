"""Factory pattern for runtime interface wiring.

This module separates static topology from dynamic behavior, addressing the
security concerns of serializing callables while maintaining flexibility.
"""
from typing import Callable, Optional, Protocol
from .navigator import Navigator


class InterfaceFactory(Protocol):
    """Protocol for interface factories that wire runtime behavior."""
    
    def create_validator(self, interface_name: str) -> Callable[[], bool]:
        """Create a feature validator function for the given interface.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            A callable that returns True if current state matches this interface
        """
        ...
    
    def create_action(self, from_interface: str, to_interface: str) -> Optional[Callable[[], None]]:
        """Create an action function for transitioning between interfaces.
        
        Args:
            from_interface: Source interface name
            to_interface: Destination interface name
            
        Returns:
            A callable that performs the transition, or None if no action needed
        """
        ...


class InterfaceSpec:
    """Specification for an interface without runtime callables.
    
    This is safe to serialize as it contains only data, not code.
    """
    
    def __init__(self, name: str, description: str, action_destinations: list[str]):
        """Initialize interface specification.
        
        Args:
            name: Unique name for this interface
            description: Human-readable description
            action_destinations: List of interface names this can navigate to
        """
        self.name = name
        self.description = description
        self.action_destinations = action_destinations
    
    def to_interface(self, factory: InterfaceFactory) -> Navigator.Interface:
        """Convert to Navigator.Interface using provided factory.
        
        Args:
            factory: Factory to create validators and actions
            
        Returns:
            Navigator.Interface with wired behavior
        """
        features = [factory.create_validator(self.name)]
        actions = {}
        for dest in self.action_destinations:
            action = factory.create_action(self.name, dest)
            if action is not None:
                actions[dest] = action
        
        return Navigator.Interface(
            name=self.name,
            description=self.description,
            features=features,
            actions=actions
        )


def build_interfaces_from_specs(
    specs: list[InterfaceSpec],
    factory: InterfaceFactory
) -> list[Navigator.Interface]:
    """Build Navigator.Interface list from specs using factory.
    
    Args:
        specs: List of interface specifications
        factory: Factory to wire runtime behavior
        
    Returns:
        List of Navigator.Interface objects ready for use
    """
    return [spec.to_interface(factory) for spec in specs]

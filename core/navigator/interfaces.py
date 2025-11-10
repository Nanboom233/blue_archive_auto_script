from functools import partial
from typing import Callable, Optional

import core.color
from .navigator import Navigator
from .factory import InterfaceFactory, InterfaceSpec, build_interfaces_from_specs


class BaasInterfaceFactory:
    """Factory for creating BAAS thread interface validators and actions."""
    
    def __init__(self, baas_thread):
        """Initialize factory with BAAS thread instance.
        
        Args:
            baas_thread: The BAAS thread instance for validation and actions
        """
        self.baas_thread = baas_thread
        self._validators = {
            "main_page": "main_page",
            "group_page": "group_menu",
        }
        self._actions = {
            ("main_page", "group_page"): (565, 648),
            ("group_page", "main_page"): (920, 159),
        }
    
    def create_validator(self, interface_name: str) -> Callable[[], bool]:
        """Create a feature validator for the interface.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            Validator function that checks if current state matches interface
        """
        rgb_feature = self._validators.get(interface_name)
        if rgb_feature is None:
            raise ValueError(f"No validator defined for interface: {interface_name}")
        return partial(core.color.match_rgb_feature, self.baas_thread, rgb_feature)
    
    def create_action(self, from_interface: str, to_interface: str) -> Optional[Callable[[], None]]:
        """Create an action for transitioning between interfaces.
        
        Args:
            from_interface: Source interface name
            to_interface: Destination interface name
            
        Returns:
            Action function or None if no action needed
        """
        coords = self._actions.get((from_interface, to_interface))
        if coords is None:
            return None
        x, y = coords
        return partial(self.baas_thread.click, x, y)


class Interfaces:
    """Interface manager using factory pattern for runtime wiring."""
    
    baas_thread: core.Baas_thread
    factory: BaasInterfaceFactory
    
    # Interface specifications (data only, safe to serialize)
    SPECS = [
        InterfaceSpec(
            name="main_page",
            description="The main menu interface",
            action_destinations=["group_page"]
        ),
        InterfaceSpec(
            name="group_page",
            description="The group page interface",
            action_destinations=["main_page"]
        ),
    ]

    def list_interfaces(self) -> list[Navigator.Interface]:
        """Build interface list with runtime behavior wired.
        
        Returns:
            List of Navigator.Interface objects
        """
        return build_interfaces_from_specs(self.SPECS, self.factory)

    def __init__(self, baas_thread: core.Baas_thread):
        """Initialize interfaces with BAAS thread.
        
        Args:
            baas_thread: The BAAS thread instance
        """
        self.baas_thread = baas_thread
        self.factory = BaasInterfaceFactory(baas_thread)

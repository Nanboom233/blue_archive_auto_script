"""Demonstration of improved navigator with factory pattern and async execution.

This demo shows:
1. How to use InterfaceSpec for safe serialization
2. How to use InterfaceFactory for runtime wiring
3. How to use AsyncNavigator for handling delayed actions
"""
import sys
from pathlib import Path
from functools import partial
import time

# Add root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.navigator import (
    Navigator,
    AsyncNavigator,
    AsyncNavigatorConfig,
    InterfaceSpec,
    InterfaceFactory,
    build_interfaces_from_specs,
)

# Global state to simulate environment
CURRENT_NODE: str = ""
ACTION_DELAY: float = 0.3  # Simulate UI delay


class DemoInterfaceFactory:
    """Factory for creating demo interface validators and actions."""
    
    def create_validator(self, interface_name: str):
        """Create validator that checks global CURRENT_NODE."""
        return lambda: CURRENT_NODE == interface_name
    
    def create_action(self, from_interface: str, to_interface: str):
        """Create action that updates CURRENT_NODE with simulated delay."""
        def action():
            global CURRENT_NODE
            print(f"  [Action] Navigating from {from_interface} to {to_interface}...")
            # Simulate async UI navigation
            time.sleep(ACTION_DELAY)
            CURRENT_NODE = to_interface
        return action


def build_demo_specs() -> list[InterfaceSpec]:
    """Build interface specifications (safe to serialize).
    
    Note: These specs contain only data, no code/callables.
    They can be safely saved and loaded without security concerns.
    """
    return [
        InterfaceSpec(
            name="A1",
            description="Node A1",
            action_destinations=["A2", "B1"]
        ),
        InterfaceSpec(
            name="A2",
            description="Node A2",
            action_destinations=["Gate"]
        ),
        InterfaceSpec(
            name="Gate",
            description="The gate node",
            action_destinations=["B2"]
        ),
        InterfaceSpec(
            name="B1",
            description="Node B1",
            action_destinations=["C1"]
        ),
        InterfaceSpec(
            name="B2",
            description="Node B2",
            action_destinations=[]
        ),
        InterfaceSpec(
            name="C1",
            description="Node C1",
            action_destinations=["B2", "A1"]
        ),
    ]


def demo_factory_pattern():
    """Demonstrate factory pattern for safe serialization."""
    print("\n" + "="*60)
    print("DEMO 1: Factory Pattern for Safe Serialization")
    print("="*60)
    
    # Step 1: Create specs (these are just data, safe to serialize)
    specs = build_demo_specs()
    print(f"\n✓ Created {len(specs)} interface specifications")
    print("  (These specs contain only data and can be safely serialized)")
    
    # Step 2: In production, specs could be saved/loaded
    # For now, we just show they're data-only
    print(f"\n✓ Example spec: {specs[0].name}")
    print(f"  - Description: {specs[0].description}")
    print(f"  - Destinations: {specs[0].action_destinations}")
    
    # Step 3: At runtime, use factory to wire behavior
    factory = DemoInterfaceFactory()
    interfaces = build_interfaces_from_specs(specs, factory)
    print(f"\n✓ Built {len(interfaces)} Navigator.Interface objects using factory")
    print("  (Runtime behavior wired, ready to use)")
    
    # Step 4: Create navigator with wired interfaces
    navigator = Navigator(interfaces)
    print(f"\n✓ Created Navigator with {navigator.metadata.node_volume} nodes")
    
    return navigator


def demo_async_navigation():
    """Demonstrate async navigation with validation."""
    print("\n" + "="*60)
    print("DEMO 2: Async Navigation with Validation")
    print("="*60)
    
    # Setup
    global CURRENT_NODE
    specs = build_demo_specs()
    factory = DemoInterfaceFactory()
    interfaces = build_interfaces_from_specs(specs, factory)
    
    # Create async navigator with custom config
    config = AsyncNavigatorConfig(
        action_delay=0.1,  # Wait after action
        validation_timeout=5.0,  # Max wait for validation
        validation_interval=0.2,  # Check every 200ms
        max_retries=2  # Retry failed actions twice
    )
    navigator = AsyncNavigator(interfaces, config=config)
    
    print("\n✓ Created AsyncNavigator with validation config:")
    print(f"  - Action delay: {config.action_delay}s")
    print(f"  - Validation timeout: {config.validation_timeout}s")
    print(f"  - Max retries: {config.max_retries}")
    
    # Test navigation with async handling
    CURRENT_NODE = "A1"
    print(f"\n✓ Starting at: {CURRENT_NODE}")
    
    print("\n→ Navigating A1 -> B2 (async with validation)...")
    success = navigator.goto_async("B2", log_output=True)
    
    if success:
        print(f"\n✓ Navigation succeeded!")
        print(f"  Current node: {CURRENT_NODE}")
    else:
        print(f"\n✗ Navigation failed!")


def demo_retry_logic():
    """Demonstrate retry logic for flaky actions."""
    print("\n" + "="*60)
    print("DEMO 3: Retry Logic for Flaky Actions")
    print("="*60)
    
    global CURRENT_NODE
    
    # Create a factory with a flaky action that fails first time
    class FlakyFactory(DemoInterfaceFactory):
        def __init__(self):
            super().__init__()
            self.attempt_counts = {}
        
        def create_action(self, from_interface: str, to_interface: str):
            """Create action that fails on first attempt."""
            def action():
                global CURRENT_NODE
                key = (from_interface, to_interface)
                attempts = self.attempt_counts.get(key, 0)
                self.attempt_counts[key] = attempts + 1
                
                print(f"  [Action] Attempt {attempts + 1}: {from_interface} -> {to_interface}")
                
                if attempts == 0:
                    # Fail on first attempt
                    print(f"  [Action] ✗ Simulated failure (attempt {attempts + 1})")
                    # Don't update CURRENT_NODE, stay at source
                    return
                
                # Succeed on retry
                print(f"  [Action] ✓ Success (attempt {attempts + 1})")
                time.sleep(ACTION_DELAY)
                CURRENT_NODE = to_interface
            
            return action
    
    # Setup with flaky factory
    specs = build_demo_specs()
    flaky_factory = FlakyFactory()
    interfaces = build_interfaces_from_specs(specs, flaky_factory)
    
    config = AsyncNavigatorConfig(
        action_delay=0.1,
        validation_timeout=3.0,
        validation_interval=0.2,
        max_retries=3
    )
    navigator = AsyncNavigator(interfaces, config=config)
    
    CURRENT_NODE = "A1"
    print(f"\n✓ Starting at: {CURRENT_NODE}")
    print("\n→ Navigating A1 -> A2 (will fail first attempt)...")
    
    success = navigator.goto_async("A2", log_output=True)
    
    if success:
        print(f"\n✓ Navigation succeeded despite initial failure!")
        print(f"  Current node: {CURRENT_NODE}")
    else:
        print(f"\n✗ Navigation failed even with retries!")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║  Navigator Improvements Demo                            ║")
    print("╚" + "="*58 + "╝")
    
    # Demo 1: Factory pattern
    navigator = demo_factory_pattern()
    
    # Demo 2: Async navigation
    demo_async_navigation()
    
    # Demo 3: Retry logic
    demo_retry_logic()
    
    print("\n" + "="*60)
    print("All demonstrations completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

# Navigator Improvements Documentation

## Overview

This document describes the improvements made to the Navigator system to address two key limitations:

1. **Static Graph vs Dynamic Functions**: Separating serializable topology from runtime behavior
2. **Asynchronous Action Execution**: Handling actions that don't complete instantly

## Problem Statement

### Problem 1: Static Design vs Dynamic Execution

The original navigator design had a tension between:
- **Static graph construction**: Building topology at design time for efficient path planning
- **Dynamic function wiring**: Features and actions require callable functions that can't be safely serialized

This created security concerns when trying to save/load navigator state, as deserializing arbitrary callables is dangerous.

### Problem 2: Instant vs Delayed Actions

The original `goto` method assumed actions complete instantly:
```python
action()  # Assumed to complete immediately
current_id = next_id  # Move to next hop
```

However, in Blue Archive Auto Script:
- UI navigation involves animations and loading screens
- Actions may fail due to transient states
- Need validation to confirm successful navigation

## Solutions

### Solution 1: Factory Pattern for Safe Serialization

We introduce a clean separation between static topology and runtime behavior:

#### InterfaceSpec (Data Only - Safe to Serialize)

```python
from core.navigator import InterfaceSpec

spec = InterfaceSpec(
    name="main_page",
    description="The main menu interface",
    action_destinations=["group_page", "settings_page"]
)
```

**Key Points:**
- Contains only data (strings, lists)
- Safe to serialize with pickle, JSON, etc.
- No security risks

#### InterfaceFactory (Runtime Behavior)

```python
from core.navigator import InterfaceFactory

class MyInterfaceFactory:
    def create_validator(self, interface_name: str):
        """Create function that checks if we're at this interface"""
        return lambda: check_ui_state(interface_name)
    
    def create_action(self, from_interface: str, to_interface: str):
        """Create function that navigates from -> to"""
        return lambda: perform_navigation(from_interface, to_interface)
```

**Key Points:**
- Creates callables at runtime
- No serialization needed
- Dependency injection pattern

#### Building Interfaces

```python
from core.navigator import build_interfaces_from_specs

# Load specs (from file, database, etc.)
specs = load_specs()

# Wire runtime behavior
factory = MyInterfaceFactory()
interfaces = build_interfaces_from_specs(specs, factory)

# Use with navigator
navigator = Navigator(interfaces)
```

### Solution 2: AsyncNavigator with Validation

We introduce `AsyncNavigator` that handles delayed action completion:

#### Basic Usage

```python
from core.navigator import AsyncNavigator, AsyncNavigatorConfig

# Configure async behavior
config = AsyncNavigatorConfig(
    action_delay=0.5,          # Wait after action before validation
    validation_timeout=10.0,    # Max wait for successful validation
    validation_interval=0.5,    # Check every 500ms
    max_retries=3              # Retry failed actions up to 3 times
)

# Create navigator
navigator = AsyncNavigator(interfaces, config=config)

# Navigate with validation
success = navigator.goto_async("target_interface")
```

#### How It Works

1. **Execute Action**
   ```python
   action()  # Perform navigation (e.g., click button)
   ```

2. **Wait for Action to Take Effect**
   ```python
   time.sleep(config.action_delay)  # Let UI respond
   ```

3. **Validation Loop**
   ```python
   while time_elapsed < validation_timeout:
       current = resolve_current_interface()
       if current == target:
           return True  # Success!
       elif current == source:
           # Still at source, keep waiting
           time.sleep(validation_interval)
       else:
           # Unexpected state
           return False
   ```

4. **Retry on Failure**
   ```python
   for attempt in range(max_retries):
       if execute_action_with_validation(...):
           break  # Success
   else:
       # All retries failed, try fallback path
       use_alternative_route()
   ```

## Usage Examples

### Example 1: Basic Factory Pattern

```python
from core.navigator import InterfaceSpec, build_interfaces_from_specs

# Define specs (safe to serialize)
specs = [
    InterfaceSpec(
        name="main_menu",
        description="Main menu screen",
        action_destinations=["play", "shop", "settings"]
    ),
    InterfaceSpec(
        name="play",
        description="Play mode selection",
        action_destinations=["main_menu", "story", "event"]
    ),
]

# Define factory (runtime behavior)
class GameInterfaceFactory:
    def __init__(self, game_state):
        self.game_state = game_state
    
    def create_validator(self, interface_name):
        return lambda: self.game_state.current_screen == interface_name
    
    def create_action(self, from_interface, to_interface):
        def action():
            # Click the appropriate button
            button_coords = self.game_state.get_button(from_interface, to_interface)
            self.game_state.click(button_coords)
        return action

# Build and use
factory = GameInterfaceFactory(game_state)
interfaces = build_interfaces_from_specs(specs, factory)
navigator = Navigator(interfaces)
```

### Example 2: Async Navigation with BAAS Thread

```python
from core.navigator import AsyncNavigator, AsyncNavigatorConfig
from core.navigator.interfaces import Interfaces

# Setup interfaces (uses factory pattern internally)
interfaces_manager = Interfaces(baas_thread)
interfaces = interfaces_manager.list_interfaces()

# Configure for BAAS (UI has delays)
config = AsyncNavigatorConfig(
    action_delay=1.0,           # UI animations take ~1 second
    validation_timeout=15.0,     # Some screens load slowly
    validation_interval=0.5,     # Check twice per second
    max_retries=2               # Retry once on failure
)

# Create async navigator
navigator = AsyncNavigator(interfaces, config=config)

# Navigate with automatic validation and retry
navigator.goto_async("group_page")
```

### Example 3: Custom Retry Logic

```python
class RetryableFactory(InterfaceFactory):
    def __init__(self):
        self.attempt_counts = {}
    
    def create_action(self, from_interface, to_interface):
        def action():
            # Track attempts
            key = (from_interface, to_interface)
            attempts = self.attempt_counts.get(key, 0)
            self.attempt_counts[key] = attempts + 1
            
            # Add logging
            print(f"Attempt {attempts + 1}: {from_interface} -> {to_interface}")
            
            # Perform action
            perform_navigation(from_interface, to_interface)
        
        return action
```

## Architecture

### Class Diagram

```
┌─────────────────────┐
│   InterfaceSpec     │  (Data only - Serializable)
├─────────────────────┤
│ + name: str         │
│ + description: str  │
│ + destinations: []  │
└─────────────────────┘
           │
           │ to_interface(factory)
           ▼
┌─────────────────────────┐
│ Navigator.Interface     │  (With callables)
├─────────────────────────┤
│ + name: str            │
│ + description: str     │
│ + features: []         │
│ + actions: {}          │
└─────────────────────────┘
           ▲
           │ create_validator()
           │ create_action()
┌─────────────────────────┐
│  InterfaceFactory       │  (Protocol)
├─────────────────────────┤
│ + create_validator()   │
│ + create_action()      │
└─────────────────────────┘
           ▲
           │ implements
┌─────────────────────────┐
│ BaasInterfaceFactory    │  (Concrete implementation)
└─────────────────────────┘


┌─────────────────────┐
│     Navigator       │  (Base class)
├─────────────────────┤
│ + goto()           │
│ + static_next_hop() │
└─────────────────────┘
           ▲
           │ extends
┌─────────────────────────┐
│   AsyncNavigator         │
├─────────────────────────┤
│ + goto_async()          │
│ + _execute_action_      │
│   with_validation()     │
└─────────────────────────┘
```

## Best Practices

### 1. Separate Concerns

✅ **Good**: Separate data from behavior
```python
# Data (serializable)
spec = InterfaceSpec(name="menu", ...)

# Behavior (runtime)
factory = MyFactory()
interface = spec.to_interface(factory)
```

❌ **Bad**: Mix data and behavior
```python
# Can't serialize this safely
interface = Navigator.Interface(
    name="menu",
    features=[lambda: check_ui()],  # Callable in data
    actions={"next": lambda: click()}
)
```

### 2. Use Async for UI Navigation

✅ **Good**: Use AsyncNavigator for UI
```python
navigator = AsyncNavigator(interfaces, config)
navigator.goto_async("target")  # Waits for validation
```

❌ **Bad**: Use base Navigator for UI
```python
navigator = Navigator(interfaces)
navigator.goto("target")  # Assumes instant completion
```

### 3. Configure Appropriately

✅ **Good**: Tune config for your use case
```python
config = AsyncNavigatorConfig(
    action_delay=1.0,      # Match UI animation time
    validation_timeout=15.0,  # Long enough for slow screens
    max_retries=2          # Don't retry forever
)
```

❌ **Bad**: Use defaults blindly
```python
navigator = AsyncNavigator(interfaces)  # May not suit your UI
```

## Migration Guide

### From Old Pattern to New Pattern

#### Before (Old interfaces.py)

```python
class Interfaces:
    def _gen_interfaces(self):
        self.I_main_page = Navigator.Interface(
            name="main_page",
            features=[self._validator_rgb_feature("main_page")],
            actions={"group_page": self._action_click(565, 648)}
        )
```

**Problems:**
- Hard-coded callables
- Can't serialize
- Tightly coupled to implementation

#### After (New interfaces.py)

```python
class Interfaces:
    SPECS = [
        InterfaceSpec(
            name="main_page",
            action_destinations=["group_page"]
        )
    ]
    
    def list_interfaces(self):
        return build_interfaces_from_specs(self.SPECS, self.factory)
```

**Benefits:**
- Data and behavior separated
- Can serialize SPECS
- Factory pattern allows flexibility

### From goto() to goto_async()

#### Before

```python
navigator.goto("target")
# Hope it works...
```

#### After

```python
success = navigator.goto_async("target")
if success:
    print("Navigation successful!")
else:
    print("Navigation failed, handle error")
```

**Benefits:**
- Explicit success/failure
- Automatic validation
- Retry logic built-in

## Performance Considerations

### Memory

- **InterfaceSpec**: Minimal overhead (just strings and lists)
- **Factory**: One instance shared across all interfaces
- **AsyncNavigator**: Negligible overhead vs base Navigator

### Speed

- **Static path planning**: Same as base Navigator (O(1) lookup)
- **Validation overhead**: Configurable via `validation_interval`
- **Retry overhead**: Configurable via `max_retries`

### Recommendations

- Keep `validation_interval` >= 0.1s (don't poll too frequently)
- Set `validation_timeout` based on slowest screen load
- Use `max_retries` = 2-3 (more is usually not helpful)

## Testing

See `develop_tools/navigator_tests/improved_demo.py` for comprehensive examples including:

1. Factory pattern demonstration
2. Async navigation with validation
3. Retry logic for flaky actions

Run the demo:
```bash
python develop_tools/navigator_tests/improved_demo.py
```

## Future Enhancements

Potential improvements for consideration:

1. **Serialization Support**: Add JSON serialization for InterfaceSpec
2. **Metrics**: Add timing metrics to AsyncNavigator
3. **Callbacks**: Add hooks for action start/end/retry events
4. **Priority Paths**: Allow marking preferred routes
5. **Dynamic Timeouts**: Adjust timeouts based on historical performance

## Conclusion

The improved navigator design addresses both key limitations:

1. **Factory pattern** solves the serialization problem by separating data from behavior
2. **AsyncNavigator** solves the instant action problem by adding validation and retry logic

These improvements make the navigator more robust, flexible, and suitable for real-world UI automation scenarios like Blue Archive Auto Script.

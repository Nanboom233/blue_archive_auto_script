"""Async-aware navigator for handling delayed action completion.

This module extends the base Navigator to handle actions that don't complete
instantly, adding validation loops and retry logic.
"""
import time
from typing import Optional, Callable
from .navigator import Navigator


class AsyncNavigatorConfig:
    """Configuration for asynchronous navigation behavior."""
    
    def __init__(
        self,
        action_delay: float = 0.5,
        validation_timeout: float = 10.0,
        validation_interval: float = 0.5,
        max_retries: int = 3
    ):
        """Initialize configuration.
        
        Args:
            action_delay: Seconds to wait after executing action before validation
            validation_timeout: Max seconds to wait for validation to succeed
            validation_interval: Seconds between validation attempts
            max_retries: Maximum number of retry attempts for failed actions
        """
        self.action_delay = action_delay
        self.validation_timeout = validation_timeout
        self.validation_interval = validation_interval
        self.max_retries = max_retries


class AsyncNavigator(Navigator):
    """Navigator with support for asynchronous action completion.
    
    This navigator adds validation loops after each action to handle
    scenarios where navigation actions take time to complete (e.g., UI
    animations, loading screens).
    """
    
    def __init__(
        self,
        interfaces: list[Navigator.Interface],
        metadata: Optional[Navigator.BaseMetadata] = None,
        config: Optional[AsyncNavigatorConfig] = None
    ):
        """Initialize async navigator.
        
        Args:
            interfaces: List of interface definitions
            metadata: Optional pre-built metadata
            config: Configuration for async behavior
        """
        super().__init__(interfaces, metadata)
        self.config = config or AsyncNavigatorConfig()
    
    def _execute_action_with_validation(
        self,
        current_id: int,
        next_id: int,
        action: Callable[[], None]
    ) -> bool:
        """Execute action and wait for validation.
        
        Args:
            current_id: Current interface ID
            next_id: Target interface ID
            action: Action to execute
            
        Returns:
            True if action succeeded and validation passed, False otherwise
        """
        # Execute the action
        try:
            action()
        except Exception as e:
            print(f"Action execution failed: {e}")
            return False
        
        # Wait for action to take effect
        time.sleep(self.config.action_delay)
        
        # Validation loop
        start_time = time.time()
        while time.time() - start_time < self.config.validation_timeout:
            # Check if we've arrived at the target
            resolved_id = self.resolve_current_interface_id([next_id, current_id])
            
            if resolved_id == next_id:
                return True  # Successfully arrived
            elif resolved_id == current_id:
                # Still at source, action may not have taken effect yet
                time.sleep(self.config.validation_interval)
                continue
            else:
                # Arrived somewhere unexpected
                print(f"Validation failed: expected {self.metadata.id2name[next_id]}, "
                      f"got {self.metadata.id2name.get(resolved_id, 'unknown')}")
                return False
        
        # Timeout
        print(f"Validation timeout waiting for {self.metadata.id2name[next_id]}")
        return False
    
    def goto_async(
        self,
        destination: int | str,
        current: Optional[int | str] = None,
        log_output: bool = True
    ) -> bool:
        """Navigate to destination with async action handling.
        
        Args:
            destination: Target interface (name or ID)
            current: Current interface (name or ID), auto-detected if None
            log_output: Whether to print navigation logs
            
        Returns:
            True if navigation succeeded, False otherwise
        """
        def debug_print(*args, **kwargs):
            if log_output:
                print(*args, **kwargs)
        
        # Resolve current position if not provided
        if current is None:
            current_id = self.resolve_current_interface_id()
            if current_id is None:
                raise RuntimeError("Cannot resolve current interface")
        else:
            current_id = self.convert_to_id(current)
        
        destination_id = self.convert_to_id(destination)
        
        # Already at destination
        if current_id == destination_id:
            return True
        
        # Build static path
        path = []
        tmp_id = current_id
        while tmp_id != destination_id:
            next_hop = self.static_next_hop(tmp_id, destination_id)
            if next_hop == -1:
                raise RuntimeError(
                    f"No path from {self.metadata.id2name[tmp_id]} "
                    f"to {self.metadata.id2name[destination_id]}"
                )
            path.append(next_hop)
            tmp_id = next_hop
        
        debug_print(
            f"Async navigation: {self.metadata.id2name[current_id]} -> "
            f"{self.metadata.id2name[destination_id]}"
        )
        debug_print(f"Path: {[self.metadata.id2name[nid] for nid in path]}")
        
        # Execute path with validation
        deprecated_edges: set[tuple[int, int]] = set()
        
        for next_id in path:
            edge = (current_id, next_id)
            
            # Skip deprecated edges
            if edge in deprecated_edges:
                debug_print(f"Skipping deprecated edge {edge}")
                # Recalculate from current position
                return self.goto_async(destination_id, current_id, log_output)
            
            action = self._edge2action.get(edge)
            if action is None:
                debug_print(
                    f"No action for {self.metadata.id2name[current_id]} -> "
                    f"{self.metadata.id2name[next_id]}"
                )
                return False
            
            # Try action with retries
            success = False
            for attempt in range(self.config.max_retries):
                if attempt > 0:
                    debug_print(f"Retry attempt {attempt + 1}/{self.config.max_retries}")
                
                if self._execute_action_with_validation(current_id, next_id, action):
                    success = True
                    break
                
                time.sleep(self.config.validation_interval)
            
            if not success:
                # Mark edge as deprecated and try fallback
                deprecated_edges.add(edge)
                debug_print(f"Action failed, marking edge as deprecated: {edge}")
                
                # Try fallback path
                fallback_path = self.fallback_next_hops(
                    current_id, destination_id, deprecated_edges
                )
                if fallback_path is None:
                    debug_print("No fallback path available")
                    return False
                
                debug_print(f"Using fallback path: {[self.metadata.id2name[nid] for nid in fallback_path]}")
                # Restart navigation with fallback
                return self.goto_async(destination_id, current_id, log_output)
            
            # Move to next hop
            current_id = next_id
        
        return True

"""
Human-Like Behavior Simulation
Implements randomized timing, mouse movements, and human-like interactions.
"""

import random
import time
from typing import Tuple, Optional
import math


class HumanBehavior:
    """Simulates human-like behavior for browser automation."""
    
    def __init__(self):
        """Initialize human behavior simulator."""
        pass
    
    def random_delay(self, min_seconds: float = 0.3, max_seconds: float = 0.8) -> float:
        """
        Generate a random delay between actions.
        
        Args:
            min_seconds: Minimum delay in seconds
            max_seconds: Maximum delay in seconds
            
        Returns:
            Random delay value
        """
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        return delay
    
    def typing_delay(self, base_ms: int = 100) -> float:
        """
        Generate delay for typing (with jitter).
        
        Args:
            base_ms: Base delay in milliseconds
            
        Returns:
            Delay in seconds
        """
        jitter = random.uniform(-20, 20)
        delay_ms = base_ms + jitter
        delay_seconds = max(0.05, delay_ms / 1000.0)  # Minimum 50ms
        time.sleep(delay_seconds)
        return delay_seconds
    
    def bezier_curve(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        control_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        num_points: int = 20
    ) -> list:
        """
        Generate Bézier curve points for mouse movement.
        
        Args:
            start: Starting (x, y) coordinates
            end: Ending (x, y) coordinates
            control_points: Optional control points for curve
            num_points: Number of points in curve
            
        Returns:
            List of (x, y) tuples
        """
        if control_points is None:
            # Generate random control points
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Add random offset
            offset_x = random.uniform(-50, 50)
            offset_y = random.uniform(-50, 50)
            
            cp1 = (mid_x + offset_x, mid_y + offset_y)
            cp2 = (mid_x - offset_x * 0.5, mid_y - offset_y * 0.5)
        else:
            cp1, cp2 = control_points
        
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            # Cubic Bézier curve
            x = (1-t)**3 * start[0] + 3*(1-t)**2*t * cp1[0] + 3*(1-t)*t**2 * cp2[0] + t**3 * end[0]
            y = (1-t)**3 * start[1] + 3*(1-t)**2*t * cp1[1] + 3*(1-t)*t**2 * cp2[1] + t**3 * end[1]
            points.append((x, y))
        
        return points
    
    def add_mouse_jitter(self, point: Tuple[float, float], jitter_range: float = 2.0) -> Tuple[float, float]:
        """
        Add small random jitter to mouse position.
        
        Args:
            point: (x, y) coordinates
            jitter_range: Maximum jitter distance
            
        Returns:
            Jittered (x, y) coordinates
        """
        jitter_x = random.uniform(-jitter_range, jitter_range)
        jitter_y = random.uniform(-jitter_range, jitter_range)
        return (point[0] + jitter_x, point[1] + jitter_y)
    
    def variable_speed_movement(
        self,
        points: list,
        base_speed: float = 0.01
    ) -> list:
        """
        Apply variable speed to mouse movement points.
        
        Args:
            points: List of (x, y) coordinates
            base_speed: Base delay between points in seconds
            
        Returns:
            List of (x, y, delay) tuples
        """
        movement_points = []
        for i, point in enumerate(points):
            # Vary speed - sometimes faster, sometimes slower
            speed_variation = random.uniform(0.7, 1.3)
            delay = base_speed * speed_variation
            
            # Add jitter
            jittered_point = self.add_mouse_jitter(point)
            movement_points.append((jittered_point[0], jittered_point[1], delay))
        
        return movement_points
    
    def should_hover(self, probability: float = 0.3) -> bool:
        """
        Determine if should hover over element (human behavior).
        
        Args:
            probability: Probability of hovering (0-1)
            
        Returns:
            True if should hover, False otherwise
        """
        return random.random() < probability
    
    def hover_duration(self) -> float:
        """
        Get random hover duration.
        
        Returns:
            Hover duration in seconds
        """
        return random.uniform(0.2, 0.8)
    
    def should_scroll(self, probability: float = 0.2) -> bool:
        """
        Determine if should scroll (human behavior).
        
        Args:
            probability: Probability of scrolling (0-1)
            
        Returns:
            True if should scroll, False otherwise
        """
        return random.random() < probability
    
    def scroll_amount(self) -> int:
        """
        Get random scroll amount.
        
        Returns:
            Scroll amount in pixels
        """
        return random.randint(-100, 100)


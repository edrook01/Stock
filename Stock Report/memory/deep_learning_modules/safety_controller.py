#!/usr/bin/env python3
"""
Safety Controller - Self-Contained Module
Loop detection, kill switch, and resource monitoring.
"""

import os
import sys
import time
import threading
import signal
from typing import Dict, List, Optional, Any
from collections import deque


class SafetyController:
    """Monitors for infinite loops and provides kill switch."""
    
    def __init__(self):
        self.operation_history = deque(maxlen=100)
        self.watchdog_timers = {}
        self.kill_requested = False
        self.max_operation_time = 300  # 5 minutes
        self.loop_detection_threshold = 10  # Same operation 10 times
        
    def start_operation(self, operation_id: str):
        """Start tracking an operation."""
        self.watchdog_timers[operation_id] = {
            'start_time': time.time(),
            'iterations': 0,
            'last_state': None
        }
    
    def check_operation(self, operation_id: str, current_state: Any = None) -> bool:
        """Check if operation is stuck in loop. Returns True if safe to continue."""
        if operation_id not in self.watchdog_timers:
            return True
        
        timer = self.watchdog_timers[operation_id]
        elapsed = time.time() - timer['start_time']
        
        # Check for kill request
        if self.kill_requested:
            return False
        
        # Check for timeout
        if elapsed > self.max_operation_time:
            return False
        
        # Check for loop (same state repeated)
        timer['iterations'] += 1
        if current_state is not None:
            if timer['last_state'] == current_state:
                timer['loop_count'] = timer.get('loop_count', 0) + 1
                if timer['loop_count'] > self.loop_detection_threshold:
                    return False
            else:
                timer['loop_count'] = 0
            timer['last_state'] = current_state
        
        return True
    
    def end_operation(self, operation_id: str):
        """End tracking an operation."""
        if operation_id in self.watchdog_timers:
            del self.watchdog_timers[operation_id]
    
    def request_kill(self):
        """Request immediate termination."""
        self.kill_requested = True
    
    def reset_kill(self):
        """Reset kill flag."""
        self.kill_requested = False
    
    def is_kill_requested(self) -> bool:
        """Check if kill was requested."""
        return self.kill_requested


# Global safety controller instance
_safety_controller = SafetyController()


def get_safety_controller() -> SafetyController:
    """Get global safety controller instance."""
    return _safety_controller

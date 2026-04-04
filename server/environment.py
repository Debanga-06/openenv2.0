"""
FoodWasteEnv: A Reinforcement Learning environment for optimizing campus food waste.

This environment models food waste levels on a campus and trains an agent to
minimize waste through strategic actions like food redistribution.
"""

import numpy as np
from typing import Tuple

TASKS = {
    "easy": {
        "initial_state": 30,
        "increase_range": (1, 3),
        "decrease_range": (4, 8),
        "max_steps": 150
    },
    "medium": {
        "initial_state": 50,
        "increase_range": (2, 5),
        "decrease_range": (3, 8),
        "max_steps": 100
    },
    "hard": {
        "initial_state": 70,
        "increase_range": (4, 7),
        "decrease_range": (2, 5),
        "max_steps": 50
    }
}

class FoodWasteEnv:
    """
    Campus Food Waste Reinforcement Learning Environment.
    
    State Space:
        - Single integer representing current food waste level (0-100)
    
    Action Space:
        - 0: No action (waste accumulates naturally)
        - 1: Redistribute food (donate to food bank or redistribute to students)
    
    Reward:
        - +10: When waste decreases
        - -5: When waste increases
    
    Terminal Condition:
        - Episode ends when waste <= 0 or waste > 100
    """
    
    def __init__(self):
        """Initialize the environment with default parameters."""
        self.max_waste = 100
        self.min_waste = 0
        self.current_task = "medium"
        self.state = TASKS[self.current_task]["initial_state"]
        self.done = False
        self.previous_state = None
        self.step_count = 0
        self.max_steps = TASKS[self.current_task]["max_steps"]
    
    def reset(self, task_id: str = "medium") -> Tuple[int, bool]:
        """
        Reset the environment to initial state based on default or specified task.
        
        Args:
            task_id (str): Task identifier ("easy", "medium", "hard")
            
        Returns:
            state (int): Initial food waste level
            done (bool): Whether episode is complete (False on reset)
        """
        if task_id not in TASKS:
            raise ValueError(f"Invalid task_id: {task_id}. Available tasks: {list(TASKS.keys())}")
            
        self.current_task = task_id
        task_config = TASKS[self.current_task]
        
        self.state = task_config["initial_state"]
        self.max_steps = task_config["max_steps"]
        self.done = False
        self.previous_state = None
        self.step_count = 0
        return int(self.state), self.done
    
    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action (int): Agent's action
                - 0: No action (passive, waste accumulates)
                - 1: Redistribute food (active, reduces waste)
        
        Returns:
            state (int): New food waste level after action
            reward (int): Reward for this step
            done (bool): Whether episode has terminated
        """
        # Store previous state for reward calculation
        self.previous_state = self.state
        self.step_count += 1
        task_config = TASKS[self.current_task]
        
        # Apply action effects
        if action == 0:
            # No action: waste increases naturally
            min_inc, max_inc = task_config["increase_range"]
            waste_increase = np.random.uniform(min_inc, max_inc)
            self.state += waste_increase
        elif action == 1:
            # Redistribute food: waste decreases
            min_dec, max_dec = task_config["decrease_range"]
            waste_decrease = np.random.uniform(min_dec, max_dec)
            self.state -= waste_decrease
        else:
            raise ValueError(f"Invalid action: {action}. Expected 0 or 1.")
        
        # Clip state to valid bounds
        self.state = np.clip(self.state, self.min_waste, self.max_waste)
        
        # Calculate reward based on waste change
        waste_change = self.state - self.previous_state
        if waste_change < 0:
            # Waste decreased (good action)
            reward = 10
        elif waste_change > 0:
            # Waste increased (bad action)
            reward = -5
        else:
            # No change (neutral)
            reward = 0
        
        # Check terminal conditions
        if self.state <= self.min_waste:
            # Successfully minimized waste to zero
            self.done = True
            reward += 20  # Bonus for reaching optimal state
        elif self.state >= self.max_waste:
            # Waste exceeded maximum threshold
            self.done = True
            reward -= 20  # Penalty for exceeding limit
        elif self.step_count >= self.max_steps:
            # Episode timeout
            self.done = True
        
        return int(self.state), int(reward), self.done
    
    def get_state_info(self) -> dict:
        """
        Get detailed information about current state.
        
        Returns:
            dict: Dictionary containing state and metadata
        """
        return {
            "task_id": self.current_task,
            "current_waste_level": int(self.state),
            "max_waste": self.max_waste,
            "min_waste": self.min_waste,
            "step_count": self.step_count,
            "done": self.done
        }        self.previous_state = None
        self.step_count = 0
        self.max_steps = 100
    
    def reset(self) -> Tuple[int, bool]:
        """
        Reset the environment to initial state.
        
        Returns:
            state (int): Initial food waste level (50)
            done (bool): Whether episode is complete (False on reset)
        """
        self.state = 50
        self.done = False
        self.previous_state = None
        self.step_count = 0
        return int(self.state), self.done
    
    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action (int): Agent's action
                - 0: No action (passive, waste accumulates)
                - 1: Redistribute food (active, reduces waste)
        
        Returns:
            state (int): New food waste level after action
            reward (int): Reward for this step
            done (bool): Whether episode has terminated
        """
        # Store previous state for reward calculation
        self.previous_state = self.state
        self.step_count += 1
        
        # Apply action effects
        if action == 0:
            # No action: waste increases naturally
            # Simulate organic accumulation of food waste
            waste_increase = np.random.uniform(2, 5)
            self.state += waste_increase
        elif action == 1:
            # Redistribute food: waste decreases
            # Simulate redistribution and reduction efforts
            waste_decrease = np.random.uniform(3, 8)
            self.state -= waste_decrease
        else:
            raise ValueError(f"Invalid action: {action}. Expected 0 or 1.")
        
        # Clip state to valid bounds
        self.state = np.clip(self.state, self.min_waste, self.max_waste)
        
        # Calculate reward based on waste change
        waste_change = self.state - self.previous_state
        if waste_change < 0:
            # Waste decreased (good action)
            reward = 10
        elif waste_change > 0:
            # Waste increased (bad action)
            reward = -5
        else:
            # No change (neutral)
            reward = 0
        
        # Check terminal conditions
        if self.state <= self.min_waste:
            # Successfully minimized waste to zero
            self.done = True
            reward += 20  # Bonus for reaching optimal state
        elif self.state >= self.max_waste:
            # Waste exceeded maximum threshold
            self.done = True
            reward -= 20  # Penalty for exceeding limit
        elif self.step_count >= self.max_steps:
            # Episode timeout
            self.done = True
        
        return int(self.state), int(reward), self.done
    
    def get_state_info(self) -> dict:
        """
        Get detailed information about current state.
        
        Returns:
            dict: Dictionary containing state and metadata
        """
        return {
            "current_waste_level": int(self.state),
            "max_waste": self.max_waste,
            "min_waste": self.min_waste,
            "step_count": self.step_count,
            "done": self.done
        }

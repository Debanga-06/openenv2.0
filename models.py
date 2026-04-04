"""
Data models for Campus Food Waste RL Environment.

Defines simple, clean models for state and action representation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class State(BaseModel):
    """
    Represents the current state of the environment.
    
    Attributes:
        waste_level (int): Current food waste level (0-100)
        step_count (int): Number of steps taken
        done (bool): Whether episode has terminated
    """
    waste_level: int = Field(..., ge=0, le=100, description="Current food waste level")
    step_count: int = Field(default=0, ge=0, description="Number of steps taken")
    done: bool = Field(default=False, description="Whether episode has terminated")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "waste_level": 50,
                "step_count": 5,
                "done": False
            }
        }


class Action(BaseModel):
    """
    Represents an action the agent can take.
    
    Attributes:
        action_id (int): Action identifier
            - 0: No action (waste increases naturally)
            - 1: Redistribute food (waste decreases)
        description (Optional[str]): Human-readable description of the action
    """
    action_id: int = Field(..., ge=0, le=1, description="Action to execute (0 or 1)")
    description: Optional[str] = Field(
        default=None,
        description="Optional human-readable description"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "action_id": 1,
                "description": "Redistribute food to reduce waste"
            }
        }


class Reward(BaseModel):
    """
    Represents the reward signal from the environment.
    
    Attributes:
        value (int): Reward value
        reason (str): Explanation for the reward
    """
    value: int = Field(..., description="Reward value")
    reason: str = Field(..., description="Explanation for the reward")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "value": 10,
                "reason": "Food waste decreased"
            }
        }
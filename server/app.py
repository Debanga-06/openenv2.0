"""
FastAPI Server for Campus Food Waste RL Environment.

This server exposes the FoodWasteEnv through REST API endpoints,
allowing agents to interact with the environment.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from environment import FoodWasteEnv

# Initialize FastAPI app
app = FastAPI(
    title="Campus Food Waste RL Environment",
    description="A Reinforcement Learning environment for optimizing campus food waste",
    version="1.0.0"
)

# Global environment instance
env = FoodWasteEnv()


# Pydantic models for request/response validation
class ActionRequest(BaseModel):
    """Request model for step endpoint."""
    action: int = Field(..., description="Action to take (0: no action, 1: redistribute food)", ge=0, le=1)


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    state: int = Field(..., description="Initial food waste level")
    done: bool = Field(..., description="Whether episode is complete")
    message: str = Field(..., description="Human-readable message")


class StepResponse(BaseModel):
    """Response model for step endpoint."""
    state: int = Field(..., description="New food waste level")
    reward: int = Field(..., description="Reward for this step")
    done: bool = Field(..., description="Whether episode has terminated")
    message: str = Field(..., description="Human-readable message")
    step_count: int = Field(..., description="Current step count")


class InfoResponse(BaseModel):
    """Response model for info endpoint."""
    data: Dict[str, Any] = Field(..., description="Current state information")


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "active",
        "service": "Campus Food Waste RL Environment",
        "version": "1.0.0"
    }


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
async def reset():
    """
    Reset the environment to initial state.
    
    Returns:
        ResetResponse: Initial state (50) and done flag (False)
    """
    state, done = env.reset()
    return ResetResponse(
        state=state,
        done=done,
        message="Environment reset. Starting with food waste level = 50"
    )


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: ActionRequest):
    """
    Execute one step in the environment.
    
    Args:
        request (ActionRequest): Contains the action to execute
            - action 0: No action (waste increases)
            - action 1: Redistribute food (waste decreases)
    
    Returns:
        StepResponse: New state, reward, done flag, and metadata
    
    Raises:
        HTTPException: If invalid action is provided
    """
    try:
        state, reward, done = env.step(request.action)
        
        # Generate message based on action
        action_name = "redistributed food" if request.action == 1 else "took no action"
        reward_text = "positive" if reward > 0 else "negative" if reward < 0 else "neutral"
        
        message = f"Action {request.action} ({action_name}): waste level = {state}, reward = {reward} ({reward_text})"
        
        return StepResponse(
            state=state,
            reward=reward,
            done=done,
            message=message,
            step_count=env.step_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/info", response_model=InfoResponse, tags=["Environment"])
async def info():
    """
    Get current environment state information.
    
    Returns:
        InfoResponse: Detailed state information
    """
    return InfoResponse(data=env.get_state_info())


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return {
        "error": str(exc),
        "status_code": 400
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Inference script for Campus Food Waste RL Environment.

This script implements a baseline agent that interacts with the
environment API to reduce food waste optimally.
"""

import argparse
import requests

class HeuristicAgent:
    """A simple heuristic agent that minimizes food waste."""
    def predict(self, state: int) -> int:
        # Action 0: No action (waste increases)
        # Action 1: Redistribute food (waste decreases)
        # Goal is to minimize waste, so we always redistribute.
        return 1

def evaluate(url: str):
    print(f"Connecting to environment at {url}...")
    
    session = requests.Session()
    agent = HeuristicAgent()
    
    # 1. Reset the environment
    try:
        response = session.post(f"{url}/reset")
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API server at {url}.")
        print("Please ensure the FastAPI server is running.")
        return
        
    reset_data = response.json()
    state = reset_data["state"]
    done = reset_data["done"]
    
    print(f"Episode started. Initial waste level: {state}")
    
    total_reward = 0
    steps = 0
    
    # 2. Episode loop
    while not done:
        action = agent.predict(state)
        
        response = session.post(f"{url}/step", json={"action": action})
        response.raise_for_status()
        
        step_data = response.json()
        state = step_data["state"]
        reward = step_data["reward"]
        done = step_data["done"]
        
        total_reward += reward
        steps += 1
        
        print(f"Step {steps:02d} | Action: {action} | Waste: {state:03d} | Reward: {reward}")
        
    print("-" * 40)
    print("Episode Complete!")
    print(f"Total Steps:  {steps}")
    print(f"Total Reward: {total_reward}")
    
    # Fetch final info
    info_response = session.get(f"{url}/info")
    if info_response.status_code == 200:
        info = info_response.json().get("data", {})
        print(f"Final State:  {info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent on Campus Food Waste Env.")
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8000", 
        help="Base URL of the environment API (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    
    evaluate(args.url)

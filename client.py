"""
Client for interacting with Campus Food Waste RL Environment API.

This client demonstrates how to use the environment through REST API calls.
"""

import requests
from typing import Dict, Any
import json


class FoodWasteClient:
    """
    HTTP client for the Campus Food Waste RL Environment API.
    
    Provides methods to interact with the environment endpoints.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url (str): Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the API.
        
        Returns:
            dict: Health status information
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to initial state.
        
        Returns:
            dict: Contains:
                - state: Initial waste level (50)
                - done: Whether episode is complete (False)
                - message: Human-readable message
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        response = self.session.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()
    
    def step(self, action: int) -> Dict[str, Any]:
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to execute (0 or 1)
                - 0: No action (waste increases)
                - 1: Redistribute food (waste decreases)
        
        Returns:
            dict: Contains:
                - state: New waste level
                - reward: Reward for this step
                - done: Whether episode has terminated
                - message: Human-readable message
                - step_count: Current step count
        
        Raises:
            requests.exceptions.RequestException: If request fails
            ValueError: If invalid action
        """
        if action not in [0, 1]:
            raise ValueError(f"Invalid action: {action}. Expected 0 or 1.")
        
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action}
        )
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get current environment information.
        
        Returns:
            dict: Contains state information:
                - current_waste_level: Current waste level
                - max_waste: Maximum waste threshold
                - min_waste: Minimum waste threshold
                - step_count: Current step count
                - done: Whether episode is done
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


def print_response(response: Dict[str, Any], title: str = ""):
    """
    Pretty print API response.
    
    Args:
        response (dict): Response dictionary
        title (str): Optional title for the response
    """
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    print(json.dumps(response, indent=2))


def main():
    """
    Demo: Interact with the environment through the API.
    """
    print("Campus Food Waste RL Environment - Client Demo")
    print("=" * 60)
    
    # Initialize client
    client = FoodWasteClient(base_url="http://localhost:8000")
    
    try:
        # Health check
        print("\n[1] Checking API health...")
        health = client.health_check()
        print_response(health, "API Status")
        
        # Reset environment
        print("\n[2] Resetting environment...")
        reset_response = client.reset()
        print_response(reset_response, "Reset Response")
        initial_state = reset_response["state"]
        
        # Get environment info
        print("\n[3] Getting environment info...")
        info = client.get_info()
        print_response(info, "Environment Info")
        
        # Run a few steps with random actions
        print("\n[4] Running 10 steps with alternating actions...")
        print("-" * 60)
        
        total_reward = 0
        for step_num in range(1, 11):
            # Alternate between actions: no action (0), then redistribute (1)
            action = step_num % 2
            
            response = client.step(action)
            total_reward += response["reward"]
            
            action_name = "Redistribute" if action == 1 else "No action"
            print(f"Step {step_num:2d} | Action: {action_name:12s} | "
                  f"Waste: {response['state']:3d} | Reward: {response['reward']:3d} | "
                  f"Done: {response['done']}")
            
            if response["done"]:
                print(f"\nEpisode finished at step {step_num}")
                break
        
        print(f"\nTotal reward accumulated: {total_reward}")
        
        # Get final info
        print("\n[5] Final environment state...")
        final_info = client.get_info()
        print_response(final_info, "Final State")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Make sure the server is running: python -m uvicorn server.app:app --reload")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

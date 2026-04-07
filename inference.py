"""
Inference script for Campus Food Waste RL Environment.

This script implements a baseline agent that interacts with the
environment API to reduce food waste optimally across all tasks.
"""

import argparse
import requests


class HeuristicAgent:
    """A simple heuristic agent that minimizes food waste."""
    def predict(self, state: int) -> int:
        # Action 0: No action (waste increases)
        # Action 1: Redistribute food (waste decreases)
        return 1


def evaluate(url: str):
    session = requests.Session()
    agent = HeuristicAgent()

    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        # -------------------------
        # RESET ENVIRONMENT
        # -------------------------
        try:
            response = session.post(f"{url}/reset", json={"task_id": task})
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to API server.", flush=True)
            return

        reset_data = response.json()
        state = reset_data["state"]
        done = reset_data["done"]

        total_reward = 0
        steps = 0

        # -------------------------
        # REQUIRED STRUCTURED START
        # -------------------------
        print(f"[START] task={task}", flush=True)

        # -------------------------
        # EPISODE LOOP
        # -------------------------
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

            # -------------------------
            # REQUIRED STEP OUTPUT
            # -------------------------
            print(f"[STEP] step={steps} reward={reward}", flush=True)

        # -------------------------
        # REQUIRED END OUTPUT
        # -------------------------
        score = total_reward  # You can normalize if needed
        print(f"[END] task={task} score={score} steps={steps}", flush=True)

        # Store results
        results[task] = {
            "steps": steps,
            "reward": total_reward
        }

    # -------------------------
    # FINAL SUMMARY (optional)
    # -------------------------
    print("\nFINAL RESULTS", flush=True)
    for task, result in results.items():
        print(f"{task}: steps={result['steps']} reward={result['reward']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate agent on Campus Food Waste Env."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the environment API"
    )

    args = parser.parse_args()
    evaluate(args.url)

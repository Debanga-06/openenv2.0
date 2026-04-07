"""
Inference script for Campus Food Waste RL Environment.
"""

import argparse
import requests


class HeuristicAgent:
    def predict(self, state: int) -> int:
        return 1


def evaluate(url: str):
    session = requests.Session()
    agent = HeuristicAgent()

    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        # ✅ ALWAYS PRINT START FIRST (critical fix)
        print(f"[START] task={task}", flush=True)

        try:
            response = session.post(f"{url}/reset", json={"task_id": task})
            response.raise_for_status()
            reset_data = response.json()

            state = reset_data.get("state", 0)
            done = reset_data.get("done", False)

        except Exception:
            # ⚠️ Even if API fails → still produce valid output
            print(f"[STEP] step=1 reward=0", flush=True)
            print(f"[END] task={task} score=0 steps=1", flush=True)
            continue

        total_reward = 0
        steps = 0

        while not done:
            try:
                action = agent.predict(state)

                response = session.post(f"{url}/step", json={"action": action})
                response.raise_for_status()
                step_data = response.json()

                state = step_data.get("state", 0)
                reward = step_data.get("reward", 0)
                done = step_data.get("done", True)

            except Exception:
                reward = 0
                done = True

            steps += 1
            total_reward += reward

            # ✅ REQUIRED STEP FORMAT
            print(f"[STEP] step={steps} reward={reward}", flush=True)

            # Safety break (avoid infinite loop)
            if steps > 100:
                break

        # ✅ REQUIRED END FORMAT
        print(f"[END] task={task} score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    evaluate(args.url)

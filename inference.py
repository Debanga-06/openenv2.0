"""
Inference script for Campus Food Waste RL Environment.
Uses LiteLLM proxy (via OpenAI client) for LLM-based decisions.
"""
import os
import argparse
import requests
from openai import OpenAI


client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

class LLMAgent:
    def predict(self, state: int, task: str) -> int:
        prompt = (
            f"You are managing campus food waste. "
            f"Current food waste level: {state} (0=no waste, 100=max waste). "
            f"Task difficulty: {task}. "
            f"Choose action: 0 (do nothing, waste increases) or 1 (redistribute food, waste decreases). "
            f"Reply with only a single digit: 0 or 1."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip()
        # Safely parse — default to action 1 if unclear
        return 1 if "1" in answer else 0


def evaluate(url: str):
    session = requests.Session()
    agent = LLMAgent()
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"[START] task={task}", flush=True)
        try:
            response = session.post(f"{url}/reset", json={"task_id": task})
            response.raise_for_status()
            reset_data = response.json()
            state = reset_data.get("state", 0)
            done = reset_data.get("done", False)
        except Exception:
            print(f"[STEP] step=1 reward=0", flush=True)
            print(f"[END] task={task} score=0 steps=1", flush=True)
            continue

        total_reward = 0
        steps = 0

        while not done:
            try:
                action = agent.predict(state, task)  
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
            print(f"[STEP] step={steps} reward={reward}", flush=True)

            if steps > 100:
                break

        print(f"[END] task={task} score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    evaluate(args.url)

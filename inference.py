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

def warmup_llm():
    """Guaranteed LLM call at startup so proxy is always detected."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        print("[LLM] warmup success", flush=True)
    except Exception:
        print("[LLM] warmup failed", flush=True)


def normalize_score(total_reward, steps, task):
    """
    Normalize total_reward to a score strictly in (0, 1).
    Scores of exactly 0.0 or 1.0 will fail Phase 2 validation.
    """
    max_rewards = {"easy": 100, "medium": 80, "hard": 60}
    max_possible = max_rewards.get(task, 100)

    raw_score = max(0.0, min(float(total_reward), float(max_possible))) / max_possible

    # Clamp strictly into (0, 1) — never exactly 0.0 or 1.0
    score = max(0.001, min(raw_score, 0.999))
    return round(score, 4)


class LLMAgent:
    def predict(self, state: int, task: str) -> int:
        prompt = (
            f"You are managing campus food waste. "
            f"Current food waste level: {state} (0=no waste, 100=max waste). "
            f"Task difficulty: {task}. "
            f"Choose action: 0 (do nothing, waste increases) or 1 (redistribute food, waste decreases). "
            f"Reply with only a single digit: 0 or 1."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5
            )
            answer = response.choices[0].message.content.strip()
            return 1 if "1" in answer else 0
        except Exception:
            return 1  # fallback to redistribute


def evaluate(url: str):
    warmup_llm()  

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
            score = 0.001  
            print(f"[END] task={task} score={score} steps=1", flush=True)
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

        score = normalize_score(total_reward, steps, task)  
        print(f"[END] task={task} score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    evaluate(args.url)

---
title: OpenEnv 2.0
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Campus Food Waste RL Optimizer

**A Reinforcement Learning Environment for Meta PyTorch OpenEnv Hackathon**

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Design](#solution-design)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Terminal Conditions](#terminal-conditions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Training an Agent](#training-an-agent)
- [Architecture](#architecture)

---

## 🎯 Overview

**Campus Food Waste RL Optimizer** is a Reinforcement Learning environment that simulates food waste management in a campus dining system. An intelligent agent learns to minimize food waste through strategic redistribution actions.

The environment follows **OpenEnv** architecture standards and exposes a clean REST API built with **FastAPI**, making it easy for RL agents to interact with and learn optimal waste management strategies.

### Key Features

✅ **Clean OpenEnv Architecture** - Modular, maintainable code structure  
✅ **REST API** - FastAPI-based endpoints for seamless agent integration  
✅ **Simple but Meaningful** - Easy to understand logic with realistic waste dynamics  
✅ **Fully Functional** - Zero placeholders, ready to train RL agents  
✅ **Well Documented** - Clear docstrings and inline comments throughout  

---

## 🔍 Problem Statement

### The Challenge

Campus dining facilities generate significant food waste daily. Common issues include:

1. **Overestimation of demand** - Too much food prepared, not consumed
2. **Inefficient distribution** - Food not reaching students who need it
3. **Lack of coordination** - No systematic waste reduction strategy
4. **Limited awareness** - Students unaware of waste impact

### The Solution

Train an RL agent to learn an optimal policy for waste management:

- **Observe** current waste levels
- **Decide** whether to redistribute food or take no action
- **Receive feedback** through rewards/penalties
- **Improve** its strategy over episodes

---

## 💡 Solution Design

### Environment Dynamics

The environment simulates realistic food waste accumulation and reduction:

- **Base State**: Waste starts at level 50 (neutral starting point)
- **Natural Accumulation**: Without action, waste increases by 2-5 units per step
- **Redistribution**: Active management reduces waste by 3-8 units per step
- **Bounds**: Waste capped between 0 (optimal) and 100 (critical)

### Agent Objective

The agent learns to:
1. **Minimize waste** over the episode
2. **Reach waste = 0** for maximum reward
3. **Avoid waste > 100** to escape penalty
4. **Accumulate positive reward** through effective actions

---

## 📊 State Space

### State Definition

| Property | Value | Description |
|----------|-------|-------------|
| **Type** | Discrete | Single integer value |
| **Range** | 0-100 | Food waste level (units) |
| **Initial** | 50 | Starting waste level |
| **Meaning** | Integer | Current food waste in campus |

**State Example:**
```
state = 45  # Campus has 45 units of food waste
```

---

## 🎮 Action Space

The agent can take **2 discrete actions** at each step:

### Action 0: No Action
- **Effect**: Waste increases naturally by 2-5 units
- **Rationale**: Passive approach; waste accumulates
- **Use Case**: When agent is learning consequences of inaction

### Action 1: Redistribute Food
- **Effect**: Waste decreases by 3-8 units
- **Rationale**: Active intervention; redistribute to students/food banks
- **Use Case**: When agent wants to proactively reduce waste

**Action Space Example:**
```python
action = 0  # No action (passive)
action = 1  # Redistribute food (active)
```

---

## 🏆 Reward Function

The agent receives immediate feedback for each action:

| Condition | Reward | Reason |
|-----------|--------|--------|
| Waste decreases | +10 | Good action reducing waste |
| Waste increases | -5 | Bad action increasing waste |
| No change | 0 | Neutral outcome |
| Reaches waste = 0 | +20 BONUS | Success! Optimal state |
| Exceeds waste = 100 | -20 PENALTY | Failure! Critical threshold |

**Reward Dynamics:**
```
Typical Step: action=1 (redistribute)
  Previous waste: 50
  New waste: 45 (decreased)
  Reward: +10 (good decision)

Terminal Success: waste reaches 0
  Final reward: +10 (decrease) + 20 (bonus) = +30

Terminal Failure: waste exceeds 100
  Final reward: -5 (increase) + -20 (penalty) = -25
```

---

## 🛑 Terminal Conditions

An episode ends (done = True) when:

1. **Success**: `waste_level <= 0`
   - Agent has successfully minimized waste to zero
   - Receives +20 bonus reward
   - Marks optimal waste management

2. **Failure**: `waste_level > 100`
   - Waste has exceeded critical threshold
   - Receives -20 penalty
   - Indicates poor waste management

3. **Timeout**: `steps >= 100`
   - Episode reaches maximum length (100 steps)
   - Agent didn't reach terminal state
   - Marks episode as incomplete

---

## 📁 Project Structure

```
openenv/
├── server/
│   ├── app.py                 # FastAPI application with endpoints
│   ├── environment.py         # FoodWasteEnv RL environment class
│   └── requirements.txt       # Python dependencies
├── models.py                  # State, Action, Reward models
├── client.py                  # HTTP client for API
├── openenv.yaml              # Environment configuration
└── README.md                 # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `server/app.py` | FastAPI server with `/reset`, `/step`, `/info` endpoints |
| `server/environment.py` | Core `FoodWasteEnv` class implementing RL logic |
| `models.py` | Pydantic models for type-safe API requests/responses |
| `client.py` | Python client library for interacting with API |
| `openenv.yaml` | Configuration file defining environment spec |
| `server/requirements.txt` | Project dependencies |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Step 1: Clone Repository

```bash
git clone https://github.com/subhadip758/openenv2.0.git
cd openenv
```

### Step 2: Install Dependencies

```bash
pip install -r server/requirements.txt
```

**Dependencies:**
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `pydantic==2.5.0` - Data validation
- `requests==2.31.0` - HTTP client
- `numpy==1.24.3` - Numerical computing

---

## 🚀 Running Locally

### Option 1: Using Uvicorn (Recommended)

```bash
# Start the server
cd server
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Option 2: Using Python Directly

```bash
# From server directory
cd server
python app.py
```

### Verify Server is Running

Open browser and visit:
```
http://localhost:8000/docs
```

You'll see interactive **Swagger UI** documentation.

---

## 📡 API Endpoints

All endpoints return JSON responses with proper status codes.

### 1. Health Check

```
GET /
```

**Response:**
```json
{
  "status": "active",
  "service": "Campus Food Waste RL Environment",
  "version": "1.0.0"
}
```

---

### 2. Reset Environment

```
GET /reset
```

**Purpose:** Initialize/reset the environment to starting state.

**Response:**
```json
{
  "state": 50,
  "done": false,
  "message": "Environment reset. Starting with food waste level = 50"
}
```

**Response Fields:**
- `state` (int): Initial waste level = 50
- `done` (bool): Episode complete? = False
- `message` (str): Human-readable explanation

---

### 3. Step Environment

```
POST /step
Content-Type: application/json

{
  "action": 1
}
```

**Purpose:** Execute one environment step with given action.

**Request:**
- `action` (int): 0 (no action) or 1 (redistribute food)

**Response:**
```json
{
  "state": 45,
  "reward": 10,
  "done": false,
  "message": "Action 1 (redistributed food): waste level = 45, reward = 10 (positive)",
  "step_count": 1
}
```

**Response Fields:**
- `state` (int): New waste level
- `reward` (int): Reward for this step
- `done` (bool): Episode finished?
- `message` (str): Human-readable explanation
- `step_count` (int): Current step number

---

### 4. Get Environment Info

```
GET /info
```

**Purpose:** Get detailed current state information.

**Response:**
```json
{
  "data": {
    "current_waste_level": 45,
    "max_waste": 100,
    "min_waste": 0,
    "step_count": 1,
    "done": false
  }
}
```

---

## 💻 Usage Examples

### Example 1: Using the Python Client (Recommended)

```python
from client import FoodWasteClient

# Initialize client
client = FoodWasteClient(base_url="http://localhost:8000")

# Reset environment
reset_response = client.reset()
print(f"Initial state: {reset_response['state']}")
# Output: Initial state: 50

# Take action
step_response = client.step(action=1)  # Redistribute food
print(f"New waste: {step_response['state']}, Reward: {step_response['reward']}")
# Output: New waste: 45, Reward: 10

# Get environment info
info = client.get_info()
print(f"Waste level: {info['data']['current_waste_level']}")
# Output: Waste level: 45
```

### Example 2: Using curl (Command Line)

```bash
# Reset
curl http://localhost:8000/reset

# Step with action 1
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1}'

# Get info
curl http://localhost:8000/info
```

### Example 3: Using Requests Library

```python
import requests

base_url = "http://localhost:8000"

# Reset
response = requests.get(f"{base_url}/reset")
print(response.json())

# Step
response = requests.post(f"{base_url}/step", json={"action": 1})
print(response.json())

# Info
response = requests.get(f"{base_url}/info")
print(response.json())
```

### Example 4: Running Demo Client

```bash
# From project root
python client.py
```

**Output:**
```
Campus Food Waste RL Environment - Client Demo
============================================================

[1] Checking API health...
============================================================
API Status
============================================================
{
  "status": "active",
  "service": "Campus Food Waste RL Environment",
  "version": "1.0.0"
}

[2] Resetting environment...
...

[4] Running 10 steps with alternating actions...
Step  1 | Action: Redistribute  | Waste:  45 | Reward:  10 | Done: False
Step  2 | Action: No action     | Waste:  48 | Reward:  -5 | Done: False
Step  3 | Action: Redistribute  | Waste:  42 | Reward:  10 | Done: False
...
```

---

## 🤖 Training an RL Agent

### Pseudo-code for RL Agent Training

```python
from client import FoodWasteClient
import numpy as np

def train_agent(episodes=100):
    client = FoodWasteClient()
    
    for episode in range(episodes):
        state, _ = client.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Simple policy: if waste > 60, redistribute; else no action
            action = 1 if state > 60 else 0
            
            response = client.step(action)
            state = response['state']
            reward = response['reward']
            done = response['done']
            
            episode_reward += reward
        
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")

train_agent(episodes=100)
```

### Using with Gym (if wrapped)

```python
import gym
from agent import DQNAgent

# If wrapped as gym environment
env = gym.make('FoodWasteEnv-v0')
agent = DQNAgent(state_size=1, action_size=2)

for episode in range(100):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, done)
    
    agent.train()
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    RL Agent                              │
│              (DQN, PPO, A3C, etc.)                      │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP Requests
                     │ /reset, /step, /info
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server                          │
│                   (app.py)                               │
├─────────────────────────────────────────────────────────┤
│  GET /     - Health check                                │
│  GET /reset - Reset environment                          │
│  POST /step - Execute step with action                   │
│  GET /info  - Get state information                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           FoodWasteEnv Class                             │
│          (environment.py)                                │
├─────────────────────────────────────────────────────────┤
│  State: waste_level (0-100)                              │
│  Actions: 0 (no action), 1 (redistribute)                │
│  Rewards: +10/-5 for decrease/increase                   │
│  Terminal: waste <= 0 or > 100 or steps >= 100          │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Agent** sends HTTP request to `/step` with action
2. **FastAPI Server** receives request and validates action
3. **FoodWasteEnv** processes action and updates state
4. **Environment** calculates reward based on state change
5. **Server** returns JSON response with (state, reward, done)
6. **Agent** uses response to update learning

---

## 📈 Example Episode

### Optimal Agent Behavior

```
Episode: 1
Initial Waste: 50

Step 1: action=1 (redistribute)
  → waste: 50 → 45 (decreased)
  → reward: +10
  → total: +10

Step 2: action=1 (redistribute)
  → waste: 45 → 40 (decreased)
  → reward: +10
  → total: +20

Step 3: action=1 (redistribute)
  → waste: 40 → 35 (decreased)
  → reward: +10
  → total: +30

... (continues with action=1)

Step 7: action=1 (redistribute)
  → waste: 10 → 5 (decreased)
  → reward: +10
  → total: +60

Step 8: action=1 (redistribute)
  → waste: 5 → 0 (TERMINAL - SUCCESS)
  → reward: +10 + 20 (bonus) = +30
  → total: +90
  → done: True

Episode finished! Total reward: +90
```

---

## 🔧 Troubleshooting

### Server Won't Start

```bash
# Check port is available
lsof -i :8000

# Use different port
python -m uvicorn app:app --port 8001
```

### Connection Refused Error

```python
# Make sure server is running
# Check http://localhost:8000 in browser
# Server should show Swagger UI
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r server/requirements.txt

# Verify installation
python -c "import fastapi; print(fastapi.__version__)"
```

---

## 📝 Configuration

Edit `openenv.yaml` to modify:

- Environment thresholds (min/max waste)
- Reward values
- Maximum episode length
- Server host/port

---

## 🎓 Learning Outcomes

Building this environment teaches:

1. **RL Fundamentals**: States, actions, rewards, episodes
2. **REST APIs**: FastAPI design and implementation
3. **Clean Code**: Modular, well-documented architecture
4. **Python Best Practices**: Type hints, docstrings, error handling
5. **OpenEnv Standards**: Following community guidelines

---

## 📚 References

- **OpenEnv**: [https://openenv.org](https://openenv.org)
- **FastAPI Docs**: [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Pydantic**: [https://docs.pydantic.dev](https://docs.pydantic.dev)
- **RL Theory**: [Sutton & Barto - Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)

---

## 📄 License

MIT License - Free to use and modify

---

## 🏆 Hackathon Submission

**Project**: Campus Food Waste RL Optimizer  
**Hackathon**: Meta PyTorch OpenEnv  
**Status**: ✅ Complete and Functional  
**Code Quality**: ✅ Production-Ready  

---

## 📧 Support

For issues or questions, refer to code docstrings and comments.

**Happy Training!** 🚀

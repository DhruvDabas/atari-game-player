import torch
import numpy as np
import os
from env import AtariEnv
from dqn import DQNAgent
import pickle

def main():
    TRAIN_MODEL = False  # ❌ Skip training, only run the trained model
    
    env = AtariEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dim = env.get_state().shape[0]
    num_actions = 2

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=50000,
        batch_size=32,
        target_update_freq=500
    )

    # === Load trained model ===
    best_model_path = "weights/best_dqn.pth"
    checkpoint_path = "weights/final_dqn.pth"

    if os.path.exists(best_model_path):
        print("Loading best trained model...")
        agent.load(best_model_path)
    elif os.path.exists(checkpoint_path):
        print("Loading final checkpoint model...")
        agent.load(checkpoint_path)
    else:
        raise FileNotFoundError("No trained model found! Train first before testing.")

    # === Testing trained model ===
    print("\nTesting trained model...")
    env.display = True  # ✅ Enable rendering for testing
    
    for test_episode in range(3):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # ✅ Disable exploration for deterministic play

        while not done and steps < 2000:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            env.render()  # ✅ Show gameplay

        agent.epsilon = old_epsilon
        print(f"Test Episode {test_episode + 1} | Reward: {total_reward:.2f} | Steps: {steps}")

    env.close()

if __name__ == "__main__":
    main()

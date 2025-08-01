import torch
import numpy as np
import os
from env import AtariEnv
from dqn import DQNAgent
import pickle

def main():
    TRAIN_MODEL = True
    
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

    # File paths
    best_model_path = "weights/best_dqn.pth"
    checkpoint_path = "weights/final_dqn.pth"
    replay_buffer_path = "weights/replay_buffer.pth"
    best_reward_path = "weights/best_reward.pkl"

    # === Load model & replay buffer ===
    if os.path.exists(best_model_path):
        print("Loading best model weights...")
        agent.load(best_model_path)
    elif os.path.exists(checkpoint_path):
        print("Loading last checkpoint weights...")
        agent.load(checkpoint_path)

    if os.path.exists(replay_buffer_path):
        print("Loading saved replay buffer...")
        agent.load_replay_buffer(replay_buffer_path)

    # Load best reward for comparison
    best_reward = -float("inf")
    if os.path.exists(best_reward_path):
        with open(best_reward_path, "rb") as f:
            best_reward = pickle.load(f)
        print(f"Loaded best reward: {best_reward:.2f}")

    num_episodes = 1000
    os.makedirs("weights", exist_ok=True)

    # === Training ===
    if TRAIN_MODEL:
        print("Starting training...")
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 2000:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Reward shaping
                if reward == 0.2:
                    shaped_reward = 0.1
                elif reward == 1.0:
                    shaped_reward = 1.0
                elif reward == -1.0:
                    shaped_reward = -2.0
                elif reward == 10.0:
                    shaped_reward = 10.0
                else:
                    shaped_reward = -0.01
                
                agent.store_transition(state, action, shaped_reward, next_state, done)
                
                if len(agent.replay_buffer) > 1000:
                    agent.train_step()
                
                state = next_state
                total_reward += reward
                steps += 1

                if episode % 100 == 0:
                    env.render()

            print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {agent.epsilon:.3f}")

            # Save best model & reward
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(best_model_path)
                agent.save_replay_buffer(replay_buffer_path)
                with open(best_reward_path, "wb") as f:
                    pickle.dump(best_reward, f)
                print(f">>> New best model saved at episode {episode} (Reward: {best_reward:.2f})")

            # Save checkpoint every 50 episodes
            if episode % 50 == 0:
                agent.save(checkpoint_path)
                agent.save_replay_buffer(replay_buffer_path)
                print(f"Checkpoint saved at episode {episode}")

        print("Training complete!")

    else:
        print("Training skipped - TRAIN_MODEL is False")

    # Final save after training
    agent.save(checkpoint_path)
    agent.save_replay_buffer(replay_buffer_path)
    print("Final model & replay buffer saved.")

    # === Testing trained model ===
    print("\nTesting trained model...")
    env.display = True  
    for test_episode in range(3):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Pure exploitation
        
        while not done and steps < 2000:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            env.render()
        
        agent.epsilon = old_epsilon
        print(f"Test Episode {test_episode + 1} | Reward: {total_reward:.2f} | Steps: {steps}")
    
    env.close()

if __name__ == "__main__":
    main()

import torch
import numpy as np
import os
from env import AtariEnv
from dqn import DQNAgent

def main():
    env = AtariEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.get_state().shape[0]
    num_actions = 2

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
        lr=1e-3,
        gamma=0.90,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=1000
    )

    checkpoint_path = "weights/final_dqn.pth"
    replay_buffer_path = "weights/replay_buffer.pth"

    if os.path.exists(checkpoint_path):
        print("Loading saved model weights...")
        agent.load(checkpoint_path)

    if os.path.exists(replay_buffer_path):
        print("Loading saved replay buffer...")
        agent.load_replay_buffer(replay_buffer_path)

    num_episodes = 1000
    best_reward = -float("inf")

    os.makedirs("weights", exist_ok=True)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            env.render()

        print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("weights/best_dqn.pth")
            print(f">>> Saved best model at episode {episode} with reward {best_reward:.2f}")

        if episode % 10 == 0:
            agent.save("weights/final_dqn.pth")
            agent.save_replay_buffer("weights/replay_buffer.pth")
            print(f"Checkpoint saved at episode {episode}")

    agent.save("weights/final_dqn.pth")
    agent.save_replay_buffer("weights/replay_buffer.pth")
    print("Training complete. Final model saved.")
    env.close()

if __name__ == "__main__":
    main()

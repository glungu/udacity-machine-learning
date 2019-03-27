import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.getcwd())

mean_rewards_episodes = np.zeros(390)

mean_rewards_episodes_last10 = np.zeros(390)
rewards = np.array([])
rewards_lengths = []

for i in range(1, 390):
    episode = pd.read_csv(current_dir + '/data_01/episode_' + str(i).zfill(4) + '.csv')
    episode_rewards = episode["reward"]
    mean_rewards_episodes[i] = episode_rewards.mean()

    # last 10 episodes
    rewards_lengths.append(episode_rewards.shape[0])
    rewards = np.concatenate([rewards, episode_rewards])
    if len(rewards_lengths) > 10:
        length = rewards_lengths.pop(0)
        rewards = rewards[length:]
    mean_rewards_episodes_last10[i] = rewards.mean()

plt.plot(mean_rewards_episodes, label='Mean reward by episode')
plt.plot(mean_rewards_episodes_last10, label='Mean reward over last 10 episodes')
plt.ylabel('reward')
plt.title('Mean reward over episodes')
_ = plt.ylim()
plt.show(block=True)
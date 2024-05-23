import numpy as np

import matplotlib.pyplot as plt

def print_reward(name='experiment_gru.txt'):
    # Read data from text file
    with open(name, 'r') as file:
        lines = file.readlines()

    rewards = []

    # Extract numbers after 'reward:'
    for line in lines:
        if 'reward:' in line:
            reward = float(line.split('reward:')[1])
            rewards.append(reward)

    # Plot the rewards
    # Smooth the rewards using a moving average
    window_size = 10
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.plot(smoothed_rewards, label=name[:-4])
    plt.legend()    

print_reward('experiment_mlp.txt')
print_reward()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Visualization')
plt.show()
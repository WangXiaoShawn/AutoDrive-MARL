import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def greedy_evaluation(agent_filename, env_name, num_tests, output_image, device):
    try:
        with open(agent_filename, 'rb') as f:
            checkpoint = pickle.load(f)
        print("Checkpoint keys:", checkpoint.keys())
        if 'actor_state_dicts' not in checkpoint:
            raise KeyError("The checkpoint does not contain 'actor_state_dicts'.")
        actor_state_dicts = checkpoint['actor_state_dicts']
        num_agents = len(actor_state_dicts)
        print(f"Number of agents detected: {num_agents}")
        policy_nets = []
        for i, state_dict in enumerate(actor_state_dicts):
            input_dim = state_dict['fc.0.weight'].shape[1]
            action_dim = state_dict['fc.4.weight'].shape[0]
            policy_net = Actor(input_dim=input_dim, action_dim=action_dim).to(device)
            policy_net.load_state_dict(state_dict)
            policy_net.eval()
            policy_nets.append(policy_net)
            print(f"Loaded policy network for agent {i+1} with input_dim={input_dim} and action_dim={action_dim}")
        env = gym.make(env_name)
        rewards = []
        
        for test_num in range(1, num_tests + 1):
            state = env.reset()
            if isinstance(state, list):
                state = [np.array(s, dtype=np.float32) for s in state]
            else:
                state = [np.array(state, dtype=np.float32)]
            
            total_reward = [0] * num_agents
            done = [False] * num_agents
            step = 0
            max_steps = env._max_steps if hasattr(env, '_max_steps') else 1000
            print(f"\nStarting Test {test_num}/{num_tests}")
            while not all(done) and step < max_steps:
                actions = []
                for i in range(num_agents):
                    if not done[i]:
                        print(f"Agent {i+1} state shape before tensor conversion: {state[i].shape}")
                        state_tensor = torch.tensor(state[i], dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = policy_nets[i](state_tensor)
                            probs = F.softmax(logits, dim=-1)
                            action = torch.argmax(probs, dim=-1).item()
                        actions.append(action)
                    else:
                        actions.append(0)  
                next_state, rewards_env, dones, _ = env.step(actions)
                if isinstance(next_state, list):
                    next_state = [np.array(s, dtype=np.float32) for s in next_state]
                else:
                    next_state = [np.array(next_state, dtype=np.float32)]
                for i in range(num_agents):
                    if not done[i]:
                        total_reward[i] += rewards_env[i]
                state = next_state
                done = dones
                step += 1
            
            avg_total_reward = np.mean(total_reward)
            rewards.append(avg_total_reward)
            print(f"Test {test_num}: Average Total Reward = {avg_total_reward}")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_tests + 1), rewards, marker='o')
        plt.xticks(range(1, num_tests + 1))
        plt.xlabel('Test Number')
        plt.ylabel('Average Total Reward')
        plt.title('Greedy Test Rewards over Evaluations')
        plt.grid(True)
        avg_reward = np.mean(rewards)
        plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Average Reward: {avg_reward:.2f}')
        plt.legend()
        plt.savefig(output_image)
        plt.close()
        print(f"\nPlot saved as '{output_image}' in the current directory.")
    except FileNotFoundError:
        print(f"Error: The file '{agent_filename}' was not found.")
    except pickle.UnpicklingError:
        print("Error: Failed to load the pickle file.")
    except KeyError as e:
        print(f"An error occurred due to missing keys in the checkpoint: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    agent_filename = 'xwang277_jiaxingl_final_TrafficJunction10_a3c_ppo_individual.pkl' 
    env_name = 'ma_gym:TrafficJunction10-v1'  
    num_tests = 10 
    output_image = 'xwang277_jiaxingl_final_TrafficJunction10_a3c_ppo_individual_greedy_test_plot.png'  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    print(f"Using device: {device}")
    try:
        env = gym.make(env_name)
        num_agents = env.n_agents
        action_dim = env.action_space[0].n
        input_dim = env.observation_space[0].shape[0]
        env.close()
        
        print(f"Number of agents: {num_agents}")
        print(f"Action dimension: {action_dim}")
        print(f"Input dimension: {input_dim}")
    except Exception as e:
        print(f"Error initializing environment '{env_name}': {e}")
        return
    greedy_evaluation(
        agent_filename=agent_filename,
        env_name=env_name,
        num_tests=num_tests,
        output_image=output_image,
        device=device
    )

if __name__ == '__main__':
    main()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import json
import pickle
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.92, 0.999)):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

def ensure_shared_grads(local_model, global_model):
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if local_param.grad is not None:
            global_param._grad = local_param.grad.clone()

class IndependentA3CAgent:
    def __init__(self, action_dim, input_dim, env_name, algorithm_name, num_agents, num_workers=8, T_max=5, T_max_global=int(1e7), log_file="a3c_log.json"):
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.env_name = env_name
        self.algorithm_name = algorithm_name
        self.log_file = log_file
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.max_grad_norm = 1.0
        self.num_workers = num_workers
        self.T_max = T_max
        self.T_max_global = T_max_global
        self.device = device
        self.num_agents = num_agents
        self.global_policy_nets = []
        self.global_value_nets = []
        self.optimizers = []
        for _ in range(num_agents):
            policy_net = Actor(input_dim=input_dim, action_dim=action_dim).to(self.device)
            value_net = Critic(input_dim=input_dim).to(self.device)
            policy_net.share_memory()
            value_net.share_memory()
            optimizer = SharedAdam(list(policy_net.parameters()) + list(value_net.parameters()), lr=self.learning_rate)
            self.global_policy_nets.append(policy_net)
            self.global_value_nets.append(value_net)
            self.optimizers.append(optimizer)
        self.global_steps = mp.Value('i', 0)
        self.episode_rewards = mp.Manager().list()
        self.lock = mp.Lock()
        self.best_performance = -float('inf')
        with open(self.log_file, "w") as f:
            json.dump({"logs": []}, f)
        self.model_lock = mp.Lock()
        self.optimizer_locks = [mp.Lock() for _ in range(num_agents)]
        
    def evaluate(self, env, num_episodes=10):
        print(f"Evaluating {num_episodes} episodes using greedy strategy.")
        total_rewards = []
        self.model_lock.acquire()
        print("Pausing training during evaluation.")
        try:
            for net in self.global_policy_nets:
                net.eval()
            
            for i_episode in range(1, num_episodes + 1):
                state = env.reset()
                state = [np.array(s, dtype=np.float32) for s in state]
                total_reward = [0] * self.num_agents
                done = [False] * self.num_agents
                step = 0
                max_steps = env._max_steps if hasattr(env, '_max_steps') else 1000

                with tqdm(total=max_steps, desc=f"Evaluation Episode {i_episode}/{num_episodes}", unit="steps") as pbar:
                    while not all(done) and step < max_steps:
                        actions = []
                        for i in range(self.num_agents):
                            if not done[i]:
                                state_tensor = torch.tensor(state[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    logits = self.global_policy_nets[i](state_tensor)
                                    probs = F.softmax(logits, dim=-1)
                                    action = torch.argmax(probs, dim=-1).cpu().numpy()[0]
                                actions.append(action)
                            else:
                                actions.append(0)  # Placeholder action
                        next_state, rewards, dones, _ = env.step(actions)
                        done = dones
                        for i in range(self.num_agents):
                            total_reward[i] += rewards[i]
                        pbar.update(1)
                        if all(done):
                            avg_total_reward = np.mean(total_reward)
                            pbar.set_postfix({"Total Reward": avg_total_reward, "Steps": step+1})
                            break
                        state = [np.array(s, dtype=np.float32) for s in next_state]
                        step += 1
                avg_reward = np.mean(total_reward)
                print(f"Evaluation Episode {i_episode} completed, Average Total Reward: {avg_reward}")
                total_rewards.append(avg_reward)

            avg_reward_over_episodes = np.mean(total_rewards) if total_rewards else 0
            print(f"Average reward over {num_episodes} evaluation episodes: {avg_reward_over_episodes:.2f}")

            if avg_reward_over_episodes > self.best_performance:
                self.best_performance = avg_reward_over_episodes
                checkpoint_filename = f'best_independent_a3c_checkpoint_{self.env_name.replace("/", "_")}_reward{avg_reward_over_episodes:.2f}.pkl'
                self.save_checkpoint(checkpoint_filename)
                print(f"New best performance achieved! Model saved as {checkpoint_filename}")

            epoch_data = {
                "evaluation_average_reward": avg_reward_over_episodes,
                "global_steps": self.global_steps.value
            }
            self.save_training_log_incremental(epoch_data)

            if total_rewards:
                plt.figure()
                plt.title('Evaluation Results')
                plt.xlabel('Episode')
                plt.ylabel('Average Total Reward')
                plt.plot(total_rewards, marker='o')
                plt.grid(True)
                plt.show()
            else:
                print("No evaluation rewards to plot.")
        finally:
            for net in self.global_policy_nets:
                net.train()
            self.model_lock.release()
            print("Evaluation finished, resuming training.")
    def save_checkpoint(self, filename="checkpoint.pkl"):
        checkpoint = {
            'policy_state_dicts': [net.state_dict() for net in self.global_policy_nets],
            'value_state_dicts': [net.state_dict() for net in self.global_value_nets],
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'episode_rewards': list(self.episode_rewards)
        }
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Model saved to {filename}")
    def load_checkpoint(self, filename="checkpoint.pkl"):
        checkpoint_path = Path(filename)
        if not checkpoint_path.exists():
            print(f"Model file {checkpoint_path.resolve()} does not exist.")
            return
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        for i in range(self.num_agents):
            self.global_policy_nets[i].load_state_dict(checkpoint['policy_state_dicts'][i])
            self.global_value_nets[i].load_state_dict(checkpoint['value_state_dicts'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
        self.episode_rewards.extend(checkpoint.get('episode_rewards', []))
        print(f"Loaded model from {checkpoint_path.resolve()}")
    def save_training_log_incremental(self, epoch_data):
        try:
            with open(self.log_file, 'r') as f:
                training_log = json.load(f)
        except FileNotFoundError:
            training_log = {"logs": []}
        except Exception as e:
            print(f"Error reading training log: {e}")
            training_log = {"logs": []}

        training_log["logs"].append(epoch_data)
        try:
            with open(self.log_file, 'w') as f:
                json.dump(training_log, f, indent=4)
            print(f"Training log updated to {self.log_file}")
        except Exception as e:
            print(f"Error writing training log: {e}")

    def worker(self, worker_id):
        env = gym.make(self.env_name)
        local_policy_nets = []
        local_value_nets = []
        for i in range(self.num_agents):
            policy_net = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            value_net = Critic(input_dim=self.input_dim).to(self.device)
            policy_net.train()
            value_net.train()
            local_policy_nets.append(policy_net)
            local_value_nets.append(value_net)
        while True:
            state = env.reset()
            state = [np.array(s, dtype=np.float32) for s in state]
            done = [False] * self.num_agents
            total_reward = [0] * self.num_agents
            for i in range(self.num_agents):
                local_policy_nets[i].load_state_dict(self.global_policy_nets[i].state_dict())
                local_value_nets[i].load_state_dict(self.global_value_nets[i].state_dict())
            while not all(done):
                log_probs = [[] for _ in range(self.num_agents)]
                values = [[] for _ in range(self.num_agents)]
                rewards = [[] for _ in range(self.num_agents)]
                dones_list = [[] for _ in range(self.num_agents)]

                for _ in range(self.T_max):
                    actions = []
                    for i in range(self.num_agents):
                        if not done[i]:
                            state_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(self.device)
                            logits = local_policy_nets[i](state_tensor)
                            value = local_value_nets[i](state_tensor)
                            probs = F.softmax(logits, dim=-1)
                            dist = torch.distributions.Categorical(probs)
                            action = dist.sample()
                            log_prob = dist.log_prob(action)

                            actions.append(action.item())

                            log_probs[i].append(log_prob)
                            values[i].append(value)
                        else:
                            actions.append(0) 

                    next_state, rewards_env, dones, _ = env.step(actions)
                    next_state = [np.array(s, dtype=np.float32) for s in next_state]

                    for i in range(self.num_agents):
                        if not done[i]:
                            rewards[i].append(torch.tensor(rewards_env[i], dtype=torch.float32).to(self.device))
                            dones_list[i].append(dones[i])
                            total_reward[i] += rewards_env[i]
                    state = next_state
                    done = dones  

                    with self.global_steps.get_lock():
                        self.global_steps.value += 1
                        if self.global_steps.value >= self.T_max_global:
                            env.close()
                            return
                    if all(done):
                        break
                for i in range(self.num_agents):
                    if len(rewards[i]) == 0:
                        continue 
                    R = torch.zeros(1, 1).to(self.device)
                    if not done[i]:
                        state_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(self.device)
                        R = local_value_nets[i](state_tensor).detach()
                    values[i].append(R)
                    policy_loss = 0
                    value_loss = 0
                    for j in reversed(range(len(rewards[i]))):
                        mask = (1 - torch.tensor(dones_list[i][j], dtype=torch.float32).unsqueeze(0).to(self.device))
                        R = rewards[i][j].unsqueeze(0) + self.gamma * R * mask
                        advantage = R - values[i][j]
                        policy_loss -= (log_probs[i][j] * advantage.detach().squeeze()).mean()
                        value_loss += 0.5 * F.mse_loss(values[i][j], R.detach())
                    total_loss = policy_loss + value_loss
                    self.optimizers[i].zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(local_policy_nets[i].parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(local_value_nets[i].parameters(), self.max_grad_norm)
                    ensure_shared_grads(local_policy_nets[i], self.global_policy_nets[i])
                    ensure_shared_grads(local_value_nets[i], self.global_value_nets[i])
                    with self.optimizer_locks[i]:
                        self.optimizers[i].step()

                if all(done):
                    avg_total_reward = np.mean(total_reward)
                    print(f"Worker: {worker_id}, Global Steps: {self.global_steps.value}, Total Reward: {avg_total_reward}")
                    self.episode_rewards.append(avg_total_reward)
                    break
    def train_a3c(self, evaluation_interval=100000, num_evaluations=10):
        processes = []
        for worker_id in range(self.num_workers):
            p = mp.Process(target=self.worker, args=(worker_id,))
            p.start()
            processes.append(p)
        evaluations_done = 0
        next_evaluation_step = evaluation_interval
        
        while evaluations_done < num_evaluations:
            time.sleep(5)
            with self.global_steps.get_lock():
                current_steps = self.global_steps.value
            if current_steps >= next_evaluation_step:
                try:
                    eval_env = gym.make(self.env_name)
                    self.evaluate(eval_env, num_episodes=5)
                    eval_env.close()
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                evaluations_done += 1
                next_evaluation_step += evaluation_interval
        
        for p in processes:
            p.join()
        
        final_checkpoint = f"best_independent_a3c_checkpoint_{self.env_name.replace('/', '_')}_{self.algorithm_name}.pkl"
        self.save_checkpoint(final_checkpoint)
def main():
    env_name = 'ma_gym:TrafficJunction10-v1'
    env = gym.make(env_name)
    num_agents = env.n_agents
    action_dim = env.action_space[0].n
    input_dim = env.observation_space[0].shape[0]
    env.close()

    agent = IndependentA3CAgent(
        action_dim=action_dim,
        input_dim=input_dim,
        env_name=env_name,
        algorithm_name='IndependentA3C',
        num_agents=num_agents,
        num_workers=8,
        T_max=20,
        T_max_global=int(1e7),
        log_file="independent_a3c_multiagent_log.json"
    )
    agent.train_a3c(evaluation_interval=5000, num_evaluations=10000)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

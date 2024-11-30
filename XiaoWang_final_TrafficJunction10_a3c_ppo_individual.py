import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        logits = self.fc(x)
        return logits

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
            global_param._grad = local_param.grad

class MultiAgentPPOA3C:
    def __init__(self, action_dim, input_dim, env_name, algorithm_name, num_agents, num_workers=8, T_max=2048, K_epochs=4, eps_clip=0.2, gamma=0.99, lr=1e-4, entropy_coeff=0.01, T_max_global=int(1e7), log_file="multiagent_ppo_a3c_log.json"):
        self.action_dim = action_dim
        self.input_dim = input_dim  
        self.env_name = env_name
        self.algorithm_name = algorithm_name
        self.log_file = log_file  
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr = lr
        self.entropy_coeff = entropy_coeff
        self.num_workers = num_workers
        self.T_max = T_max
        self.T_max_global = T_max_global
        self.device = device
        self.num_agents = num_agents

        self.global_actors = []
        self.global_critics = []
        self.optimizer_actors = []
        self.optimizer_critics = []

        for _ in range(self.num_agents):
            actor = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            critic = Critic(input_dim=self.input_dim).to(self.device)
            actor.share_memory()
            critic.share_memory()
            self.global_actors.append(actor)
            self.global_critics.append(critic)

            optimizer_actor = SharedAdam(actor.parameters(), lr=self.lr)
            optimizer_critic = SharedAdam(critic.parameters(), lr=self.lr)
            self.optimizer_actors.append(optimizer_actor)
            self.optimizer_critics.append(optimizer_critic)

        self.global_steps = mp.Value('i', 0)
        self.episode_rewards = mp.Manager().list()
        self.lock = mp.Lock()
        self.best_performance = -float('inf')

        with open(self.log_file, "w") as f:
            json.dump({"logs": []}, f)

        self.model_lock = mp.Lock()
        self.optimizer_locks = [mp.Lock() for _ in range(self.num_agents)]

    def evaluate(self, env, num_episodes=10):
        print(f"Evaluating {num_episodes} episodes using greedy policy.")
        total_rewards = []
        self.model_lock.acquire()
        print("Pausing training during evaluation.")
        try:
            for actor in self.global_actors:
                actor.eval()

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
                                    logits = self.global_actors[i](state_tensor)
                                    probs = F.softmax(logits, dim=-1)
                                    action = torch.argmax(probs, dim=-1).cpu().numpy()[0]
                                actions.append(action)
                            else:
                                actions.append(0)
                        next_state, rewards_env, dones, _ = env.step(actions)
                        next_state = [np.array(s, dtype=np.float32) for s in next_state]
                        done = dones
                        for i in range(self.num_agents):
                            total_reward[i] += rewards_env[i]
                        pbar.update(1)
                        if all(done):
                            avg_total_reward = np.mean(total_reward)
                            pbar.set_postfix({"Total Reward": avg_total_reward, "Steps": step+1})
                            break
                        state = next_state
                        step += 1
                avg_reward = np.mean(total_reward)
                print(f"Evaluation Episode {i_episode} completed, Average Total Reward: {avg_reward}")
                total_rewards.append(avg_reward)

            avg_reward_over_episodes = np.mean(total_rewards) if total_rewards else 0
            print(f"Average Reward over {num_episodes} episodes: {avg_reward_over_episodes:.2f}")

            if avg_reward_over_episodes > self.best_performance:
                self.best_performance = avg_reward_over_episodes
                checkpoint_filename = f'best_multiagent_ppo_a3c_checkpoint_{self.env_name.replace("/", "_")}_reward{avg_reward_over_episodes:.2f}.pkl'
                self.save_checkpoint(checkpoint_filename)
                print(f"New best performance! Model saved as {checkpoint_filename}")

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
            for actor in self.global_actors:
                actor.train()
            self.model_lock.release()
            print("Evaluation completed, resuming training.")

    def save_checkpoint(self, filename="checkpoint.pkl"):
        checkpoint = {
            'actor_state_dicts': [actor.state_dict() for actor in self.global_actors],
            'critic_state_dicts': [critic.state_dict() for critic in self.global_critics],
            'actor_optimizer_state_dicts': [opt.state_dict() for opt in self.optimizer_actors],
            'critic_optimizer_state_dicts': [opt.state_dict() for opt in self.optimizer_critics],
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
            self.global_actors[i].load_state_dict(checkpoint['actor_state_dicts'][i])
            self.global_critics[i].load_state_dict(checkpoint['critic_state_dicts'][i])
            self.optimizer_actors[i].load_state_dict(checkpoint['actor_optimizer_state_dicts'][i])
            self.optimizer_critics[i].load_state_dict(checkpoint['critic_optimizer_state_dicts'][i])
        self.episode_rewards.extend(checkpoint.get('episode_rewards', []))
        print(f"Model loaded from {checkpoint_path.resolve()}")

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
        local_actors = []
        local_critics = []
        for _ in range(self.num_agents):
            actor = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            critic = Critic(input_dim=self.input_dim).to(self.device)
            actor.train()
            critic.train()
            local_actors.append(actor)
            local_critics.append(critic)

        while True:
            state = env.reset()
            state = [np.array(s, dtype=np.float32) for s in state]
            done = [False] * self.num_agents
            total_reward = [0] * self.num_agents

            for i in range(self.num_agents):
                local_actors[i].load_state_dict(self.global_actors[i].state_dict())
                local_critics[i].load_state_dict(self.global_critics[i].state_dict())

            memory = [{} for _ in range(self.num_agents)]
            for i in range(self.num_agents):
                memory[i]['states'] = []
                memory[i]['actions'] = []
                memory[i]['log_probs'] = []
                memory[i]['rewards'] = []
                memory[i]['dones'] = []
                memory[i]['values'] = []

            steps = 0
            while not all(done) and steps < self.T_max:
                actions = []
                for i in range(self.num_agents):
                    if not done[i]:
                        state_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(self.device)
                        logits = local_actors[i](state_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        value = local_critics[i](state_tensor)

                        actions.append(action.item())

                        memory[i]['states'].append(state[i])
                        memory[i]['actions'].append(action.item())
                        memory[i]['log_probs'].append(log_prob.item())
                        memory[i]['values'].append(value.item())
                    else:
                        actions.append(0)

                next_state, rewards_env, dones, _ = env.step(actions)
                next_state = [np.array(s, dtype=np.float32) for s in next_state]

                for i in range(self.num_agents):
                    if not done[i]:
                        memory[i]['rewards'].append(rewards_env[i])
                        memory[i]['dones'].append(dones[i])
                        total_reward[i] += rewards_env[i]
                state = next_state
                done = dones.copy()
                steps += 1

                with self.global_steps.get_lock():
                    self.global_steps.value += 1
                    if self.global_steps.value >= self.T_max_global:
                        env.close()
                        return

            for i in range(self.num_agents):
                if len(memory[i]['rewards']) == 0:
                    continue

                if not done[i]:
                    state_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(self.device)
                    next_value = local_critics[i](state_tensor).item()
                else:
                    next_value = 0

                advantages, returns = self.compute_gae(memory[i]['rewards'], memory[i]['dones'], memory[i]['values'], next_value)

                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
                states = torch.FloatTensor(np.array(memory[i]['states'])).to(self.device)
                actions_tensor = torch.LongTensor(memory[i]['actions']).to(self.device)
                old_log_probs = torch.FloatTensor(memory[i]['log_probs']).to(self.device)

                for _ in range(self.K_epochs):
                    logits = self.global_actors[i](states)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions_tensor)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                    values_pred = self.global_critics[i](states).squeeze()
                    critic_loss = F.mse_loss(values_pred, returns)

                    self.optimizer_actors[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.global_actors[i].parameters(), 0.5)
                    with self.optimizer_locks[i]:
                        self.optimizer_actors[i].step()

                    self.optimizer_critics[i].zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.global_critics[i].parameters(), 0.5)
                    with self.optimizer_locks[i]:
                        self.optimizer_critics[i].step()

            avg_total_reward = np.mean(total_reward)
            print(f"Worker: {worker_id}, Global Steps: {self.global_steps.value}, Average Total Reward: {avg_total_reward}")
            self.episode_rewards.append(avg_total_reward)

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * (next_value if not dones[step] else 0) - values[step]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
            next_value = values[step]
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def train(self, evaluation_interval=100000, num_evaluations=10):
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
        
        final_checkpoint = f"best_multiagent_ppo_a3c_checkpoint_{self.env_name.replace('/', '_')}_{self.algorithm_name}.pkl"
        self.save_checkpoint(final_checkpoint)

def main():
    env_name = 'ma_gym:TrafficJunction10-v1'
    env = gym.make(env_name)
    num_agents = env.n_agents
    action_dim = env.action_space[0].n
    input_dim = env.observation_space[0].shape[0]
    env.close()

    agent = MultiAgentPPOA3C(
        action_dim=action_dim,
        input_dim=input_dim,
        env_name=env_name,
        algorithm_name='MultiAgentPPO_A3C',
        num_agents=num_agents,
        num_workers=8,
        T_max=2048,
        K_epochs=4,
        eps_clip=0.2,
        gamma=0.99,
        lr=1e-4,
        entropy_coeff=0.01,
        T_max_global=int(1e7),
        log_file="multiagent_ppo_a3c_v10_log.json"
    )
    agent.train(evaluation_interval=5000, num_evaluations=20000)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

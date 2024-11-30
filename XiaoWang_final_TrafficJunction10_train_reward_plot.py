import os
import json
import matplotlib.pyplot as plt

def load_log_file(log_file, max_steps):
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        logs = data.get("logs", [])
        steps = []
        rewards = []
        for entry in logs:
            step = entry.get("global_steps")
            reward = entry.get("evaluation_average_reward")
            if step is not None and reward is not None and step <= max_steps:
                steps.append(step)
                rewards.append(reward)
        
        return steps, rewards
    except FileNotFoundError:
        print(f"Error: The file '{log_file}' was not found.")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file '{log_file}'.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred while loading '{log_file}': {e}")
        return [], []

def plot_logs(log_files, labels, output_image, max_steps):
    plt.figure(figsize=(12, 8))
    
    # Updated colors for better distinction and swap A3C Share and Reward Upper Bound colors
    colors = [
        'blue',        # A3C Individual
        'orange',      # PPO Share
        'green',       # PPO Individual
        'gray'         # A3C Share (changed to gray)
    ]
    
    for log_file, label, color in zip(log_files, labels, colors):
        steps, rewards = load_log_file(log_file, max_steps)
        if not steps or not rewards:
            print(f"Warning: No data to plot for '{log_file}'.")
            continue
        plt.plot(steps, rewards, label=label, color=color)
    
    # Add a horizontal dashed line for Reward Upper Bound with red color
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Reward Upper Bound')
    
    plt.xlabel('Global Steps')
    plt.ylabel('Average Evaluation Reward')
    plt.title('Comparison of Different Algorithms on TrafficJunction10-v1 (Truncated)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image)
    plt.show()
    print(f"Plot saved as '{output_image}'.")

def main():
    log_files = [
        'xwang277_jiaxingl_final_TrafficJunction10_a3c_individual_log.json',
        'xwang277_jiaxingl_final_TrafficJunction10_a3c_ppo_share_log.json',
        'xwang277_jiaxingl_final_TrafficJunction10_a3c_ppo_individual_log.json',
        'xwang277_jiaxingl_final_TrafficJunction10_a3c_share_log.json'
    ]
    
    labels = [
        'A3C Individual',
        'PPO Share',
        'PPO Individual',
        'A3C Share'
    ]
    
    max_steps = 0.4 * 1e7
    output_image = 'xwang277_jiaxingl_final_TrafficJunction10_train_reward.png'
    
    missing_files = [f for f in log_files if not os.path.isfile(f)]
    if missing_files:
        print("Error: The following log files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    plot_logs(log_files, labels, output_image, max_steps)

if __name__ == '__main__':
    main()
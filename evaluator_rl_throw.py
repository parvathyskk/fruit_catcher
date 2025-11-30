import matplotlib.pyplot as plt
import numpy as np
import os

class Evaluator:
    def __init__(self, save_dir="evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Per-step metrics
        self.losses = []
        self.epsilons = []
        
        # Per-episode metrics
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lives = []
        
        self.episode_count = 0

    def log_step(self, loss, epsilon):
        """Log metrics for a single training step."""
        if loss is not None:
            self.losses.append(loss)
        self.epsilons.append(epsilon)

    def log_episode(self, total_reward, final_score, lives_left):
        """Log metrics at the end of an episode."""
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        self.episode_scores.append(final_score)
        self.episode_lives.append(lives_left)
        
        print(f"[Episode {self.episode_count}] Score: {final_score}, Lives: {lives_left}, Avg Loss: {np.mean(self.losses[-10:]) if self.losses else 0:.4f}")

    def plot(self):
        """Generate and save plots of the metrics."""
        try:
            plt.figure(figsize=(12, 8))
            
            # 1. Scores
            plt.subplot(2, 2, 1)
            plt.plot(self.episode_scores, label="Score")
            plt.title("Episode Scores")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.grid(True)
            
            # 2. Rewards
            plt.subplot(2, 2, 2)
            plt.plot(self.episode_rewards, label="Total Reward", color="orange")
            plt.title("Episode Total Rewards")
            plt.xlabel("Episode")
            plt.grid(True)
            
            # 3. Loss
            plt.subplot(2, 2, 3)
            if self.losses:
                # Plot moving average of loss to reduce noise
                window = 50
                if len(self.losses) > window:
                    avg_loss = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                    plt.plot(avg_loss, label="Loss (MA)", color="red")
                else:
                    plt.plot(self.losses, label="Loss", color="red")
            plt.title("Training Loss")
            plt.xlabel("Step")
            plt.yscale("log")
            plt.grid(True)
            
            # 4. Epsilon
            plt.subplot(2, 2, 4)
            plt.plot(self.epsilons, label="Epsilon", color="green")
            plt.title("Epsilon Decay")
            plt.xlabel("Step")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "training_metrics.png"))
            plt.close()
            print(f"Plots saved to {self.save_dir}/training_metrics.png")
            
        except Exception as e:
            print(f"Failed to plot metrics: {e}")

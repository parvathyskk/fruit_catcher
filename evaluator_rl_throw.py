import matplotlib.pyplot as plt
import numpy as np
import os


class Evaluator:
    def __init__(self, save_dir="evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.losses = []
        self.epsilons = []
<<<<<<< HEAD
        self.q_values = []
=======
        self.max_q_values = []
        self.q_deltas = []
>>>>>>> b7d0811 (graph changes ppp)
        
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lives = []
        
        self.episode_count = 0

<<<<<<< HEAD
    def log_step(self, loss, epsilon, q_value):
        """Log metrics for a single training step."""
        if loss is not None:
            self.losses.append(loss)
        self.epsilons.append(epsilon)
        if q_value is not None:
            self.q_values.append(q_value)
=======
    def log_step(self, loss, epsilon, max_q=None, q_delta=None):
        if loss is not None:
            self.losses.append(loss)
        self.epsilons.append(epsilon)
        if max_q is not None:
            self.max_q_values.append(max_q)
        if q_delta is not None:
            self.q_deltas.append(q_delta)
>>>>>>> b7d0811 (graph changes ppp)

    def log_episode(self, total_reward, final_score, lives_left):
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        self.episode_scores.append(final_score)
        self.episode_lives.append(lives_left)
<<<<<<< HEAD
        
        avg_q = np.mean(self.q_values[-10:]) if self.q_values else 0.0
        print(f"[Episode {self.episode_count}] Score: {final_score}, Lives: {lives_left}, Avg Loss: {np.mean(self.losses[-10:]) if self.losses else 0:.4f}, Avg Q: {avg_q:.4f}")
=======
>>>>>>> b7d0811 (graph changes ppp)

    def plot(self, show_plots=False):
        try:
<<<<<<< HEAD
            plt.figure(figsize=(15, 10))
            
            # 1. Scores
            plt.subplot(2, 3, 1)
            plt.plot(self.episode_scores, label="Score")
            plt.title("Episode Scores")
=======
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle("Thrower Agent - Training Metrics", fontsize=16, fontweight="bold")
            
            plt.subplot(2, 3, 1)
            plt.plot(self.episode_scores, linewidth=2, marker="o", markersize=4, color="teal")
            plt.title("Episode Scores", fontweight="bold")
>>>>>>> b7d0811 (graph changes ppp)
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.grid(True, alpha=0.3)
            
<<<<<<< HEAD
            # 2. Rewards
            plt.subplot(2, 3, 2)
            plt.plot(self.episode_rewards, label="Total Reward", color="orange")
            plt.title("Episode Total Rewards")
=======
            plt.subplot(2, 3, 2)
            plt.plot(self.episode_rewards, linewidth=2, marker="s", markersize=4, color="orange")
            plt.title("Episode Total Rewards", fontweight="bold")
>>>>>>> b7d0811 (graph changes ppp)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            
<<<<<<< HEAD
            # 3. Loss
=======
>>>>>>> b7d0811 (graph changes ppp)
            plt.subplot(2, 3, 3)
            if self.losses:
                window = 100
                if len(self.losses) > window:
                    avg_loss = np.convolve(self.losses, np.ones(window)/window, mode="valid")
                    plt.plot(avg_loss, linewidth=1.5, label="Loss (Moving Avg)", color="red")
                else:
                    plt.plot(self.losses, linewidth=1.5, label="Loss", color="red")
                plt.yscale("log")
                plt.legend()
            plt.title("Training Loss per Step (log scale)", fontweight="bold")
            plt.xlabel("Step")
            plt.grid(True, alpha=0.3)
            
<<<<<<< HEAD
            # 4. Epsilon
            plt.subplot(2, 3, 4)
            plt.plot(self.epsilons, label="Epsilon", color="green")
            plt.title("Epsilon Decay")
=======
            plt.subplot(2, 3, 5)
            plt.plot(self.episode_lives, linewidth=2, marker="^", markersize=4, color="purple")
            plt.title("Lives Remaining per Episode", fontweight="bold")
            plt.xlabel("Episode")
            plt.ylabel("Lives")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 6)
            if self.max_q_values:
                qvals = np.array(self.max_q_values)
                window = 50
                if len(qvals) > window:
                    q_ma = np.convolve(qvals, np.ones(window)/window, mode="valid")
                    plt.plot(q_ma, linewidth=1.5, label="Q-value (MA)", color="brown")
                else:
                    plt.plot(qvals, linewidth=1.0, label="Q-value", color="brown")
                plt.legend()
            plt.title("Q-value Convergence", fontweight="bold")
>>>>>>> b7d0811 (graph changes ppp)
            plt.xlabel("Step")
            plt.ylabel("Q-Value")
            plt.grid(True, alpha=0.3)
            
            # 5. Q-Values
            plt.subplot(2, 3, 5)
            if self.q_values:
                # Plot moving average of Q-values
                window = 50
                if len(self.q_values) > window:
                    avg_q = np.convolve(self.q_values, np.ones(window)/window, mode='valid')
                    plt.plot(avg_q, label="Avg Q-Value (MA)", color="purple")
                else:
                    plt.plot(self.q_values, label="Avg Q-Value", color="purple")
            plt.title("Average Q-Values")
            plt.xlabel("Step")
            plt.grid(True)
            
            plt.tight_layout()
            output_path = os.path.join(self.save_dir, "thrower_training_metrics.png")
            plt.savefig(output_path, dpi=150)
            if show_plots:
                plt.show()
            plt.close()
            print(f"Thrower metrics saved: {output_path}")
            
        except Exception as e:
            print(f"Failed to plot thrower metrics: {e}")

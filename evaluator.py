import matplotlib.pyplot as plt
import numpy as np
import os


class Evaluator:
    def __init__(self, save_dir="evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=["episode", "score", "lives", "avg_loss", "epsilon", "avg_q"])
            df.to_csv(self.log_file, index=False)

    def log_episode(self, episode, score, lives, avg_loss, epsilon, avg_q):
        """Log metrics for a single episode."""
        record = {
            "episode": episode,
            "score": score,
            "lives": lives,
            "avg_loss": avg_loss,
            "epsilon": epsilon,
            "avg_q": avg_q
        }
        self.metrics.append(record)
        
        self.episode_scores = []
        self.episode_lives = []
        self.episode_avg_losses = []
        self.episode_rewards = []
        
        print(f"Episode {episode}: Score={score}, Lives={lives}, Avg Loss={avg_loss:.4f}, Epsilon={epsilon:.4f}, Avg Q={avg_q:.4f}")

    def log_step(self, loss, epsilon, max_q=None, q_delta=None):
        if loss is not None:
            self.step_losses.append(loss)
        self.step_epsilons.append(epsilon)
        if max_q is not None:
            self.step_max_q_values.append(max_q)
        if q_delta is not None:
            self.step_q_deltas.append(q_delta)

    def log_episode(self, episode, score, lives, avg_loss, epsilon, total_reward=None):
        self.episode_count += 1
        if total_reward is not None:
            self.episode_rewards.append(total_reward)
        self.episode_scores.append(score)
        self.episode_lives.append(lives)
        self.episode_avg_losses.append(avg_loss)

    def plot(self, show_plots=False):
        try:
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle("Catcher Agent - Training Metrics", fontsize=16, fontweight="bold")
            
            plt.subplot(2, 3, 1)
            plt.plot(self.episode_scores, linewidth=2, marker="o", markersize=4, color="blue")
            plt.title("Episode Scores", fontweight="bold")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 2)
            if self.episode_rewards:
                plt.plot(self.episode_rewards, linewidth=2, marker="s", markersize=4, color="green")
                plt.title("Episode Total Rewards", fontweight="bold")
                plt.ylabel("Reward")
            else:
                plt.plot(self.episode_lives, linewidth=2, marker="s", markersize=4, color="green")
                plt.title("Lives Remaining", fontweight="bold")
                plt.ylabel("Lives")
            plt.xlabel("Episode")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 3)
            if self.episode_avg_losses:
                plt.plot(self.episode_avg_losses, linewidth=2, color="red")
                plt.yscale("log")
            plt.title("Avg Loss per Episode (log scale)", fontweight="bold")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 4)
            if self.step_losses:
                window = 100
                if len(self.step_losses) > window:
                    avg_loss = np.convolve(self.step_losses, np.ones(window)/window, mode="valid")
                    plt.plot(avg_loss, linewidth=1.5, label="Loss (Moving Avg)", color="purple")
                else:
                    plt.plot(self.step_losses, linewidth=1.5, label="Loss", color="purple")
                plt.yscale("log")
                plt.legend()
            plt.title("Training Loss per Step", fontweight="bold")
            plt.xlabel("Step")
            plt.grid(True, alpha=0.3)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

            # Plot Score
            ax1.plot(df['episode'], df['score'], label='Score', color='blue')
            ax1.set_title('Score per Episode')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            ax1.grid(True)

            # Plot Loss
            ax2.plot(df['episode'], df['avg_loss'], label='Avg Loss', color='red')
            ax2.set_title('Average Loss per Episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.grid(True)

            # Plot Avg Q-Value
            ax3.plot(df['episode'], df['avg_q'], label='Avg Q-Value', color='green')
            ax3.set_title('Average Q-Value per Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Avg Q-Value')
            ax3.grid(True)

            plt.tight_layout()
            output_path = os.path.join(self.save_dir, "catcher_training_metrics.png")
            plt.savefig(output_path, dpi=150)
            if show_plots:
                plt.show()
            plt.close()
            print(f"Catcher metrics saved: {output_path}")
            
        except Exception as e:
            print(f"Failed to plot catcher metrics: {e}")

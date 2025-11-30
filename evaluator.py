import pandas as pd
import matplotlib.pyplot as plt
import os

class Evaluator:
    def __init__(self, log_file="training_log.csv", plot_file="training_plot.png"):
        self.log_file = log_file
        self.plot_file = plot_file
        self.metrics = []
        
        # Initialize log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=["episode", "score", "lives", "avg_loss", "epsilon"])
            df.to_csv(self.log_file, index=False)

    def log_episode(self, episode, score, lives, avg_loss, epsilon):
        """Log metrics for a single episode."""
        record = {
            "episode": episode,
            "score": score,
            "lives": lives,
            "avg_loss": avg_loss,
            "epsilon": epsilon
        }
        self.metrics.append(record)
        
        # Append to CSV immediately
        df = pd.DataFrame([record])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        
        print(f"Episode {episode}: Score={score}, Lives={lives}, Avg Loss={avg_loss:.4f}, Epsilon={epsilon:.4f}")

    def plot_metrics(self):
        """Generate and save plots for training metrics."""
        if not os.path.exists(self.log_file):
            return

        try:
            df = pd.read_csv(self.log_file)
            if len(df) < 2:
                return

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

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

            plt.tight_layout()
            plt.savefig(self.plot_file)
            plt.close()
        except Exception as e:
            print(f"Error plotting metrics: {e}")

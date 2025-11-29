from env_rlcatch import FruitCatcherEnv
from dqn_agent import DQNAgent
from evaluator import Evaluator
import pygame
import numpy as np
import random
import os

# Configuration
TOTAL_EPISODES = 120
MODEL_PATH = "dqn_model.pth"

# Initialize
env = FruitCatcherEnv()
agent = DQNAgent()
evaluator = Evaluator()

# Load existing model if available
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    agent.load(MODEL_PATH)
else:
    print("No saved model found. Starting from scratch.")

running = True
clock = pygame.time.Clock()

episode = 1
episode_loss = []
state = env.reset()

print(f"--- Starting Automated Training for {TOTAL_EPISODES} Episodes ---")

while running and episode <= TOTAL_EPISODES:
    # Run faster than human mode, but limit FPS to avoid CPU hogging
    # Set to 0 for max speed, or keep 30/60 to watch it
    clock.tick(100) 

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # -----------------------------------
    # AUTO THROWER (Simulates Human)
    # -----------------------------------
    if env.object_type is None:
        # Randomly choose object (1-4: fruits, 5: bomb)
        obj_id = random.randint(1, 5)
        # Randomly choose position
        throw_x = random.randint(0, env.grid_width - 1)
        env.human_spawn(obj_id, throw_x)

    # -----------------------------------
    # RL AGENT CONTROL
    # -----------------------------------
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    reward = reward / 10.0  # optional scaling

    agent.remember(state, action, reward, next_state, done)
    
    # Train and track loss
    loss = agent.train_step()
    if loss is not None:
        episode_loss.append(loss)
        
    # Decay epsilon (exploration rate)
    agent.decay()

    state = next_state

    # -----------------------------------
    # RENDER (Optional: comment out for faster training)
    # -----------------------------------
    env.draw()

    # -----------------------------------
    # EPISODE END
    # -----------------------------------
    if done:
        # Calculate episode metrics
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        
        # Log metrics
        evaluator.log_episode(
            episode=episode,
            score=env.score,
            lives=env.lives,
            avg_loss=avg_loss,
            epsilon=agent.epsilon
        )
        
        # Plot every 5 episodes
        if episode % 5 == 0:
            evaluator.plot_metrics()
            
        print(f"Episode {episode}/{TOTAL_EPISODES} Finished. Score: {env.score}")
        
        # Save model every episode
        agent.save(MODEL_PATH)

        state = env.reset()
        agent.update_target()
        
        episode += 1
        episode_loss = []

print("--- Automated Training Completed ---")
pygame.quit()

from env_rlcatch import FruitCatcherEnv
from dqn_agent import DQNAgent
from evaluator import Evaluator
import pygame
import numpy as np
import os

env = FruitCatcherEnv()
agent = DQNAgent()
evaluator = Evaluator()

# Load existing model if available
MODEL_PATH = "dqn_model.pth"
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

print(f"--- Episode {episode} Start ---")

while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    mx, my = pygame.mouse.get_pos()
    throw_x = mx // env.scale

    # -----------------------------------
    # HUMAN OBJECT THROWING
    # -----------------------------------
    if env.object_type is None:
        if keys[pygame.K_1]: env.human_spawn(1, throw_x)
        if keys[pygame.K_2]: env.human_spawn(2, throw_x)
        if keys[pygame.K_3]: env.human_spawn(3, throw_x)
        if keys[pygame.K_4]: env.human_spawn(4, throw_x)
        if keys[pygame.K_5]: env.human_spawn(5, throw_x)

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
        
    agent.decay()

    state = next_state

    # -----------------------------------
    # RENDER
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
            
        print(f"--- Episode {episode} Over ---")
        print(f"Final Score: {env.score}")
        
        # Save model
        agent.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        state = env.reset()          # do not quit
        agent.update_target()        # optional but very good for stability
        
        episode += 1
        episode_loss = []
        print(f"--- Episode {episode} Start ---")
        continue

pygame.quit()

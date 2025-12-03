from env_rlcatch import FruitCatcherEnv
from dqn_agent import DQNAgent
from evaluator import Evaluator
import pygame
import numpy as np
import random
import os

TOTAL_EPISODES = 20
MODEL_PATH = "dqn_model.pth"
RENDER_INTERVAL = 1
LOG_INTERVAL = 5
Q_CONV_WINDOW = 100
Q_CONV_THRESHOLD = 0.001
Q_CONV_PATIENCE = 5

print("\nFRUIT CATCHER - CATCHER AGENT TRAINING\n")

env = FruitCatcherEnv()
agent = DQNAgent()
evaluator = Evaluator()

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    agent.load(MODEL_PATH)
else:
    print("Starting from scratch.\n")

print(f"Config: Episodes={TOTAL_EPISODES}, Batch Size={agent.batch_size}\n")

running = True
clock = pygame.time.Clock()

# convergence counter for Q-value stability
q_conv_counter = 0

episode = 1
episode_loss = []
episode_q = []
state = env.reset()

print(f"--- Episode {episode} Start ---")

while running and episode <= TOTAL_EPISODES:
    clock.tick(100)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    
    # AUTO THROWER (Simulates Human)
   
    if env.object_type is None:
        obj_id = random.randint(1, 5)
        throw_x = random.randint(0, env.grid_width - 1)
        env.human_spawn(obj_id, throw_x)


    # RL AGENT CONTROL
    
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    reward = reward / 10.0
    episode_reward += reward

    agent.remember(state, action, reward, next_state, done)
    
    # Train and track loss
    result = agent.train_step()
    if result is not None:
        loss, avg_q = result
        episode_loss.append(loss)
        episode_q.append(avg_q)
        
    # Decay epsilon (exploration rate)
    agent.decay()

    state = next_state

   #render___may
    env.draw()

    if done:
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        avg_q_val = np.mean(episode_q) if episode_q else 0.0
        
        # Log metrics
        evaluator.log_episode(
            episode=episode,
            score=env.score,
            lives=env.lives,
            avg_loss=avg_loss,
            epsilon=agent.epsilon,
            avg_q=avg_q_val
        )

        status = "PASS" if env.score > 0 else "FAIL"
        print(f"{status} Episode {episode:3d}/{TOTAL_EPISODES} | "
              f"Score: {env.score:4d} | Lives: {env.lives:2d} | "
              f"Loss: {avg_loss:.6f} | Epsilon: {agent.epsilon:.4f}")

        if episode % LOG_INTERVAL == 0:
            evaluator.plot()

        agent.save(MODEL_PATH)

        state = env.reset()
        agent.update_target()

        episode += 1
        episode_loss = []
        episode_q = []

print("\nTRAINING COMPLETED\n")
evaluator.plot()
pygame.quit()

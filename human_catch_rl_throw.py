import pygame
from env import FruitCatcherEnv
from dqn_spawner import SpawnerAgent
import torch
import time
import numpy as np
from evaluator import Evaluator 

env = FruitCatcherEnv()
agent = SpawnerAgent()
evaluator = Evaluator() 
state = env.get_spawner_state()

episode = 1
episode_reward = 0
episode_loss = []

running = True
clock = pygame.time.Clock()

print(f"--- Episode {episode} Start ---")

while running:
    clock.tick(30)

    # Human controls basket
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        env.basket_x -= 3
    if keys[pygame.K_RIGHT]:
        env.basket_x += 3

    env.basket_x = max(0, min(env.grid_width - 1, env.basket_x))

    # Quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # RL chooses when object does not exist
    if env.object_type is None:
        # Calculate max Q-value for the current state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.q(state_tensor)
            max_q_value = torch.max(q_values).item()

        action = agent.act(state)
        # Log the max Q-value for this decision step
        evaluator.log_step(None, agent.epsilon, max_q=max_q_value)

        env.rl_spawn(action)
        episode_reward = 0 # Reset episode reward for the new drop

    # Step environment
    reward, done = env.step(0)  # 0 = no horizontal movement (human already moves)
    episode_reward += reward

    next_state = env.get_spawner_state()

    agent.remember(state, action, reward, next_state, done)
    loss = agent.train_step()
    if loss is not None:
        episode_loss.append(loss)
    agent.decay()

    state = next_state

    env.render()

    # Log step metrics for evaluator
    if loss is not None:
        evaluator.log_step(loss, agent.epsilon) # Q-value is logged separately

    if done:
        agent.update_target()
        agent.save()

        # Log episode metrics for evaluator
        evaluator.log_episode(total_reward=episode_reward, final_score=env.score, lives_left=env.lives)
        evaluator.plot()

        # Print episode summary
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        print(f"\nEpisode {episode}: Score={env.score}, Lives={env.lives}, Total Reward={episode_reward:.2f}, Avg Loss={avg_loss:.4f}, Epsilon={agent.epsilon:.4f}")
        print(f"--- Episode {episode} End ---")

        time.sleep(1)
        env.reset()
        state = env.get_spawner_state()
        episode += 1
        episode_reward = 0
        episode_loss = []
        print(f"\n--- Episode {episode} Start ---")

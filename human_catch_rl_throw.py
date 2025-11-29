# human_catch_rl_throw.py (FINAL CLEANED)

import pygame
from env import FruitCatcherEnv
from dqn_spawner import SpawnerAgent
import time

env = FruitCatcherEnv()
agent = SpawnerAgent()

state = env.get_spawner_state()
running = True
clock = pygame.time.Clock()

while running:
    clock.tick(30)

    # Human basket movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:  env.basket_x -= 3
    if keys[pygame.K_RIGHT]: env.basket_x += 3

    env.basket_x = max(0, min(env.grid_width - 1, env.basket_x))

    # Quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # RL Spawner chooses object
    if env.object_type is None:
        action = agent.act(state)
        env.rl_spawn(action)

    # Step environment
    reward, done = env.step(0)
    next_state = env.get_spawner_state()

    agent.remember(state, action, reward, next_state, done)
    agent.train_step()
    agent.decay()

    state = next_state
    env.render()

    if done:
        print("\n===== GAME OVER =====")
        print("Spawner Episode Finished")

        agent.update_target()
        agent.save()

        time.sleep(1)
        env.reset()
        state = env.get_spawner_state()

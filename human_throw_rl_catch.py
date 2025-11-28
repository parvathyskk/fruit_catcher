# from env_rlcatch import FruitCatcherEnv
# from dqn_agent import DQNAgent
# import pygame
# import time

# env = FruitCatcherEnv()
# agent = DQNAgent()

# running = True
# clock = pygame.time.Clock()

# state = env.reset()

# while running:
#     clock.tick(30)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     keys = pygame.key.get_pressed() 
#     mx, my = pygame.mouse.get_pos()
#     throw_x = mx // env.scale

#     # HUMAN THROWER
#     if env.object_type is None:
#         if keys[pygame.K_1]: env.human_spawn(1, throw_x)
#         if keys[pygame.K_2]: env.human_spawn(2, throw_x)
#         if keys[pygame.K_3]: env.human_spawn(3, throw_x)
#         if keys[pygame.K_4]: env.human_spawn(4, throw_x)
#         if keys[pygame.K_5]: env.human_spawn(5, throw_x)

#     # RL CATCHER
#     action = agent.act(state)  # 0=LEFT,1=STAY,2=RIGHT
#     next_state, reward, done = env.step(action)

#     agent.remember(state, action, reward, next_state, done)
#     agent.train_step()
#     agent.decay()
#     state = next_state

#     env.draw()

#     if done:
#         print("===== GAME OVER =====")
#         print(f"Final Score: {env.score}")
#         running = False  # stop main loop

from env_rlcatch import FruitCatcherEnv
from dqn_agent import DQNAgent
import pygame

env = FruitCatcherEnv()
agent = DQNAgent()

running = True
clock = pygame.time.Clock()

state = env.reset()

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
    agent.train_step()
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
        print("===== GAME OVER =====")
        print(f"Final Score: {env.score}")

        state = env.reset()          # do not quit
        agent.update_target()        # optional but very good for stability
        continue

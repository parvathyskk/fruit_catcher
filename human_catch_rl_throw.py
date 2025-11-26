import pygame
from env import FruitCatcherEnv

env = FruitCatcherEnv()
env.reset()

clock = pygame.time.Clock()

running = True
while running:
    action = 0   # default: stay

    # Human controls
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = -1
    if keys[pygame.K_RIGHT]:
        action = 1

    reward, done = env.step(action)
    env.render()

    if done:
        print("GAME OVER. SCORE:", env.score)
        running = False

    clock.tick(30)

pygame.quit()

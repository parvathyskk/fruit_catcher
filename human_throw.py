import pygame
import time
import numpy as np
from env import FruitCatcherEnv
from dqn_catcher import CatcherAgent # Assuming a new agent for the basket
from evaluator_rl_throw import Evaluator # Re-using the updated evaluator

def run_game():
    """
    Main game loop where a human 'throws' the fruit and an RL agent learns to 'catch' it.
    """
    env = FruitCatcherEnv()
    agent = CatcherAgent() # Agent to control the basket
    evaluator = Evaluator(save_dir="evaluation_results_catcher") # Use a different directory

    state = env.get_catcher_state()

    episode = 1
    episode_reward = 0
    episode_loss = []

    running = True
    clock = pygame.time.Clock()

    print(f"--- Episode {episode} Start ---")

    while running:
        clock.tick(30)

        # --- Human Controls Spawner ---
        # On spacebar press, spawn a fruit at a random location
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and env.object_type is None:
            spawn_action = np.random.randint(0, env.grid_width)
            env.rl_spawn(spawn_action) # Using rl_spawn for consistency
            print(f"\nHuman spawned fruit at column {spawn_action}.")

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- RL Agent Controls Basket ---
        # Calculate and display Q-values for the catcher agent
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q(state_tensor)[0]
            max_q_value = torch.max(q_values).item()
            q_string = ", ".join([f"{q:.2f}" for q in q_values])
            print(f"Catcher Q-values (L, S, R): [{q_string}]", end='\r')

        # Agent chooses an action (0: left, 1: stay, 2: right)
        action = agent.act(state)
        evaluator.log_step(None, agent.epsilon, max_q=max_q_value)

        # Map agent action to environment movement
        movement = 0
        if action == 0: # Move Left
            movement = -3
        elif action == 2: # Move Right
            movement = 3
        env.basket_x += movement
        env.basket_x = max(0, min(env.grid_width - 1, env.basket_x))

        # --- Environment Step and Learning ---
        reward, done = env.step(0) # Pass 0 as human movement is disabled
        episode_reward += reward

        next_state = env.get_catcher_state()

        agent.remember(state, action, reward, next_state, done)
        loss = agent.train_step()
        if loss is not None:
            episode_loss.append(loss)
        agent.decay()

        state = next_state
        env.render()

        # Log step metrics
        evaluator.log_step(loss if loss is not None else 0, agent.epsilon)

        if done:
            print("\n" + "="*20 + " GAME OVER " + "="*20)
            agent.update_target()
            agent.save()
            evaluator.log_episode(episode_reward, env.score, env.lives)
            avg_loss = np.mean(episode_loss) if episode_loss else 0.0
            print(f"Episode {episode}: Score={env.score}, Lives={env.lives}, Total Reward={episode_reward:.2f}, Avg Loss={avg_loss:.4f}, Epsilon={agent.epsilon:.4f}")
            print(f"--- Episode {episode} End ---")

            time.sleep(1)
            env.reset()
            state = env.get_catcher_state()
            episode += 1
            episode_reward = 0
            episode_loss = []
            print(f"\n--- Episode {episode} Start ---")

if __name__ == '__main__':
    run_game()

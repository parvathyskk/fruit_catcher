import pygame
from env import FruitCatcherEnv
from dqn_spawner import SpawnerAgent
from evaluator_rl_throw import Evaluator
import time
import numpy as np

def main():
    # Initialize Environment, Agent, and Evaluator
    env = FruitCatcherEnv()
    agent = SpawnerAgent()
    evaluator = Evaluator(save_dir="evaluation_results")
    
    # Try to load existing model
    try:
        agent.load()
        print("Loaded existing model.")
    except:
        print("No existing model found, starting fresh.")

    # Training Parameters
    TOTAL_EPISODES = 100
    current_episode = 0
    
    clock = pygame.time.Clock()
    running = True

    # RL State Management
    current_state = env.get_spawner_state()
    current_action = None
    episode_reward = 0
    waiting_for_result = False

    print(f"Starting Automated Training for {TOTAL_EPISODES} episodes...")

    while running and current_episode < TOTAL_EPISODES:
        # Run faster than human speed (e.g., 120 FPS or unlimited)
        # Set to 0 for max speed, or 60/120 to watch it
        clock.tick(120) 

        # ------------------------------------------------
        # 1. BOT CATCHER LOGIC
        # ------------------------------------------------
        # Simple heuristic: Move towards the object
        bot_action = 0
        if env.object_type is not None:
            # If it's a fruit, try to catch it
            if "fruit" in env.object_type:
                if env.basket_x < env.object_x:
                    bot_action = 1  # Move Right (Slower)
                elif env.basket_x > env.object_x:
                    bot_action = -1 # Move Left (Slower)
            
            # If it's a bomb, try to avoid it
            elif env.object_type == "bomb":
                # If bomb is to the right, move left
                if env.object_x > env.basket_x:
                     bot_action = -1
                # If bomb is to the left, move right
                elif env.object_x < env.basket_x:
                     bot_action = 1
                # If bomb is directly above, move anywhere (e.g. left)
                else:
                    bot_action = -1

        # Quit check
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ------------------------------------------------
        # 2. RL Logic (Spawn Decision)
        # ------------------------------------------------
        if env.object_type is None and not waiting_for_result:
            current_state = env.get_spawner_state()
            current_action = agent.act(current_state)
            env.rl_spawn(current_action)
            episode_reward = 0
            waiting_for_result = True

        # ------------------------------------------------
        # 3. Step Environment
        # ------------------------------------------------
        reward, done = env.step(bot_action)
        episode_reward += reward

        # ------------------------------------------------
        # 4. RL Learning (End of Drop)
        # ------------------------------------------------
        if waiting_for_result and env.object_type is None:
            next_state = env.get_spawner_state()
            agent.remember(current_state, current_action, episode_reward, next_state, done)
            
            result = agent.train_step()
            agent.decay()
            
            if result is not None:
                loss, avg_q = result
                evaluator.log_step(loss, agent.epsilon, avg_q)
            else:
                evaluator.log_step(None, agent.epsilon, None)
            waiting_for_result = False
            
            # Optional: Print less frequently
            # if agent.epsilon > agent.eps_min:
            #     print(f"Drop finished. Reward: {episode_reward}, Eps: {agent.epsilon:.3f}")
            print(f"Buffer size: {len(agent.buffer)}")

        # ------------------------------------------------
        # 5. Render (Optional - can comment out for max speed)
        # ------------------------------------------------
        env.render()

        # ------------------------------------------------
        # 6. Episode Handling
        # ------------------------------------------------
        if done:
            current_episode += 1
            print(f"Episode {current_episode}/{TOTAL_EPISODES} Completed. Score: {env.score}, Lives: {env.lives}")
            
            evaluator.log_episode(episode_reward, env.score, env.lives)
            
            agent.update_target()
            agent.save() # Save every episode to be safe

            env.reset()
            
            current_state = env.get_spawner_state()
            current_action = None
            episode_reward = 0
            waiting_for_result = False

    print("\n===== TRAINING COMPLETE =====")
    evaluator.plot()
    pygame.quit()

if __name__ == "__main__":
    main()

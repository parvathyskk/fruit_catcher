import pygame
from env import FruitCatcherEnv
from dqn_spawner import SpawnerAgent
from evaluator_rl_throw import Evaluator
import time 
import numpy as np

TOTAL_EPISODES = 20
LOG_INTERVAL = 10
REPLAY_BUFFER_SIZE = 80000
Q_CONV_WINDOW = 100
Q_CONV_THRESHOLD = 0.001
Q_CONV_PATIENCE = 5

def main():
    print("\nFRUIT CATCHER - THROWER AGENT TRAINING\n")
    
    env = FruitCatcherEnv()
    agent = SpawnerAgent()
    evaluator = Evaluator(save_dir="evaluation_results")
    
    try:
        agent.load()
        print("Loaded existing thrower model.")
    except FileNotFoundError:
        print("Starting fresh.\n")
    
    print(f"Config: Episodes={TOTAL_EPISODES}, Batch Size={agent.batch_size}")
    print(f"Replay Buffer Size: {REPLAY_BUFFER_SIZE}\n")
    
    current_episode = 0
    clock = pygame.time.Clock()
    running = True
    q_conv_counter = 0

    # RL State Management
    current_state = env.get_spawner_state()
    current_action = None
    episode_reward = 0
    waiting_for_result = False

    print(f"--- Episode {current_episode + 1} Start ---")

    while running and current_episode < TOTAL_EPISODES:
        clock.tick(120)

        bot_action = 0
        if env.object_type is not None:
            if "fruit" in env.object_type:
                if env.basket_x < env.object_x:
                    bot_action = 1
                elif env.basket_x > env.object_x:
                    bot_action = -1 # Move Left (Slower)
            
            # If it's a bomb, try to avoid it
            elif env.object_type == "bomb":
                if env.object_x > env.basket_x:
                    bot_action = -1
                elif env.object_x < env.basket_x:
                    bot_action = 1
                else:
                    bot_action = -1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if env.object_type is None and not waiting_for_result:
            current_state = env.get_spawner_state()
            # record max-q for this decision so evaluator can log convergence
            try:
                last_max_q = agent.get_max_q(current_state)
            except Exception:
                last_max_q = None

            current_action = agent.act(current_state)
            env.rl_spawn(current_action)
            episode_reward = 0
            waiting_for_result = True

        reward, done = env.step(bot_action)
        episode_reward += reward

       
        #RL learning
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
            
            
            # if agent.epsilon > agent.eps_min:
            #     print(f"Drop finished. Reward: {episode_reward}, Eps: {agent.epsilon:.3f}")
            print(f"Buffer size: {len(agent.buffer)}")

        # ------------------------------------------------
        # 5. Render (Optional - can comment out for max speed)
        # ------------------------------------------------
        env.render()
        if done:
            current_episode += 1
            evaluator.log_episode(episode_reward, env.score, env.lives)
            
            avg_loss = np.mean([l for l in evaluator.losses if l is not None]) if evaluator.losses else 0.0
            
            status = "PASS" if env.score > 0 else "FAIL"
            print(f"{status} Episode {current_episode:3d}/{TOTAL_EPISODES} | "
                  f"Score: {env.score:4d} | Lives: {env.lives:2d} | "
                  f"Reward: {episode_reward:6.2f} | Loss: {avg_loss:.6f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
            
            if current_episode % LOG_INTERVAL == 0:
                evaluator.plot()
            
            agent.update_target()
            agent.save()

            env.reset()
            current_state = env.get_spawner_state()
            current_action = None
            episode_reward = 0
            waiting_for_result = False
            if current_episode < TOTAL_EPISODES:
                print(f"\n--- Episode {current_episode + 1} Start ---")

    print("\n===== TRAINING COMPLETE =====")
    evaluator.plot()
    pygame.quit()


if __name__ == "__main__":
    main()

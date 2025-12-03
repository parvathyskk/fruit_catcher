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
<<<<<<< HEAD
episode_q = []
=======
episode_reward = 0.0
>>>>>>> b7d0811 (graph changes ppp)
state = env.reset()

print(f"--- Episode {episode} Start ---")

while running and episode <= TOTAL_EPISODES:
    clock.tick(100)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

<<<<<<< HEAD
    
    # AUTO THROWER (Simulates Human)
   
=======
>>>>>>> b7d0811 (graph changes ppp)
    if env.object_type is None:
        obj_id = random.randint(1, 5)
        throw_x = random.randint(0, env.grid_width - 1)
        env.human_spawn(obj_id, throw_x)

<<<<<<< HEAD

    # RL AGENT CONTROL
    
=======
    # capture max-Q for current state to log Q-value convergence
    try:
        max_q = agent.get_max_q(state)
    except Exception:
        max_q = None

>>>>>>> b7d0811 (graph changes ppp)
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    reward = reward / 10.0
    episode_reward += reward

    agent.remember(state, action, reward, next_state, done)
<<<<<<< HEAD
    
    # Train and track loss
    result = agent.train_step()
    if result is not None:
        loss, avg_q = result
        episode_loss.append(loss)
        episode_q.append(avg_q)
        
    # Decay epsilon (exploration rate)
=======

    res = agent.train_step()
    loss_val = None
    q_delta = None
    if res is not None:
        if isinstance(res, tuple) or (hasattr(res, '__len__') and len(res) == 2):
            loss_val, q_delta = res
        else:
            loss_val = res

    if loss_val is not None:
        episode_loss.append(loss_val)

    evaluator.log_step(loss_val, agent.epsilon, max_q=max_q, q_delta=q_delta)

    # Q-value convergence check: compute average change every `Q_CONV_WINDOW` updates
    if q_delta is not None and len(evaluator.step_q_deltas) >= Q_CONV_WINDOW:
        if len(evaluator.step_q_deltas) % Q_CONV_WINDOW == 0:
            recent_avg = float(np.mean(evaluator.step_q_deltas[-Q_CONV_WINDOW:]))
            if recent_avg < Q_CONV_THRESHOLD:
                q_conv_counter += 1
            else:
                q_conv_counter = 0

            if q_conv_counter >= Q_CONV_PATIENCE:
                print(f"Q-value convergence detected: avg change={recent_avg:.6e} (< {Q_CONV_THRESHOLD}); stopping training.")
                running = False
                break

>>>>>>> b7d0811 (graph changes ppp)
    agent.decay()

    state = next_state

<<<<<<< HEAD
   #render___may
=======
>>>>>>> b7d0811 (graph changes ppp)
    env.draw()

    if done:
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
<<<<<<< HEAD
        avg_q_val = np.mean(episode_q) if episode_q else 0.0
        
        # Log metrics
=======

>>>>>>> b7d0811 (graph changes ppp)
        evaluator.log_episode(
            episode=episode,
            score=env.score,
            lives=env.lives,
            avg_loss=avg_loss,
            epsilon=agent.epsilon,
<<<<<<< HEAD
            avg_q=avg_q_val
=======
            total_reward=episode_reward,
>>>>>>> b7d0811 (graph changes ppp)
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
<<<<<<< HEAD
        episode_q = []
=======
        episode_reward = 0.0
        if episode <= TOTAL_EPISODES:
            print(f"\n--- Episode {episode} Start ---")
>>>>>>> b7d0811 (graph changes ppp)

print("\nTRAINING COMPLETED\n")
evaluator.plot()
pygame.quit()

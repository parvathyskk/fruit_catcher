import pygame
import random
import numpy as np

class FruitCatcherEnv:
    def __init__(self, grid_width=100, grid_height=100, scale=6):
        pygame.init()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.scale = scale

        self.width = grid_width * scale
        self.height = grid_height * scale

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Fruit Catcher Game (RL Spawner Compatible)")

        # ================= LOAD IMAGES =================
        self.fruit_imgs = [
            pygame.image.load("assets/apple.png"),
            pygame.image.load("assets/orange.png"),
            pygame.image.load("assets/pear.png"),
            pygame.image.load("assets/grape.png"),
        ]
        self.bomb_img = pygame.image.load("assets/bomb.png")
        self.basket_img = pygame.image.load("assets/baskettt.png")

        # Basket size in grid cells
        self.basket_width_cells = self.basket_img.get_width() // self.scale
        self.basket_height_cells = self.basket_img.get_height() // self.scale

        # Basket initial position
        self.basket_x = (self.grid_width - self.basket_width_cells) // 2
        self.basket_y = self.grid_height - self.basket_height_cells - 1

        # Falling object info
        self.object_x = None
        self.object_y = None
        self.object_type = None
        self.object_img = None

        self.score = 0
        self.lives = 3

        self.font = pygame.font.SysFont("Arial", 24)

    # ==========================================================
    def reset(self):
        self.score = 0
        self.lives = 3

        # No initial object for RL spawner
        self.object_x = None
        self.object_y = None
        self.object_type = None
        self.object_img = None

        return self.get_spawner_state()

    # ==========================================================
    # STATE FOR RL SPAWNER
    # ==========================================================
    def get_spawner_state(self):
        return np.array([
            self.basket_x / self.grid_width,           # catcher position
            (self.object_x or 0) / self.grid_width,    # last object x
            (self.object_y or 0) / self.grid_height,   # last object y
            0 if self.object_type is None else (1 if self.object_type != "bomb" else 2) / 2
        ], dtype=np.float32)

    # ==========================================================
    # RL SPAWNER ACTION → (fruit type + X position)
    # ==========================================================
    def rl_spawn(self, action):
        fruit_id = action // 100   # 0..4  (0–3 fruits, 4=bomb)
        x_pos = action % 100       # 0..99

        self.object_x = x_pos
        self.object_y = 0

        if fruit_id == 4:
            self.object_type = "bomb"
            self.object_img = self.bomb_img
        else:
            self.object_type = f"fruit_{fruit_id + 1}"
            self.object_img = self.fruit_imgs[fruit_id]

    # ==========================================================
    # RANDOM SPAWNER (used only for human games)
    # ==========================================================
    def spawn_object(self):
        self.object_x = random.randint(0, self.grid_width - 1)
        self.object_y = 0

        choices = ["fruit_1", "fruit_2", "fruit_3", "fruit_4", "bomb"]
        self.object_type = random.choice(choices)

        if self.object_type == "bomb":
            self.object_img = self.bomb_img
        else:
            idx = int(self.object_type[-1]) - 1
            self.object_img = self.fruit_imgs[idx]

    # ==========================================================
    def step(self, action):
        """
        Action is basket movement from human or RL catcher:
        -1 = left
         0 = stay
        +1 = right
        """
        reward = 0
        done = False

        # Move catcher
        self.basket_x += action
        self.basket_x = max(0, min(self.grid_width - 1, self.basket_x))

        # If no object yet → nothing else to do
        if self.object_type is None:
            return reward, False

        # Move falling object
        self.object_y += 1

        CATCH_RANGE = self.basket_width_cells // 2

        # When object reaches the basket line
        if self.object_y >= self.basket_y:
            caught = (abs(self.object_x - self.basket_x) <= CATCH_RANGE)

            # FRUITS
            if "fruit" in str(self.object_type):
                fruit_values = {"fruit_1": 10, "fruit_2": 20, "fruit_3": 30, "fruit_4": 40}

                if caught:
                    reward += fruit_values[self.object_type]
                    self.score += fruit_values[self.object_type]
                else:
                    reward -= 10

            # BOMB
            elif self.object_type == "bomb":
                if caught:
                    reward -= 50
                    self.lives -= 1
                else:
                    reward += 5
                    self.score += 5

            # Check game over
            if self.lives <= 0 or self.score >= 200:
                done = True

            # Remove object (new one will be spawned externally)
            self.object_type = None
            self.object_img = None

        return reward, done

    # ==========================================================
    def render(self):
        self.screen.fill((50, 50, 50))

        # Draw object
        if self.object_img:
            self.screen.blit(
                self.object_img,
                (self.object_x * self.scale, self.object_y * self.scale)
            )

        # Draw basket
        self.screen.blit(
            self.basket_img,
            (self.basket_x * self.scale, self.basket_y * self.scale)
        )

        # Score & lives
        self.screen.blit(self.font.render(f"Score: {self.score}", True, (255,255,255)), (10,10))
        self.screen.blit(self.font.render(f"Lives: {self.lives}", True, (255,50,50)), (10,40))

        pygame.display.flip()

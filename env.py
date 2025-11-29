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
        pygame.display.set_caption("Fruit Catcher Game (RL Spawner Training)")

        # Load images
        self.fruit_imgs = [
            pygame.image.load("assets/apple.png"),
            pygame.image.load("assets/orange.png"),
            pygame.image.load("assets/pear.png"),
            pygame.image.load("assets/grape.png"),
        ]
        self.bomb_img = pygame.image.load("assets/bomb.png")
        self.basket_img = pygame.image.load("assets/baskettt.png")

        # Basket dimensions in grid cells
        self.basket_width_cells = self.basket_img.get_width() // self.scale
        self.basket_height_cells = self.basket_img.get_height() // self.scale

        # Basket position
        self.basket_x = (self.grid_width - self.basket_width_cells) // 2
        self.basket_y = self.grid_height - self.basket_height_cells - 1

        HALF = self.basket_width_cells // 2
        self.min_basket_x = HALF
        self.max_basket_x = self.grid_width - HALF - 1

        # Object data
        self.object_x = None
        self.object_y = None
        self.object_type = None
        self.object_img = None

        # Viewer scoring
        self.score = 0
        self.lives = 3

        self.font = pygame.font.SysFont("Arial", 24)

    # =======================================================
    def reset(self):
        self.object_x = None
        self.object_y = None
        self.object_type = None
        self.object_img = None

        self.score = 0
        self.lives = 3

        return self.get_spawner_state()

    # =======================================================
    def get_spawner_state(self):
        return np.array([
            self.basket_x / self.grid_width,
            (self.object_x or 0) / self.grid_width,
            (self.object_y or 0) / self.grid_height,
            0 if self.object_type is None else (1 if self.object_type != "bomb" else 2) / 2
        ], dtype=np.float32)

    # =======================================================
    def rl_spawn(self, action):
        fruit_id = action // 100
        x_pos = action % 100

        self.object_x = x_pos
        self.object_y = 0

        if fruit_id == 4:
            self.object_type = "bomb"
            self.object_img = self.bomb_img
        else:
            self.object_type = f"fruit_{fruit_id + 1}"
            self.object_img = self.fruit_imgs[fruit_id]

    # =======================================================
    def step(self, basket_action):
        reward_spawner = 0
        done = False

        # Move basket with correct boundaries
        self.basket_x += basket_action

        # Basket dimensions based on LEFT CORNER positioning
        self.basket_x = max(0, min(self.grid_width - self.basket_width_cells, self.basket_x))


        if self.object_type is None:
            return reward_spawner, done

        # Move object
        self.object_y += 1
        CATCH_RANGE = self.basket_width_cells // 2

        if self.object_y >= self.basket_y:
            caught = abs(self.object_x - self.basket_x) <= CATCH_RANGE

            # ------ Spawner reward ------
            fruit_values = {
                "fruit_1": 10,
                "fruit_2": 20,
                "fruit_3": 30,
                "fruit_4": 40
            }

            if "fruit" in self.object_type:
                if caught:
                    reward_spawner = -5
                else:
                    reward_spawner = fruit_values[self.object_type]
            else:
                if caught:
                    reward_spawner = +50
                else:
                    reward_spawner = -5

            # ------ Viewer Gameplay ------
            if "fruit" in self.object_type:
                if caught:
                    self.score += fruit_values[self.object_type]
            else:
                if caught:
                    self.lives -= 1

            # ------ Terminate ------
            if self.lives <= 0 or self.score >= 200:
                done = True

            # Remove object
            self.object_type = None
            self.object_img = None

        return reward_spawner, done

    # =======================================================
    def render(self):
        self.screen.fill((40, 40, 40))

        # Object
        if self.object_img:
            self.screen.blit(
                self.object_img,
                (self.object_x * self.scale, self.object_y * self.scale)
            )

        # Basket
        self.screen.blit(
            self.basket_img,
            (self.basket_x * self.scale, self.basket_y * self.scale)
        )

        # UI
        self.screen.blit(self.font.render(f"Score: {self.score}", True, (255,255,255)), (10,10))
        self.screen.blit(self.font.render(f"Lives: {self.lives}", True, (255,50,50)), (10,40))

        pygame.display.flip()

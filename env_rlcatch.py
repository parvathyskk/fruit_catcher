import pygame
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
        pygame.display.set_caption("RL Catcher vs Human Spawner")

        # Load images
        self.fruit_imgs = [
            pygame.image.load("assets/apple.png"),
            pygame.image.load("assets/orange.png"),
            pygame.image.load("assets/pear.png"),
            pygame.image.load("assets/grape.png"),
        ]
        self.bomb_img = pygame.image.load("assets/bomb.png")
        self.basket_img = pygame.image.load("assets/baskettt.png")

        self.basket_w = self.basket_img.get_width() // self.scale
        self.basket_h = self.basket_img.get_height() // self.scale

        # Basket position
        self.basket_x = grid_width // 2
        self.basket_y = grid_height - self.basket_h - 1

        # Game parameters
        self.reset()

    # ------------------------------------------------------
    def reset(self):
        self.score = 0
        self.lives = 3
        self.bombs_caught = 0

        self.object_x = None
        self.object_y = None
        self.object_type = None
        self.throw_x = 0
        return self.get_state()

    # ------------------------------------------------------
    def get_state(self):
        return np.array([
            self.basket_x / self.grid_width,
            (self.object_x or 0) / self.grid_width,
            (self.object_y or 0) / self.grid_height,
            0 if self.object_type is None else (1 if self.object_type == "fruit" else 2) / 2,
            self.throw_x / self.grid_width,
            self.lives / 3
        ], dtype=np.float32)

    # ------------------------------------------------------
    def human_spawn(self, obj_id, x_pos):
        self.throw_x = max(0, min(self.grid_width - 1, x_pos))
        self.object_x = self.throw_x
        self.object_y = 0

        if obj_id == 5:
            self.object_type = "bomb"
        else:
            self.object_type = "fruit"
            self.fruit_index = obj_id - 1

    # ------------------------------------------------------
    def check_catch(self):
        return self.basket_x <= self.object_x <= (self.basket_x + self.basket_w)

    # ------------------------------------------------------
    def step(self, action):
        # Basket movement
        speed = 3
        if action == 0:
            self.basket_x = max(0, self.basket_x - speed)
        elif action == 2:
            self.basket_x = min(self.grid_width - self.basket_w - 1, self.basket_x + speed)

        # No object → wait
        if self.object_type is None:
            return self.get_state(), 0, False

        # Move object
        self.object_y += 1

        reward = 0
        done = False

        # Object reaches basket
        if self.object_y >= self.basket_y:
            caught = self.check_catch()

            if self.object_type == "fruit":
                fruit_rewards = [10, 20, 30, 40]
                if caught:
                    reward += fruit_rewards[self.fruit_index]
                    self.score += fruit_rewards[self.fruit_index]
                else:
                    reward -= 5

            else:  # bomb
                if caught:
                    reward -= 50
                    self.lives -= 1
                else:
                    reward += 5

            # Check for termination
            if self.lives <= 0 or self.score >= 200:
                done = True

            # Reset object
            self.object_type = None

        return self.get_state(), reward, done

    # ------------------------------------------------------
    def draw(self):
        self.screen.fill((30, 30, 30))

        font = pygame.font.SysFont("Arial", 24)
        self.screen.blit(font.render(f"Score: {self.score}", True, (255, 255, 255)), (10, 10))
        self.screen.blit(font.render(f"Lives: {self.lives}", True, (255, 100, 100)), (10, 40))

        # Object
        if self.object_type is not None:
            img = self.bomb_img if self.object_type == "bomb" else self.fruit_imgs[self.fruit_index]
            self.screen.blit(img, (self.object_x * self.scale, self.object_y * self.scale))

        # Basket
        self.screen.blit(self.basket_img, (self.basket_x * self.scale, self.basket_y * self.scale))

        pygame.display.flip()

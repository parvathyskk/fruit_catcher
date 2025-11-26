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
        pygame.display.set_caption("Fruit Catcher RL + Human Throw")

        # Load images
        self.fruit_imgs = [
            pygame.image.load("assets/apple.png"),   # +10
            pygame.image.load("assets/orange.png"),  # +20
            pygame.image.load("assets/pear.png"),    # +30
            pygame.image.load("assets/grape.png"),   # +40
        ]
        self.bomb_img = pygame.image.load("assets/bomb.png")
        self.basket_img = pygame.image.load("assets/baskettt.png")
        self.basket_w = self.basket_img.get_width() // self.scale
        self.basket_h = self.basket_img.get_height() // self.scale

        # Basket bottom position
        self.basket_x = grid_width // 2
        self.basket_y = grid_height - self.basket_h - 1

        # Object info
        self.object_x = None
        self.object_y = None
        self.object_img = None
        self.object_type = None  # "fruit" or "bomb"
        self.fruit_index = None
        self.throw_x = 0

        # Game info
        self.score = 0
        self.lives = 3
        self.bombs_caught = 0  # NEW

    # --------------------------
    def reset(self):
        self.score = 0
        self.lives = 3
        self.object_type = None
        self.throw_x = 0
        return self.get_state()

    # --------------------------
    def get_state(self):
        return np.array([
            self.basket_x / self.grid_width,
            (self.object_x if self.object_x is not None else 0) / self.grid_width,
            (self.object_y if self.object_y is not None else 0) / self.grid_height,
            0 if self.object_type is None else (1 if self.object_type=="fruit" else 2)/2,
            self.throw_x / self.grid_width,
            self.lives / 3
        ], dtype=np.float32)

    # --------------------------
    def human_spawn(self, obj_id, x_pos):
        """Human throws fruit/bomb"""
        self.throw_x = max(0, min(self.grid_width - 1, x_pos))
        self.object_x = self.throw_x
        self.object_y = 0

        if obj_id == 5:
            self.object_type = "bomb"
            self.object_img = self.bomb_img
            self.fruit_index = None
        else:
            self.object_type = "fruit"
            self.fruit_index = obj_id - 1
            self.object_img = self.fruit_imgs[self.fruit_index]

    # --------------------------
    def check_catch(self):
        """Returns True if basket catches object"""
        basket_left = self.basket_x
        basket_right = self.basket_x + self.basket_w
        return basket_left <= self.object_x <= basket_right

    # --------------------------
    def step(self, action):
        """
        action:
        0 = LEFT, 1 = STAY, 2 = RIGHT
        """

        # Move basket inside screen
        """The object can move faster than the basket (object_y += 1 per frame)
           agent cannot catch unless it predicts several frames in advance"""
        speed = 3
        if action == 0:
            self.basket_x = max(0, self.basket_x - speed)
        elif action == 2:
            self.basket_x = min(self.grid_width - self.basket_w - 1, self.basket_x + speed)


        # No object → human must throw
        if self.object_type is None:
            return self.get_state(), 0, False

        # Move object
        self.object_y += 1

        reward = 0
        done = False

        if self.object_y >= self.basket_y:
            caught = self.check_catch()

            if self.object_type == "fruit":
                fruit_rewards = [10, 20, 30, 40]
                if caught:
                    reward += fruit_rewards[self.fruit_index]
                    self.score += fruit_rewards[self.fruit_index]
                # Missing fruit → no life lost
                else:
                    reward -= 5
            else:  # bomb
                if caught:
                    reward -= 50
                    self.lives -= 1
                else:
                    reward += 5
                    self.score += 5

            # Check game over conditions
            if self.lives <= 0 or self.bombs_caught >= 3 or self.score >= 200:
                done = True

            self.object_type = None  # remove old object

        return self.get_state(), reward, done

    # --------------------------
    def draw(self):
        self.screen.fill((50, 50, 50))

        # Draw score/lives/bombs
        font = pygame.font.SysFont("Arial", 24)
        score_surf = font.render(f"Score: {self.score}", True, (255, 255, 255))
        life_surf = font.render(f"Lives: {self.lives}", True, (255, 50, 50))
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(life_surf, (10, 40))

        # Draw object
        if self.object_type is not None:
            self.screen.blit(
                self.object_img,
                (self.object_x * self.scale, self.object_y * self.scale)
            )

        # Draw basket
        self.screen.blit(
            self.basket_img,
            (self.basket_x * self.scale, self.basket_y * self.scale)
        )

        pygame.display.flip()

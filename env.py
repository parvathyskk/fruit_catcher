import pygame
import random

class FruitCatcherEnv:
    def __init__(self, grid_width=100, grid_height=100, scale=6):
        pygame.init()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.scale = scale

        self.width = grid_width * scale
        self.height = grid_height * scale

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Fruit Catcher Game")

        # ================= LOAD IMAGES FIRST =================
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

        # Basket initial position (perfectly inside screen)
        self.basket_x = (self.grid_width - self.basket_width_cells) // 2
        self.basket_y = self.grid_height - self.basket_height_cells - 1

        # Object info
        self.object_x = None
        self.object_y = None
        self.object_type = None
        self.object_img = None

        self.score = 0
        self.lives = 3

        # Font for score
        self.font = pygame.font.SysFont("Arial", 24)

    # ==========================================================
    def reset(self):
        self.score = 0
        self.lives = 3
        self.spawn_object()

    # ==========================================================
    def spawn_object(self):
        
        self.object_x = random.randint(0, self.grid_width - 1)
        self.object_y = 0

        object_list = ["fruit_1", "fruit_2", "fruit_3", "fruit_4", "bomb"]
        self.object_type = random.choice(object_list)

        # Assign images
        if self.object_type == "bomb":
            self.object_img = self.bomb_img
        else:
            index = int(self.object_type.split("_")[1]) - 1
            self.object_img = self.fruit_imgs[index]



    # ==========================================================
    def step(self, action):
        """Action: -1 = left, 0 = stay, 1 = right"""

        reward = 0
        done = False

        # Move basket
        self.basket_x += action

        HALF_BASKET = 7     # 90px basket width → 15 grid cells → half = 7

        # Clamp basket inside screen using full width
        self.basket_x = max(HALF_BASKET, min(self.grid_width - HALF_BASKET - 1, self.basket_x))

        # ---- Move falling object ----
        self.object_y += 1

        # Basket width is 90px → 90/6 = 15 grid cells
        # Catch range = ±7
        CATCH_RANGE = 7
        

        # ---- Check object reached basket level ----
        if self.object_y >= self.basket_y - 1:

            caught = abs(self.object_x - self.basket_x) <= CATCH_RANGE

            # ------------------------
            # CASE 1: OBJECT CAUGHT
            # ------------------------
            if caught:

                if self.object_type == "fruit_1":
                    reward = 10
                    self.score += 10

                elif self.object_type == "fruit_2":
                    reward = 20
                    self.score += 20

                elif self.object_type == "fruit_3":
                    reward = 30
                    self.score += 30

                elif self.object_type == "fruit_4":
                    reward = 40
                    self.score += 40

                elif self.object_type == "bomb":
                    reward = -50
                    self.lives -= 1

            # ------------------------
            # CASE 2: OBJECT MISSED
            # ------------------------
            else:

                # Missed bomb → bonus
                if self.object_type == "bomb":
                    reward = +5
                    self.score += 5

                # Missed fruit → lose life
                else:
                    reward = -10

            # Spawn next object after handling
            self.spawn_object()

        # ------------------------
        # END CONDITIONS
        # ------------------------
        if self.lives <= 0:
            done = True

        if self.score >= 200:
            self.score = 200
            done = True

        return reward, done

    # ==========================================================
    def render(self):
        self.screen.fill((50, 50, 50))

        # Draw falling object
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

        # ======== Draw Score & Lives ========
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255))
        lives_text = self.font.render(f"Lives: {self.lives}", True, (255,100,100))

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 40))

        pygame.display.flip()




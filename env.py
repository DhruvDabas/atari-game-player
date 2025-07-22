import pygame
import random
import numpy as np

class AtariEnv:
    def __init__(self):

        pygame.init()
        self.WIDTH, self.HEIGHT = 680, 580
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 85, 15
        self.BALL_SIZE = 15
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 60, 20

        self.paddle_speed = 7
        self.ball_speed = 4

        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,165,0)]

        self.display = False  # turn off when training, set True to render

        if self.display:
            self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Atari Env")

        self.clock = pygame.time.Clock()
        self.FPS = 90

        self.reset()

    def create_blocks(self):
        blocks = []
        for row in range(5):
            for col in range(self.WIDTH // (self.BLOCK_WIDTH + 10)):
                x = col * (self.BLOCK_WIDTH + 10) + 35
                y = row * (self.BLOCK_HEIGHT + 5) + 30
                rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                blocks.append((rect, self.colors[row % len(self.colors)]))
        return blocks

    def reset(self):
        self.paddle = pygame.Rect(
            random.randint(0, self.WIDTH - self.PADDLE_WIDTH),
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball = pygame.Rect(
            random.randint(50, self.WIDTH - 50 - self.BALL_SIZE),
            random.randint(self.HEIGHT // 3, self.HEIGHT // 2),
            self.BALL_SIZE,
            self.BALL_SIZE
        )
        self.ball_vel = [random.choice([-4, -3, 3, 4]), -4]
        self.blocks = self.create_blocks()
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array([
            self.paddle.x / self.WIDTH,
            self.ball.x / self.WIDTH,
            self.ball.y / self.HEIGHT,
            self.ball_vel[0] / self.ball_speed,
            self.ball_vel[1] / self.ball_speed
        ], dtype=np.float32)

    def step(self, action):
        if action == 1 and self.paddle.left > 0:
            self.paddle.x -= self.paddle_speed
        elif action == 2 and self.paddle.right < self.WIDTH:
            self.paddle.x += self.paddle_speed

        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        reward = 0

        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1

        if self.ball.colliderect(self.paddle):
            self.ball_vel[1] *= -1
            self.ball.y = self.paddle.top - self.BALL_SIZE
            reward = 0.2

        hit_index = self.ball.collidelist([block[0] for block in self.blocks])
        if hit_index != -1:
            block_rect, _ = self.blocks.pop(hit_index)
            if abs(self.ball.bottom - block_rect.top) < 10 or abs(self.ball.top - block_rect.bottom) < 10:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
            reward = 1.0  # reward for breaking block

        if self.ball.bottom >= self.HEIGHT:
            self.done = True
            reward = -1.0  # penalty for losing

        if not self.blocks:
            self.done = True
            reward = 10.0  # bonus for winning

        return self.get_state(), reward, self.done, {}

    def render(self):
        if not self.display:
            return

        self.clock.tick(self.FPS)
        self.WIN.fill((30, 30, 30))
        pygame.draw.rect(self.WIN, (255, 255, 255), self.paddle)
        pygame.draw.ellipse(self.WIN, (200, 200, 200), self.ball)
        for block, color in self.blocks:
            pygame.draw.rect(self.WIN, color, block)
        pygame.display.flip()

    def close(self):
        pygame.quit()

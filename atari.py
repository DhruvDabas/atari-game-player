import pygame
import random
import sys

pygame.init()
WIDTH, HEIGHT = 680, 580
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("あたり")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 90

PADDLE_WIDTH, PADDLE_HEIGHT = 85, 15
BALL_SIZE = 15
BLOCK_WIDTH, BLOCK_HEIGHT = 60, 20

paddle = pygame.Rect(WIDTH//2 - PADDLE_WIDTH//2, HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH//2, HEIGHT//2, BALL_SIZE, BALL_SIZE)
ball_vel = [4, -4]
paddle_speed = 7

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,165,0)]

def create_blocks():
    blocks = []
    for row in range(5):
        for col in range(WIDTH // (BLOCK_WIDTH + 10)):
            x = col * (BLOCK_WIDTH + 10) + 35
            y = row * (BLOCK_HEIGHT + 5) + 30
            rect = pygame.Rect(x, y, BLOCK_WIDTH, BLOCK_HEIGHT)
            blocks.append((rect, colors[row % len(colors)]))
    return blocks

blocks = create_blocks()

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 36)

def reset_ball_paddle():
    paddle.x = random.randint(0, WIDTH - PADDLE_WIDTH)
    
    paddle.y = HEIGHT - 40

    ball.x = random.randint(50, WIDTH - 50 - BALL_SIZE)
    ball.y = random.randint(HEIGHT // 3, HEIGHT // 2)

    ball_vel[0] = random.choice([-4, -3, 3, 4])
    ball_vel[1] = -4

reset_ball_paddle()

def show_message(msg):
    text = font.render(msg, True, WHITE)
    rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    WIN.blit(text, rect)
    pygame.display.flip()
    pygame.time.delay(2000)

running = True
while running:
    clock.tick(FPS)
    WIN.fill((30, 30, 30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.x -= paddle_speed
    if keys[pygame.K_RIGHT] and paddle.right < WIDTH:
        paddle.x += paddle_speed

    ball.x += ball_vel[0]
    ball.y += ball_vel[1]

    if ball.left <= 0 or ball.right >= WIDTH:
        ball_vel[0] *= -1
    if ball.top <= 0:
        ball_vel[1] *= -1
    if ball.colliderect(paddle):
        ball_vel[1] *= -1
        ball.y = paddle.top - BALL_SIZE

    hit_index = ball.collidelist([block[0] for block in blocks])
    if hit_index != -1:
        block_rect, _ = blocks.pop(hit_index)
        if abs(ball.bottom - block_rect.top) < 10 or abs(ball.top - block_rect.bottom) < 10:
            ball_vel[1] *= -1
        else:
            ball_vel[0] *= -1

    if ball.bottom >= HEIGHT:
        reset_ball_paddle()
        blocks = create_blocks()

    if not blocks:
        WIN.fill((0, 0, 0))
        show_message("Winner! ")
        reset_ball_paddle()
        blocks = create_blocks()

    pygame.draw.rect(WIN, WHITE, paddle)
    pygame.draw.ellipse(WIN, (200, 200, 200), ball)
    for block, color in blocks:
        pygame.draw.rect(WIN, color, block)

    pygame.display.flip()

pygame.quit()
sys.exit()

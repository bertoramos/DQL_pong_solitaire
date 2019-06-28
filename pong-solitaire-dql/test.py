
import pygame
import gym
import gym_pong_solitaire

env = gym.make('PongSolitaire-v0')




reloj = pygame.time.Clock()

for _ in range(100):
    a = env.reset()
    print(a)
    done = False
    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render(mode='human')
        reloj.tick(60)
env.close()

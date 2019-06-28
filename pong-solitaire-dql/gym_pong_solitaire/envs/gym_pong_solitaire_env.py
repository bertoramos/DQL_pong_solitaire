
import numpy as np
import pygame
import gym
from gym import spaces


class PongSolitaireEnv(gym.Env):

    metadata = {'render.modes': ['human', 'stat']}

    def __init__(self):
        super(PongSolitaireEnv, self).__init__()

        self.width = 300
        self.height = 300

        self.ball_radius = 10
        self.paddle_width = 100
        self.paddle_height = 5

        self.dx_paddle = int(self.width/self.paddle_width)
        self.y_paddle = self.height - self.paddle_height - 10

        self.low = np.array([0, 0, -1, -1, 0])  # x_ball, y_ball, dx, dy, x_paddle
        self.high = np.array([self.width, self.height, 1, 1, self.width-self.paddle_width])

        self.action_space = spaces.Discrete(3)  # 0 = left ; 1 = straight ; 2 = right
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.x_ball = int(self.width / 2)
        self.y_ball = int(self.height / 2)
        self.dx = self.high[2]
        self.dy = self.high[3]
        self.x_paddle = int(self.width / 2)

        self.done = False
        # Graficos
        self.pantalla = None
        self.n = 1

    def _move_ball(self):
        self.x_ball += self.dx
        self.y_ball += self.dy

        bounce_paddle = False
        if self.x_ball - self.ball_radius <= 0:  # Pared izquierda
            self.dx *= -1
        elif self.x_ball + self.ball_radius >= self.width:  # Pared derecha
            self.dx *= -1
        elif self.y_ball - self.ball_radius <= 0:  # Techo
            self.dy *= -1
        elif self.y_ball + self.ball_radius >= self.y_paddle and \
                self.x_paddle -10 <= self.x_ball <= self.x_paddle + 10 + self.paddle_width:  # Choca en raqueta
            self.dy *= -1
            bounce_paddle = True
        return bounce_paddle

    def _move_paddle(self, action):
        if action == 0:  # Left
            self.x_paddle -= self.dx_paddle
            if self.x_paddle < 0:
                self.x_paddle = self.width - self.paddle_width
        elif action == 1:  # Straight
            pass
        elif action == 2:  # Right
            self.x_paddle += self.dx_paddle
            if self.x_paddle + self.paddle_width >= self.width:
                self.x_paddle = 0

    def step(self, action):
        """
        :param action: 0 = left ; 1 = straight ; 2 = right
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._move_paddle(action)
        bounce_paddle = self._move_ball()

        self.done = True if self.y_ball + self.ball_radius > self.y_paddle else False

        reward = 1 if bounce_paddle else 0

        return np.array([self.x_ball, self.y_ball, self.dx, self.dy, self.x_paddle]), reward, self.done, ["x_ball", "y_ball", "dy", "x_paddle"]

    def reset(self):
        # x_ball, y_ball, dx, dy, x_paddle
        self.x_ball = int(self.width/2)
        self.y_ball = int(self.height/2)
        self.dx = self.high[2]
        self.dy = self.high[3]
        self.x_paddle = int(self.width/2)
        self.done = False
        return np.array([self.x_ball, self.y_ball, self.dx, self.dy, self.x_paddle])

    def _draw_state(self):
        NEGRO = (0, 0, 0)
        BLANCO = (255, 255, 255)
        self.pantalla.fill(NEGRO)

        pygame.draw.circle(self.pantalla, BLANCO, (self.x_ball, self.y_ball), self.ball_radius, self.ball_radius)
        pygame.draw.rect(self.pantalla, BLANCO, (self.x_paddle, self.y_paddle, self.paddle_width, self.paddle_height))

        pygame.display.flip()

    def render(self, mode='human', close=False):
        if mode == 'human':
            if self.pantalla is None:
                self.pantalla = pygame.display.set_mode([self.width, self.height])
                pygame.init()
            self._draw_state()
        elif mode == 'stat':
            self.n += 1
            print('Render time : ' + str(self.n) + "\n"
                  '\tx_ball: ' + str(self.x_ball) + "\n" +
                  '\ty_ball: ' + str(self.y_ball) + "\n" +
                  '\tdx: ' + str(self.dx) + "\n" +
                  '\tdy: ' + str(self.dy) + "\n" +
                  '\tx_paddle: ' + str(self.x_paddle))

    def close(self):
        pygame.quit()

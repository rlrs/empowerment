import numpy as np
import random

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Maze(object):
    def __init__(self, layout):
        self.action_space = [UP, RIGHT, DOWN, LEFT]
        self.layout = layout
        self.pos = []

    def step(self, action):
        if action == UP and not self.is_wall(self.pos[0] - 1, self.pos[1]):
            self.pos[0] -= 1
        elif action == RIGHT and not self.is_wall(self.pos[0], self.pos[1] + 1):
            self.pos[1] += 1
        elif action == DOWN and not self.is_wall(self.pos[0] + 1, self.pos[1]):
            self.pos[0] += 1
        elif action == LEFT and not self.is_wall(self.pos[0], self.pos[1] - 1):
            self.pos[1] -= 1

        s = self.get_state()

        return s

    def reset(self, pos):
        self.pos = np.asarray(pos)

    def get_state(self):
        s = self.layout.copy()
        s[self.pos[0], self.pos[1]] = 2
        return s

    def is_wall(self, y, x):
        return self.layout[y, x] == 1

    def free(self):
        res = []
        for pos, val in np.ndenumerate(self.layout):
            if val == 0:
                res.append(pos)
        return res

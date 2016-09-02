"""
Replay memory with uniform minibatch sampling, implementation inspired by Tambet Matiisen's "Simple DQN".

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import numpy as np
import random


class ReplayMemory(object):
    def __init__(self, config):
        self.nsteps = int(config['n_steps'])
        self.nactions = int(config['num_actions'])
        self.max_size = int(config['memory_size'])
        self.batch_size = int(config['batch_size'])
        self.state_size = int(config['state_size'])
        self.state_encoder = config['state_encoder']
        self.states = np.empty([self.max_size, config['state_size']])
        self.actions = np.empty([self.max_size, self.nsteps, self.nactions], dtype=np.uint8)
        self.terminals = np.empty([self.max_size], dtype=np.bool)

        self.count = 0
        self.current = 0

    def get_minibatch(self):
        assert self.count > self.batch_size, "Not enough data in replay memory"

        indices = []
        s = np.empty((self.batch_size, self.state_size))
        ns = np.empty((self.batch_size, self.state_size))
        while len(indices) < self.batch_size:
            while True:
                idx = random.randint(1, self.count - 1)
                # if we wrap over current pointer, try again
                if idx >= self.current > idx - 1:
                    continue
                # if we cross a terminal state, try again (last state can be terminal)
                if self.terminals[idx - 1].any():
                    continue
                break

            s[len(indices), ...] = self.get_state(idx - 1)
            ns[len(indices), ...] = self.get_state(idx)
            indices.append(idx)

        a = self.actions[indices]
        t = self.terminals[indices]

        return s, a, ns, t

    def get_state(self, idx):
        assert self.count > 0, "Replay memory is empty."
        assert idx < self.count, "idx not in range"

        view = self.states[idx, ...]
        return view

    def get_current(self):
        return self.get_state(self.current - 1)

    def add(self, s, a, t):
        # state is after transition
        assert s.shape == (self.state_size,), "State has wrong dimension {}".format(s.shape)

        self.states[self.current, ...] = s
        self.actions[self.current, ...] = a
        self.terminals[self.current] = t

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.max_size


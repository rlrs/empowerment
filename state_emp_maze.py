from Maze import Maze
from Empowerment import Empowerment
from VectorEmpReplayMemory import ReplayMemory
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import random
import copy
import itertools
import timeit
import numpy as np

config = dict(
    num_actions = 4,
    state_encoder = False,
    state_size = 2,
    rnn_size = 128,
    n_steps = 2,
    beta = 1,
    emp_lr = 0.001,
    beta1 = 0.9,
    beta2 = 0.999,
    emp_eps = 1e-8,

    lr = 0.0001,
    opt_decay = 0.95,
    momentum = 0.0,
    opt_eps = 0.01,
    memory_size = 1e5,
    state_frames = 1,
    batch_size = 8,
    device = '/gpu:0',
    train_steps = 5e6,
    tensorboard = False,
)


def calc_emp(emp, env):
    """
    Calculate empowerment values in the maze(, and draw t-sne of state embeddings)
    """
    #tsne = TSNE(verbose=5, learning_rate=300, n_iter=3000, method='exact')
    #s = []
    #es = []
    approx = np.zeros_like(env.layout, dtype=np.float)
    cenv = copy.copy(env)
    for (y, x) in env.free():
        cenv.pos = np.array([y, x])
        s = cenv.pos
        approx[y, x] = emp.state_get(s[np.newaxis, ...])
        #s.append(emp.get_state(o[np.newaxis, ..., np.newaxis]))  # state embedding
        #es.append(e)  # empowerment corresponding to state
    return approx
    #Y = tsne.fit_transform(s)
    #plt.scatter(Y[:, 0], Y[:, 1], c=es, cmap='gray', s=40)
    #plt.show()


def pathcount(env):
    cenv = copy.copy(env)
    res = np.zeros_like(env.layout, dtype=np.float)

    for pos in env.free():
        action_seqs = itertools.product(range(config['num_actions']), repeat=config['n_steps'])  # cartesian product of actions
        states = []
        for a_seq in action_seqs:
            cenv.pos = np.array(pos)
            for a in a_seq:
                cenv.step(a)
            states.append(tuple(cenv.pos))

        # count unique states
        keys = {}
        for s in states:
            keys[s] = 1

        # emp = log(|s|)
        (y, x) = pos
        res[y, x] = np.log(len(keys))
    return res


def error(emp, env, truth):
    """
    Calculate sum squared error between ground truth and empowerment approximation.
    """
    approx = calc_emp(emp, env)
    return ((truth - approx) ** 2).sum()


def plot(a, fname):
    plt.matshow(a, cmap='gist_heat')
    plt.xticks(range(len(a)))
    plt.yticks(range(len(a)))
    minrange = np.min(a[np.nonzero(a)])
    maxrange = np.max(a[np.nonzero(a)])
    plt.colorbar(boundaries=np.linspace(minrange, maxrange, 10),
                 shrink=0.8)#, ticks=np.linspace(minrange, maxrange, 10))
    plt.savefig(fname)
    plt.show()

def main():
    np.set_printoptions(precision=2)
    layout = np.array(np.mat('1 1 1 1 1 1 1 1 1 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 1 0 0 0 0 0 0 1; \
                              1 0 0 0 0 0 0 0 0 1; \
                              1 1 1 1 1 1 1 1 1 1'))
    env = Maze(layout)

    ## Runtime for
    #n_steps = config['n_steps']
    #for i in range(2, 9):
    #    config['n_steps'] = i
    #    _pathcount = lambda: pathcount(env)
    #    print i, timeit.Timer(_pathcount).timeit(5) / 5 * 1000
    #config['n_steps'] = n_steps
    ground_truth = pathcount(env)
    #plot(ground_truth, 'groundtruth_k2.pdf')

    env.reset(random.choice(env.free()))
    emp = Empowerment(config)
    mem = ReplayMemory(config)
    s = env.pos
    mem.add(s, [0], 0)

    # EMPOWERMENT
    for i in range(int(config['train_steps'])):
        # TEMPORARY TEST
        if True:
            env.pos = np.array([1,1])
            s = env.pos
            mem.add(s, [0], 0)

        a_k = emp.state_draw_actions(s[np.newaxis, ...])
        for a_oh in a_k:
            a = np.where(a_oh == 1)[0][0]  # convert onehot -> number
            o = env.step(a)
            s = env.pos

        mem.add(s, a_k, 1)
        if i >= 100:  # put some transitions into memory before training
            (bs, ba, bns, bt) = mem.get_minibatch()
            #print bs, ba, bns
            (L_ksi, L_theta, e, logsoftmus) = emp.state_train(bs, ba, bns)
            #print bs, bns

            if i % 10 == 0:
                err = error(emp, env, ground_truth)
                #print bs, bns, np.exp(logsoftmus)
                print i, err, L_ksi, L_theta
                #print np.exp(emp.state_predictive([[1,1]], [[4,1]]))
                if L_ksi < 1 and L_theta < 1:
                    plot(calc_emp(emp, env), 'mazeapprox.pdf')
                    #print [1,1], np.exp(emp.state_policy([[1,1]]))
                    #print [5,5], np.exp(emp.state_policy([[5,5]]))
                    #print ground_truth
                    #draw_emp(emp, env)
                if i > 30000:
                    plot(calc_emp(emp, env), 'mazeapprox.pdf')
            #print i, L_ksi, L_theta, e
            #if L_ksi < 0.05:
            #    draw_emp(emp, env)
            #    return

if __name__ == '__main__':
    sys.exit(main())
from Maze import Maze
from Empowerment import Empowerment
from EmpReplayMemory import ReplayMemory
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # necessary before importing pyplot
import matplotlib.pyplot as plt
import sys
import random
import copy
import itertools
import timeit
import numpy as np

config = dict(
    in_width = 10,
    in_height = 10,
    num_actions = 4,
    state_encoder = True,
    state_size = 20,
    rnn_size = 20,
    n_steps = 2,
    beta = 1.0,
    emp_lr = 0.0001,
    beta1 = 0.9,
    beta2 = 0.99,
    emp_eps = 1e-8,

    lr = 0.0001,
    opt_decay = 0.95,
    momentum = 0.0,
    opt_eps = 0.01,
    memory_size = 1e5,
    state_frames = 1,
    batch_size = 16,
    device = '/gpu:0',
    train_steps = 5e6,
    test_eps = 5,
    tensorboard = False,
)



def draw_emp(emp, env):
    """
    Print empowerment values in the maze, and draw t-sne of state embeddings.
    """
    l = env.layout.copy().astype(np.float32)
    tsne = TSNE(verbose=5, learning_rate=300, n_iter=3000, method='exact')
    s = []
    es = []
    for (y, x) in env.free():
        o = env.layout.copy()
        o[y, x] = 2
        e = emp.get(o[np.newaxis, ..., np.newaxis])
        l[y, x] = e  # grid of empowerment
        #s.append(emp.get_state(o[np.newaxis, ..., np.newaxis]))  # state embedding
        #es.append(e)  # empowerment corresponding to state
    print l
    #Y = tsne.fit_transform(s)
    #plt.scatter(Y[:, 0], Y[:, 1], c=es, cmap='gray', s=40)
    #plt.show()

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
        cenv.pos = np.array([y,x])
        o = cenv.get_state()
        approx[y, x] = emp.get(o[np.newaxis, ..., np.newaxis])
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
    approx = np.zeros_like(env.layout, dtype=np.float)
    cenv = copy.copy(env)
    for (y, x) in env.free():
        cenv.pos = np.array([y,x])
        o = cenv.get_state()[np.newaxis, ..., np.newaxis]
        approx[y, x] = emp.get(o)
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
    #plt.show()

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

    cenv = copy.copy(env)

    #_pathcount = lambda: pathcount(env)
    #print 'Path count time elapsed (ms): ', timeit.Timer(_pathcount).timeit(5) / 5 * 1000
    ground_truth = pathcount(env)

    #sum = 0  # calculate uniform expected empowerment
    #for (y,x) in env.free():
    #    sum += ground_truth[y,x]
    #expected_emp = sum / len(env.free())
    #print 'Expected emp: ', expected_emp
    #err = 0  # squared loss of uniform expected empowerment
    #for (y,x) in env.free():
    #    err += (ground_truth[y,x] - expected_emp) ** 2
    #print 'Error: ', err

    #plot(ground_truth, 'groundtruth_k2.pdf')

    env.reset(random.choice(env.free()))
    emp = Empowerment(config)
    mem = ReplayMemory(config)
    o = env.get_state()
    mem.add(o, [0], 0)
    minerr = 999999

    # EMPOWERMENT
    for i in range(int(config['train_steps'])):
        a_k = emp.draw_actions(o[np.newaxis, ..., np.newaxis])
        for a_oh in a_k:
            a = np.where(a_oh == 1)[0][0]  # convert onehot -> number
            o = env.step(a)
            s = env.pos
        # TEMPORARY TEST
        #if i % 1 == 0:
        #    mem.add(o, a_k, 1)
        #    env.pos = np.array(random.choice([[5,5], [1,1]]))
        #    o = env.get_state()
        #    mem.add(o, [0], 0)
        #else:
        mem.add(o, a_k, 0)

        if i > 100:  # put some transitions into memory before training
            (bs, ba, bns, bt) = mem.get_minibatch()
            L_ksi, L_theta = emp.train(bs, ba, bns)
            #print ba, arows
            #emp.train(oo[np.newaxis, ..., np.newaxis], np.array(a_k)[np.newaxis, ...], o[np.newaxis, ..., np.newaxis])

            if i % 1000 == 0:
                err = error(emp, env, ground_truth)
                print '#', i, err, L_ksi, L_theta

                print "#[1,1] -> [3,1]"
                cenv.pos = [1,1]
                co = cenv.get_state()[np.newaxis, ..., np.newaxis]
                cenv.pos = [3,1]
                cno = cenv.get_state()[np.newaxis, ..., np.newaxis]
                pred = np.exp(emp.predictive(co, cno))
                pol = np.exp(emp.policy(co))
                print pred
                print pol

                print "#[1,1] -> [2,1]"
                cenv.pos = [1,1]
                co = cenv.get_state()[np.newaxis, ..., np.newaxis]
                cenv.pos = [2,1]
                cno = cenv.get_state()[np.newaxis, ..., np.newaxis]
                pred = np.exp(emp.predictive(co, cno))
                print pred

                print "#[5,5] -> [5,7]"
                cenv.pos = [5, 5]
                co = cenv.get_state()[np.newaxis, ..., np.newaxis]
                cenv.pos = [5, 7]
                cno = cenv.get_state()[np.newaxis, ..., np.newaxis]
                pred = np.exp(emp.predictive(co, cno))
                pol = np.exp(emp.policy(co))
                print pred
                print pol

                print "#[8,8] -> [6,8]"
                cenv.pos = [8, 8]
                co = cenv.get_state()[np.newaxis, ..., np.newaxis]
                cenv.pos = [6, 8]
                cno = cenv.get_state()[np.newaxis, ..., np.newaxis]
                pred = np.exp(emp.predictive(co, cno))
                pol = np.exp(emp.policy(co))
                print pred
                print pol

                print "#[8,8] -> [8,8]"
                cenv.pos = [8, 8]
                co = cenv.get_state()[np.newaxis, ..., np.newaxis]
                cenv.pos = [8, 8]
                cno = cenv.get_state()[np.newaxis, ..., np.newaxis]
                pred = np.exp(emp.predictive(co, cno))
                pol = np.exp(emp.policy(co))
                print pred
                print pol
            #if i > 20000:
            #    plot(calc_emp(emp, env), 'empmaze.pdf')
                #cenv.pos = [1,1]
                #co = cenv.get_state()[np.newaxis, ..., np.newaxis]
                #cenv.pos = [3,1]
                #cno = cenv.get_state()[np.newaxis, ..., np.newaxis]
                #print np.exp(emp.predictive(co, cno))
                #print np.exp(emp.policy(co))
                #if err < 100 and err < minerr:
                #    minerr = err
                #    plot(calc_emp(emp, env), 'empmaze_20_k4.pdf')
            #        print ground_truth
            #        draw_emp(emp, env)
            #print i, L_ksi, L_theta, e
            #if L_ksi < 0.05:
            #    draw_emp(emp, env)
            #    return

if __name__ == '__main__':
    sys.exit(main())

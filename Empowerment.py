import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class Empowerment(object):
    def __init__(self, config):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False,
                                                     inter_op_parallelism_threads=4,
                                                     intra_op_parallelism_threads=4))  # TF uses all cores by default
        with tf.device(config['device']):
            self.beta = tf.constant(config['beta'], dtype=tf.float32)
            self.actions = tf.placeholder('float',
                                          [None, config['n_steps'],
                                           config['num_actions']],
                                          name='actions')

            if not config['state_encoder']:  # feed state directly
                self.before_s = tf.placeholder('float', [None, config['state_size']], name='before_s')
                self.after_s = tf.placeholder('float', [None, config['state_size']], name='after_s')
                self.combined_s = tf.concat(1, [self.before_s, self.after_s])
            else:  # feed observations through CNN state encoder
                self.obs_before = tf.placeholder('float',
                                                 [None, config['in_width'], config['in_height'], config['state_frames']],
                                                 name='obs_before')

                self.obs_after = tf.placeholder('float',
                                                [None, config['in_width'], config['in_height'], config['state_frames']],
                                                name='obs_after')

                conv_in = tf.concat(0, [self.obs_before, self.obs_after])  # combine states to push through CNN

                with tf.variable_scope('state_encoder'):
                    # conv layers
                    with tf.variable_scope('conv1'):
                        shape = [3, 3, config['state_frames'], 8]
                        W1 = self.make_weight(shape)
                        b1 = self.make_bias(8)
                        conv1 = self.conv2d(conv_in, W1, stride=1)
                        conv1 = tf.nn.bias_add(conv1, b1)
                        conv1 = tf.nn.relu(conv1)

                    with tf.variable_scope('conv2'):
                        shape = [3, 3, 8, 8]
                        W2 = self.make_weight(shape)
                        b2 = self.make_bias(8)
                        conv2 = self.conv2d(conv1, W2, stride=1)
                        conv2 = tf.nn.bias_add(conv2, b2)
                        conv2 = tf.nn.relu(conv2)

                    # flatten
                    conv_neurons = 1
                    for d in conv2.get_shape()[1:].as_list():
                        conv_neurons *= d
                    flat = tf.reshape(conv2, [-1, conv_neurons])

                    # fully connected to state representation
                    with tf.variable_scope('state'):
                        shape = [conv_neurons, config['state_size']]
                        W3 = self.make_weight(shape)
                        b3 = self.make_bias(config['state_size'])
                        self.all_s = tf.nn.relu_layer(flat, W3, b3)  # before and after encoded states
                        [self.before_s, self.after_s] = tf.split(0, 2, self.all_s)
                        self.combined_s = tf.concat(1, [self.before_s, self.after_s])

            self.batch_size = tf.shape(self.combined_s)[0]

            # q_ksi, variational distribution
            repeated_s_sp = tf.tile(tf.expand_dims(self.combined_s, 1),
                                    [1, config['n_steps'], 1])
            s_sp_list = tf.split(1, config['n_steps'], repeated_s_sp)
            for i, row in enumerate(s_sp_list):
                s_sp_list[i] = tf.squeeze(row, [1])
            #idx = range(config['n_steps'])
            #shifted_a = tf.gather(tf.transpose(self.actions, [1, 0, 2]),
            #                                   idx[-1:] + idx[:-1]) # shift actions 'forward' one step
            # replace first (well, it was last) action by zeros
            #shifted_a = tf.concat(0, [tf.zeros([1, tf.shape(shifted_a)[1], config['num_actions']]),
            #                          tf.gather(shifted_a, idx[1:])])
            #shifted_a = tf.transpose(shifted_a, [1, 0, 2])
            #a_s_sp = tf.concat(2, [shifted_a, repeated_s_sp])  # construct matrix with rows (a_{k-1}, s, s')
            #self.a_s_sp_list = tf.split(1, config['n_steps'], a_s_sp) # split each row into a tensor
            #for i, row in enumerate(self.a_s_sp_list):
            #    self.a_s_sp_list[i] = tf.squeeze(row, [1])
            self.action_rows = [tf.squeeze(x, squeeze_dims=[1]) for x in tf.split(1, config['n_steps'], self.actions)]

            with tf.variable_scope('q_ksi'):
                cell = rnn_cell.BasicLSTMCell(config['rnn_size'], state_is_tuple=True)
                shape = [config['rnn_size'], config['num_actions']]
                Wmlp = self.make_weight(shape)
                bmlp = self.make_bias(config['num_actions'])

                with tf.variable_scope('lstm') as varscope:
                    # likelihood graph - given actions a_k
                    state = cell.zero_state(self.batch_size, tf.float32)
                    self.logsoftmus = []
                    logmu = []  # categorical means for given actions
                    for i, row in enumerate(s_sp_list):
                        (f_k, state) = cell(row, state)
                        mlp = tf.nn.bias_add(tf.matmul(f_k, Wmlp), bmlp)
                        logsoftmu = tf.nn.log_softmax(mlp)
                        self.logsoftmus.append(logsoftmu)
                        selected_a = tf.reduce_sum(tf.mul(logsoftmu, self.action_rows[i]), reduction_indices=1)
                        logmu.append(selected_a)
                        varscope.reuse_variables()  # share weights created by cell
                self.logq = tf.reduce_sum(logmu, reduction_indices=0)  # sum over steps

            #
            # h_theta
            repeated_s = tf.tile(tf.expand_dims(self.before_s, 1),
                                 [1, config['n_steps'], 1])
            #a_s = tf.concat(2, [shifted_a, repeated_s])  # rows (a_{k-1}, s)
            #self.a_s_list = tf.split(1, config['n_steps'], a_s)  # split into tensors
            s_list = tf.split(1, config['n_steps'], repeated_s)
            for i, row in enumerate(s_list):
                s_list[i] = tf.squeeze(row, [1])

            with tf.variable_scope('h_theta'):
                cell = rnn_cell.BasicLSTMCell(config['rnn_size'], state_is_tuple=True)
                shape = [config['rnn_size'], config['num_actions']]
                Wmlp = self.make_weight(shape)
                bmlp = self.make_bias(config['num_actions'])

                with tf.variable_scope('lstm') as varscope:
                    # likelihood graph - given actions a_k
                    state = cell.zero_state(self.batch_size, tf.float32)
                    logmu = []  # categorical means
                    for i, row in enumerate(s_list):
                        (f_k, state) = cell(row, state)
                        mlp = tf.nn.bias_add(tf.matmul(f_k, Wmlp), bmlp)
                        selected_a = tf.reduce_sum(tf.mul(tf.nn.log_softmax(mlp), self.action_rows[i]), reduction_indices=1)
                        logmu.append(selected_a)
                        varscope.reuse_variables()  # share weights created by cell
                    self.logh = tf.reduce_sum(logmu, 0)  # sum over steps

                    # sampling graph - sample actions a_k (same variable scope as likelihood graph, with reuse)
                    #self.a = [tf.zeros([config['num_actions']])]  # initial action is all zeros (one-hot)
                    state = cell.zero_state(1, tf.float32)  # initial state is all zeros, no batch for sampling
                    self.sampled_a = []
                    self.omega_k = []
                    for k in range(config['n_steps']):
                        # get action distribution from cell
                        (f_k, state) = cell(self.before_s, state)
                        omega_k = tf.nn.log_softmax(tf.nn.bias_add(tf.matmul(f_k, Wmlp), bmlp))
                        self.omega_k.append(omega_k)

                        # store sample
                        self.sampled_a.append(tf.one_hot(tf.multinomial(omega_k, 1)[0,0],
                                              config['num_actions']))

            # psi_theta
            with tf.variable_scope('psi_theta'):
                with tf.variable_scope('psi1'):
                    shape = [config['state_size'], 64]
                    Wpsi1 = self.make_weight(shape)
                    bpsi1 = self.make_bias(64)
                    psi1 = tf.nn.relu_layer(self.before_s, Wpsi1, bpsi1)
                with tf.variable_scope('psi2'):
                    shape = [64, 1]
                    Wpsi2 = self.make_weight(shape)
                    bpsi2 = self.make_bias(1)
                    self.psi = tf.nn.bias_add(tf.matmul(psi1, Wpsi2), bpsi2)
            self.r_theta = self.logh + self.psi

            self.emp = self.psi / self.beta  # beta != 1 genuinely doesn't seem to work?

            # loss for source distribution (theta), block gradient contribution through log q
            #self.L_theta = tf.reduce_sum(tf.square(tf.mul(self.beta, tf.stop_gradient(self.logq)) - self.r_theta),
            #                             reduction_indices=0)
            self.L_theta = tf.nn.l2_loss((self.beta * tf.stop_gradient(self.logq)) - self.r_theta)

            # loss for variational distribution q (maximize q)
            self.L_ksi = tf.reduce_sum(-self.logq, reduction_indices=0)

            # notice: lambda (cnn weights) are trained jointly by ksi_opt and theta_opt
            optimizer = tf.train.AdamOptimizer(config['emp_lr'], config['beta1'],
                                               config['beta2'], config['emp_eps'])
            self.ksi_opt = optimizer.minimize(self.L_ksi)  # notice: we maximize log q
            self.theta_opt = optimizer.minimize(self.L_theta)

            # tensorboard is useful for inspecting the graph visually
            if config['tensorboard']:
                self.merged = tf.merge_all_summaries()
                self.writer = tf.train.SummaryWriter("logs/", self.sess.graph)

            # initialize weights
            self.sess.run(tf.initialize_all_variables())

    def train(self, o, a, op):
        feed_dict = {self.obs_before: o, self.actions: a, self.obs_after: op}

        # training this in one call means theta is trained with one step old q
        #(L_ksi, L_theta, emp, _, _) = self.sess.run([self.L_ksi, self.L_theta, self.emp, self.ksi_opt, self.theta_opt], feed_dict)

        # This is the (correct) alternative, is there a real difference?
        (L_ksi, _) = self.sess.run([self.L_ksi, self.ksi_opt], feed_dict)
        (L_theta, _) = self.sess.run([self.L_theta, self.theta_opt], feed_dict)
        return L_ksi, L_theta

    def state_train(self, s, a, sp):
        feed_dict = {self.before_s: s, self.actions: a, self.after_s: sp}

        (L_ksi, _) = self.sess.run([self.L_ksi, self.ksi_opt], feed_dict)
        (L_theta, _) = self.sess.run([self.L_theta, self.theta_opt], feed_dict)
        (emp, logsoftmus) = self.sess.run([self.emp, self.logsoftmus], feed_dict)
        return L_ksi, L_theta, emp, logsoftmus

    def draw_actions(self, o):
        # hacky, feed obs_after but it's not used
        feed_dict = {self.obs_before: o, self.obs_after: o}

        actions = self.sess.run(self.sampled_a, feed_dict)
        return actions

    def state_draw_actions(self, s):
        feed_dict = {self.before_s: s}

        actions = self.sess.run(self.sampled_a, feed_dict)
        return actions

    def get(self, o):
        # sort of hacky to feed in obs_after, but it is needed just for the state encoder
        feed_dict = {self.obs_before: o, self.obs_after: o}

        return self.sess.run(self.emp, feed_dict)

    def state_get(self, s):
        feed_dict = {self.before_s: s}

        return self.sess.run(self.emp, feed_dict)

    def policy(self, o):
        feed_dict = {self.obs_before: o, self.obs_after: o}

        return self.sess.run(self.omega_k, feed_dict)

    def state_policy(self, s):
        feed_dict = {self.before_s: s}

        return self.sess.run(self.omega_k, feed_dict)

    def predictive(self, o, no):
        feed_dict = {self.obs_before: o, self.obs_after: no}

        return self.sess.run(self.logsoftmus, feed_dict)

    def state_predictive(self, s, ns):
        feed_dict = {self.before_s: s, self.after_s: ns}

        return self.sess.run(self.logsoftmus, feed_dict)

    def get_state(self, o):
        feed_dict = {self.obs_before: o, self.obs_after: o}

        return self.sess.run(self.before_s, feed_dict)

    def dists(self, o, no):
        feed_dict = {self.obs_before: o, self.obs_after: no}

        return self.sess.run([self.logq, self.logh, self.psi])

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def make_weight(shape):
        return tf.get_variable('weight', shape,
                               initializer=tf.uniform_unit_scaling_initializer(factor=1.43))  # 1.43 for relu

    @staticmethod
    def make_bias(shape):
        return tf.get_variable('bias', shape,
                               initializer=tf.constant_initializer(0.01))

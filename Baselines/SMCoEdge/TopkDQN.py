import numpy as np
import tensorflow as tf
from scipy.special import perm, comb
from itertools import combinations


class DeepQNetwork:

    def __init__(self,
                 n_BSs,  # The number of BSs
                 n_features,  # The number of features in observation state
                 n_time,  # The total number of time slot
                 n_k,  # The number of selected ESs
                 learning_rate=0.001,  # The learning rate
                 reward_decay=0.9,  # The factor of reward decay
                 e_greedy=0.99,  # The greedy probability
                 replace_target_iter=200,  # each 200 steps, update target net
                 memory_size=500,  # maximum of memory
                 batch_size=32,
                 e_greedy_increment=0.001,
                 N_L1=20  # set the number of neurons in the l1 layer network
                 ):

        self.n_actions = n_BSs
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size  # select self.batch_size number of time sequence for learning
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.N_L1 = N_L1
        self.k_action = n_k

        # The set of action space
        self.actions_set = np.asarray([c for c in combinations(range(self.n_actions), self.k_action)])
        self.q_values_num = len(self.actions_set) # which is equals to the number of action set

        # initialize zero memory np.hstack((s, a, [r], s_))
        self.memory = np.zeros((self.memory_size, self.n_features + self.k_action + 1 + self.n_features))

        # consist of [target_net, evaluate_net]
        self.build_net()

        # replace the parameters in target net
        t_params = tf.get_collection('target_net_params')  # obtain the parameters in target_net
        e_params = tf.get_collection('eval_net_params')  # obtain the parameters in eval_net

        # update the target_net parameters using eval_net parameters
        self.replace_target_net = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.history_loss = []

        self.store_q_value = list()  # store one MD's Q-value in total time slots (only the last trained MD)

    # Build the network structure, including eval_net and target_net
    def build_net(self):
        tf.reset_default_graph()
        #   Build the network layers：Return the Q-value of an action in the last layer
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):

            # The first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            # The second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # The output layer
            with tf.variable_scope('Q'):
                w3 = tf.get_variable('w3', [n_l1, self.q_values_num], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.q_values_num], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out  # The Q values of all actions with float type

        # input for eval_net
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # state features in observation
        self.q_target = tf.placeholder(tf.float32, [None, self.q_values_num], name='Q_target')  # q_target

        # input for target_net
        self.state_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        # generate eval_net, and update parameters
        with tf.variable_scope('eval_net'):

            # c_names(collections_names), will be used when update target_net
            # tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32), return a initializer
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_L1, \
                tf.random_normal_initializer(0.0, 0.5), tf.constant_initializer(0.1)  # config of layers

            # input (n_feature) -> l1 (n_l1) -> output (q_values_num)
            self.q_eval = build_layers(self.state, c_names, n_l1, w_initializer, b_initializer)

        # loss and train
        with tf.variable_scope('loss'):
            # computing the loss between Q-predict and Q-target using Mean Square Error
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            # self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  #  using RMSprop
            # self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)  # Using Gradient descent
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # using Adam

        # generate target_net
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.state_, c_names, n_l1, w_initializer, b_initializer)

    # ------ store transition information into memory (experience pool) ----
    def store_transition(self, s, a, r, s_):
        # RL.store_transition(observation (state), action, reward, observation_ (state_))
        # hasattr(object, name), if object has name attribute
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # store np.hstack((s, a, [r], s_))
        transition = np.hstack((s, a, [r], s_,))  # stack in horizontal direction

        # if memory overflows, replace old memory with new one
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # --- Choose tha action of maximizing q_value in evaluation network. -------
    def choose_action(self, observation):
        # Modify the shape of the observation in (1, size_of_observation)
        # e.g., if x1 = np.array([1, 2, 3, 4, 5]), then x1_new = x1[np.newaxis, :]. Now the shape of x1_new is (1, 5)
        observation = observation[np.newaxis, :]  # to have batch dimension when feed into tf placeholder

        # forward feed the observation and get q value for every actions
        if np.random.uniform() < self.epsilon:
            q_values = self.sess.run(self.q_eval, feed_dict={self.state: observation})  # Get all Q values by running
            self.store_q_value.append({'observation': observation, 'q_value': q_values})  # Store Q values

            # Note the q_values is 2D array. So, here we should get first row values, i.e., q_values[0]
            action = self.actions_set[np.argmax(q_values[0])]  # Achieve the action of the maximal Q-value
        else:
            action = self.actions_set[np.random.randint(0, self.q_values_num)]  # A random action in the actions set

        return action

    # model learning
    def learn(self):
        # check if replace target_net parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # run the self.replace_target_op in __int__
            self.sess.run(self.replace_target_net)
            # print('Replaced target_net parameters\n')

        # randomly pick [batch_size] memory from memory np.hstack((s, a, r, s_))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # transition tuple = np.hstack(s, a, [r], s_)
        batch_memory = self.memory[sample_index, :]

        # obtain q_next (from target_net) (to q_target) and q_eval (from eval_net)
        # minimize（target_q - q_eval）^2
        # q_target = reward + gamma * q_next
        # in the size of bacth_memory
        # q_next, given the s_ state from batch, to get q_next
        # q_eval, given the s state from batch, to get q_eval
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],  # output
            feed_dict={
                # input for target_q (last)
                self.state_: batch_memory[:, -self.n_features:],  # fixed parameters
                # input for eval_q (last)
                self.state: batch_memory[:, :self.n_features],  # newest parameters
            }
        )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # Get all actions in batch memory, where each action is vector with binary values
        eval_act_index = batch_memory[:, self.n_features:self.n_features + self.k_action].astype(int)

        # Get all reward indexes in batch memory, where each reward is a single value
        reward = batch_memory[:, self.n_features + self.k_action]

        # update the q_target at the particular batch at the corresponding action
        selected_q_next = np.max(q_next, axis=1)  # Achieve the maximal Q-values

        for i in range(self.batch_size):
            for j in range(self.q_values_num):
                if (eval_act_index[i] == self.actions_set[j]).all():
                    q_target[i, j] = reward[i] + self.gamma * selected_q_next[i]
                    break


        # both self.s and self.q_target belong to eval_q
        # input self.s and self.q_target, output self.train_op, self.loss (to minimize the gap)
        # self.sess.run: given input (feed), output the required element
        _, self.loss_ = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.state: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        # gradually increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        self.history_loss.append(self.loss_)  # store history cost




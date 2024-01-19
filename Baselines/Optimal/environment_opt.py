import numpy as np
from itertools import combinations


class OffloadEnvironment:
    def __init__(self, num_tasks, num_BSs, num_time, deadline, es_capacities, prob):
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices
        self.n_BSs = num_BSs  # The number of base station or edge server
        self.n_time = num_time  # The number of time slot set
        self.duration = 1.0  # The length of each time slot t. Unit: second
        self.ES_capacities = es_capacities  # GHz or Gigacycles/s
        self.tran_rate_BSs = np.array([400, 425, 450, 475, 500])  # Mbits/s
        # Computing density in range (100, 300) Cycles/Bit
        self.comp_density = 1024 / 1000000 * np.random.uniform(100, 300, size=[self.n_tasks])  # Gigacycles/Mbit
        self.deadline_delay = deadline  # In second
        self.max_bit = 5  # The maximal bit of arrival tasks. Mbits
        self.min_bit = 2  # The minimal bit of arrival tasks. Mbits
        self.task_arrive_prob = prob  # Set the task arrival probability by default
        # Initialize the array to storage all the arrival task data in the system
        self.arrival_bits = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the make-span of each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the array to storage the queue workload lengths before the current processing task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the transmission delay of tasks before the current task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])  # In seconds
        # Initialize the array to storage the task offloading failure results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator

        # The set of action space
        self.action_space = np.zeros([2 ** self.n_BSs - 1, self.n_BSs])
        a_index = 0
        for i in range(1, 6):
            temp_action = np.asarray([c for c in combinations(range(self.n_BSs), i)])
            len_ = len(temp_action)
            for j in range(len_):
                self.action_space[a_index, temp_action[j]] = 1
                a_index = a_index + 1
        self.action_space_len = np.size(self.action_space[:, 0])  # which is equals to the number of action set
        self.all_actions = list(np.zeros(self.n_BSs))  # which is equals to the number of action set
        self.make_spans_ = list()  # which is equals to the number of action set

    # Initialize system environment
    # Return: The state at time slot 0
    def initialize_env(self, arrival_bits_):
        # Initial all the arrival task data in the system
        self.arrival_bits = arrival_bits_
        # Initial a maximize service delay for each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the queue workload lengths before the current processing task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the transmission delay of tasks before the current transmission task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])  # In seconds
        # Initial the task offloading results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator
        self.all_actions = list(np.zeros(self.n_BSs)) # which is equals to the number of action set

    # Perform optimal task offloading to achieve:
    # (1) Service delays;
    # (2) Fail results;
    # (3) The queue workload lengths of arrival tasks in all ES.
    def opt_perform_offloading(self, t, b, n):
        allocation_fractions = np.zeros(self.n_BSs)
        tran_comp_delays = np.zeros(self.n_BSs)
        wait_delays = np.zeros(self.n_BSs)
        n_bit = self.arrival_bits[t][b][n]
        min_make_span = 10
        opt_action = []
        for i in range(self.action_space_len):
            # Initial partition fractions for each task
            allocation_fractions_temp = np.zeros(self.n_BSs)
            a_btn = np.argwhere(self.action_space[i] == 1).flatten()  # get the action of task n and convert it into integer
            action_size = np.size(a_btn)
            if action_size == 1:
                allocation_fractions_temp[a_btn] = 1
                if a_btn[0] == b:
                    wait_delays[a_btn] = (self.proc_queue_len[t][a_btn] + self.proc_queue_bef[t][a_btn]) / self.ES_capacities[a_btn]
                    tran_comp_delays[a_btn] = n_bit * self.comp_density[n] / self.ES_capacities[a_btn]
                else:
                    wait_delays[a_btn] = self.tran_delay_bef[t][b] + (self.proc_queue_len[t][a_btn] + self.proc_queue_bef[t][a_btn]) / self.ES_capacities[a_btn]
                    tran_comp_delays[a_btn] = n_bit / self.tran_rate_BSs[a_btn] + n_bit * self.comp_density[n] / self.ES_capacities[a_btn]
                n_delay = tran_comp_delays[a_btn] + wait_delays[a_btn]
            else:
                # Calculate the workload allocation fractions of the selected ESs by Equations Solving method
                for j in range(action_size):
                    if a_btn[j] == b:
                        wait_delays[a_btn[j]] = (self.proc_queue_len[t][a_btn[j]] + self.proc_queue_bef[t][a_btn[j]]) / self.ES_capacities[a_btn[j]]
                        tran_comp_delays[a_btn[j]] = n_bit * self.comp_density[n] / self.ES_capacities[a_btn[j]]
                    else:
                        wait_delays[a_btn[j]] = self.tran_delay_bef[t][b] + (self.proc_queue_len[t][a_btn[j]] + self.proc_queue_bef[t][a_btn[j]]) / self.ES_capacities[a_btn[j]]
                        tran_comp_delays[a_btn[j]] = n_bit / self.tran_rate_BSs[a_btn[j]] + n_bit * self.comp_density[n] / self.ES_capacities[a_btn[j]]

                eq_A = np.zeros([action_size, action_size])  # Initialize the left side coefficients of equations
                eq_b = np.ones([action_size])  # Initialize the right side coefficients of equations
                for k in range(action_size - 1):
                    eq_A[k][k] = tran_comp_delays[a_btn[k]]
                    eq_A[k][k + 1] = -tran_comp_delays[a_btn[k + 1]]
                    eq_b[k] = wait_delays[a_btn[k + 1]] - wait_delays[a_btn[k]]

                eq_A[action_size - 1] = np.ones([action_size])
                # Achieve the allocation fractions by Equations Solving
                allocation_fractions_temp[a_btn] = np.linalg.solve(eq_A, eq_b)
                # Here, we should ensure the allocation fractions are greater than 0 by the following operation
                allocation_fractions_temp[allocation_fractions_temp < 0] = 0
                # Reset the allocation fractions
                allocation_fractions_temp[a_btn] = allocation_fractions_temp[a_btn] / np.sum(allocation_fractions_temp[a_btn])
                # Calculate the service delay of task n, which equals to the longest service delay of subtasks
                n_delay = np.max(allocation_fractions_temp[a_btn] * tran_comp_delays[a_btn] + wait_delays[a_btn])

            if n_delay < min_make_span:
                min_make_span = n_delay
                opt_action = a_btn
                allocation_fractions = allocation_fractions_temp

        print("Output allocation decision x_" + str(b) + "," + str(n) + "," + str(t) + "=", allocation_fractions)

        self.make_spans[t][b][n] = min_make_span
        # Record the fail task and Calculate the reward of the action of task n offloading
        if self.make_spans[t][b][n] > self.deadline_delay:
            self.is_fail_tasks[t][b][n] = 1
            self.make_spans[t][b][n] = self.deadline_delay

        # Update the transmission delay of tasks before the current task
        action_size_ = np.size(opt_action)
        for i in range(action_size_):
            if opt_action[i] != b:
                self.tran_delay_bef[t][b] = self.tran_delay_bef[t][b] + allocation_fractions[opt_action[i]] * n_bit / self.tran_rate_BSs[opt_action[i]]

        # Update the queue workload length at the selected ESs before the current processing task
        self.proc_queue_bef[t][opt_action] = self.proc_queue_bef[t][opt_action] + allocation_fractions[opt_action] * n_bit * self.comp_density[n]

    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_queues(self, t):
        for b_ in range(self.n_BSs):
            self.proc_queue_len[t + 1][b_] = np.max(
                [self.proc_queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.ES_capacities[b_] * self.duration, 0])

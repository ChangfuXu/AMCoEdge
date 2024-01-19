import numpy as np


class OffloadEnvironment:
    def __init__(self, num_tasks, num_BSs, num_time, deadline, es_capacities, prob):
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices
        self.n_BSs = num_BSs  # The number of base station or edge server
        self.n_time = num_time  # The number of time slot set
        self.duration = 1.0  # The length of each time slot t. Unit: second

        self.BS_capacities = es_capacities  # GHz or Gigacycles/s
        # self.tran_rate_BSs = np.random.uniform(400, 500, size=[self.n_BSs])  # Mbits/s
        self.tran_rate_BSs = np.array([400, 425, 450, 475, 500])  # Mbits/s
        # Set the computing density in range (100, 300) cycles/Mbit
        self.comp_density = 1024 / 1000000 * np.random.uniform(100, 300, size=[self.n_tasks])  # Gigacycles/Mbit
        self.deadline_delay = deadline  # in second
        # Set the range of task size
        self.max_bit = 5  # The maximal bit of arrival tasks. Mbits
        self.min_bit = 2  # The minimal bit of arrival tasks. Mbits
        self.task_arrive_prob = prob  # Set the task arrival probability by default
        # Initialize all the arrival task data in the system
        self.arrival_bits = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the make-span of each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the processing queue lengths before processing the current task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the transmission delay of tasks before the current task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])  # In seconds
        # Initialize the array to storage the task offloading failure results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator
        self.n_features = 1 + self.n_BSs

    # Initialize system environment
    def initialize_env(self, arrival_bits_):
        # Initial all the arrival task data in the system
        self.arrival_bits = arrival_bits_
        # Initial a maximize service delay for each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the queue workload lengths before processing current  task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the transmission delay of task before the current task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])  # In seconds
        # Initial the task offloading results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator

    # Perform task offloading to achieve:
    # (1) Allocation fractions and task processing make-spans;
    # (2) Fail results;
    # (3) Update the queue data size before the current transmission task
    # (4) Update the queue workload length before the current processing task
    def perform_offloading(self, t, b, n, action_):
        allocation_fractions = np.zeros(self.n_BSs)  # Initialize the workload allocation fraction
        tran_comp_delays = np.zeros(self.n_BSs)  # Initialize the transmission and computing delays
        wait_delays = np.zeros(self.n_BSs)  # Initialize the wait delay before processing current task
        n_bit = self.arrival_bits[t][b][n]  # Achieve the size of current arrival task
        a_btn = np.array([b, action_]).astype(int)  # get the action of task n and convert it into integer
        # Calculate the workload allocation fractions of the selected ESs by Equations Solving method
        if action_ == b:  # Here, the transmission delay and transmission queuing delay equal to 0
            action_size = 1
            allocation_fractions[b] = 1
            wait_delays[b] = np.min([(self.proc_queue_len[t][b] + self.proc_queue_bef[t][b]) / self.BS_capacities[b], self.deadline_delay])
            tran_comp_delays[b] = n_bit * self.comp_density[n] / self.BS_capacities[b]
            self.make_spans[t][b][n] = tran_comp_delays[b] + wait_delays[b]
        else:
            action_size = 2
            wait_delays[b] = np.min([(self.proc_queue_len[t][b] + self.proc_queue_bef[t][b]) / self.BS_capacities[b], self.deadline_delay])
            wait_delays[action_] = np.min([self.tran_delay_bef[t][b] + (self.proc_queue_len[t][action_] + self.proc_queue_bef[t][action_]) / self.BS_capacities[action_], self.deadline_delay])
            tran_comp_delays[b] = n_bit * self.comp_density[n] / self.BS_capacities[b]
            tran_comp_delays[action_] = n_bit / self.tran_rate_BSs[action_] + n_bit * self.comp_density[n] / self.BS_capacities[action_]

            eq_A = np.zeros([action_size, action_size])  # Initialize the left side coefficients of equations
            eq_b = np.ones([action_size])  # Initialize the right side coefficients of equations
            for i in range(action_size - 1):
                eq_A[i][i] = tran_comp_delays[a_btn[i]]
                eq_A[i][i + 1] = -tran_comp_delays[a_btn[i + 1]]
                eq_b[i] = wait_delays[a_btn[i + 1]] - wait_delays[a_btn[i]]

            eq_A[action_size - 1] = np.ones([action_size])
            # Achieve the allocation fractions by Equations Solving
            allocation_fractions[a_btn] = np.linalg.solve(eq_A, eq_b)
            # Here, we should ensure the allocation fractions are greater than 0 by the following operation
            allocation_fractions[allocation_fractions < 0] = 0
            # Reset the allocation fractions
            allocation_fractions[a_btn] = allocation_fractions[a_btn] / np.sum(allocation_fractions[a_btn])
            # Calculate the service delay of task n, which equals to the longest service delay of subtasks
            self.make_spans[t][b][n] = np.max(allocation_fractions[a_btn] * tran_comp_delays[a_btn] + wait_delays[a_btn])

        # Update the transmission delay of tasks before the current task
        for k in range(action_size):
            self.tran_delay_bef[t][b] = self.tran_delay_bef[t][b] + allocation_fractions[a_btn[k]] * n_bit / self.tran_rate_BSs[a_btn[k]]

        print("Output allocation decision x_" + str(b) + "," + str(n) + "," + str(t) + "=", allocation_fractions)

        # Update the queue workload length before the current processing task
        self.proc_queue_bef[t][a_btn] = self.proc_queue_bef[t][a_btn] + allocation_fractions[a_btn] * n_bit * self.comp_density[n]

        if self.make_spans[t][b][n] > self.deadline_delay:
            self.is_fail_tasks[t][b][n] = 1
            self.make_spans[t][b][n] = self.deadline_delay

    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_queues(self, t):
        for b_ in range(self.n_BSs):
            self.proc_queue_len[t + 1][b_] = np.max(
                [self.proc_queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.BS_capacities[b_] * self.duration, 0])




import numpy as np


class OffloadEnvironment:
    def __init__(self, num_tasks, num_BSs, num_time, deadline, es_capacities, task_arrivalL_prob, alg_type):
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices
        self.n_BSs = num_BSs  # The number of base station or edge server
        self.n_time = num_time  # The number of time slot set
        self.duration = 1.0  # The length of each time slot t. Unit: second
        self.ES_capacities = es_capacities  # GHz or Gigacycles/s
        self.tran_rate_BSs = np.array([400, 425, 450, 475, 500])  # Mbits/s
        # Computing density in range (100, 300) Cycles/Bit
        self.comp_density = 1024 / 1000000 * np.random.uniform(100, 300, size=[self.n_tasks])  # Gigacycles/Mbit
        self.deadline_delay = deadline  # The deadline (In seconds) of task processing.
        # Set the range of task size
        self.max_bit = 5  # The maximal bit of arrival tasks. Mbits
        self.min_bit = 2  # The minimal bit of arrival tasks. Mbits
        self.task_arrive_prob = task_arrivalL_prob  # Set the task arrival probability by default
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
        self.n_features = 1 + self.n_BSs
        self.algorithm_type = alg_type  # Using SE method

    # Initialize system environment
    # Return: The state at time slot 0
    def reset_env(self, arrival_bits_):
        # Initial all the arrival task data in the system
        self.arrival_bits = arrival_bits_
        # Initial a maximize service delay for each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initialize the array to storage the queue workload lengths of all ESs
        self.proc_queue_len = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the array to storage the queue workload lengths before the current processing task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])  # Gigacycles
        # Initialize the transmission delay of tasks before the current task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])  # In seconds
        # Initial the task offloading results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator

    # Perform task offloading to achieve:
    # (1) Service delays;
    # (2) Fail results;
    # (3) The queue workload lengths of arrival tasks in all ES.
    def perform_offloading(self, t, b, n, action_set):
        allocation_fractions = np.zeros(self.n_BSs)  # Initialize the workload allocation fraction
        tran_comp_delays = np.zeros(self.n_BSs)  # Initialize the transmission and computing delays
        wait_delays = np.zeros(self.n_BSs)  # Initialize the wait delay before processing current task
        n_bit = self.arrival_bits[t][b][n]  # Achieve the size of current arrival task
        a_btn = np.argwhere(action_set == 1).flatten()  # Get the action index of task n at BS b in time slot t
        action_size = np.size(a_btn)
        if action_size == 1:
            allocation_fractions[a_btn] = 1
            if a_btn[0] == b:  # Here, the transmission delay and transmission queuing delay equal to 0
                wait_delays[b] = (self.proc_queue_len[t][b] + self.proc_queue_bef[t][b]) / self.ES_capacities[b]
                tran_comp_delays[b] = n_bit * self.comp_density[n] / self.ES_capacities[b]
                # Note that when the BS b is selected, the transmission queue doesn't need to update
            else:
                wait_delays[a_btn] = self.tran_delay_bef[t][b] + (self.proc_queue_len[t][a_btn] + self.proc_queue_bef[t][a_btn]) / self.ES_capacities[a_btn]
                tran_comp_delays[a_btn] = n_bit / self.tran_rate_BSs[a_btn] + n_bit * self.comp_density[n] / self.ES_capacities[a_btn]
                self.tran_delay_bef[t][b] = self.tran_delay_bef[t][b] + n_bit / self.tran_rate_BSs[a_btn]
            self.make_spans[t][b][n] = tran_comp_delays[a_btn] + wait_delays[a_btn]
        else:
            # Calculate the workload allocation fractions of the selected ESs by Equations Solving method
            for i in range(action_size):
                if a_btn[i] == b:
                    wait_delays[b] = (self.proc_queue_len[t][b] + self.proc_queue_bef[t][b]) / self.ES_capacities[b]
                    tran_comp_delays[b] = n_bit * self.comp_density[n] / self.ES_capacities[b]
                else:
                    wait_delays[a_btn[i]] = self.tran_delay_bef[t][b] + (self.proc_queue_len[t][a_btn[i]] + self.proc_queue_bef[t][a_btn[i]]) / self.ES_capacities[a_btn[i]]
                    tran_comp_delays[a_btn[i]] = n_bit / self.tran_rate_BSs[a_btn[i]] + n_bit * self.comp_density[n] / self.ES_capacities[a_btn[i]]

            if self.algorithm_type == 'CWA':  # Using CWA method
                eq_A = np.zeros([action_size, action_size])  # Initialize the left side coefficients of equations
                eq_b = np.ones([action_size])  # Initialize the right side coefficients of equations
                for j in range(action_size - 1):
                    eq_A[j][j] = tran_comp_delays[a_btn[j]]
                    eq_A[j][j + 1] = -tran_comp_delays[a_btn[j + 1]]
                    eq_b[j] = wait_delays[a_btn[j + 1]] - wait_delays[a_btn[j]]

                eq_A[action_size - 1] = np.ones([action_size])
                # Achieve the allocation fractions by Equations Solving
                allocation_fractions[a_btn] = np.linalg.solve(eq_A, eq_b)
                # Here, we should ensure the allocation fractions are greater than 0 by the following operation
                allocation_fractions[allocation_fractions < 0] = 0
                # Reset the allocation fractions
                allocation_fractions[a_btn] = allocation_fractions[a_btn] / np.sum(allocation_fractions[a_btn])
            else:  # Using HECWA method
                for i in range(action_size):
                    except_i_indexes = np.delete(a_btn, i)  # Get the ES indexes except the current ES index
                    allocation_fractions[a_btn[i]] = np.prod(tran_comp_delays[except_i_indexes])
                temp_sum = np.sum(allocation_fractions[a_btn])
                allocation_fractions[a_btn] = allocation_fractions[a_btn] / temp_sum

            # Calculate the service delay of task n, which equals to the longest service delay of subtasks
            self.make_spans[t][b][n] = np.max(allocation_fractions[a_btn] * tran_comp_delays[a_btn] + wait_delays[a_btn])

            # Update the transmission delay of tasks before the current task
            # Ensure that the transmission delay of task before current task is lower than the length of one time slot.
            for k in range(action_size):
                if a_btn[k] != b:
                    self.tran_delay_bef[t][b] = np.min([self.tran_delay_bef[t][b] + allocation_fractions[a_btn[k]] * n_bit / self.tran_rate_BSs[a_btn[k]], self.duration])

        print("Output allocation decision x_" + str(b) + "," + str(n) + "," + str(t) + "=", allocation_fractions)

        # Update the processing queue workload lengths at the selected ESs before the current processing task
        self.proc_queue_bef[t][a_btn] = self.proc_queue_bef[t][a_btn] + allocation_fractions[a_btn] * n_bit * self.comp_density[n]
        if self.make_spans[t][b][n] > self.deadline_delay:  # Failure offloading
            self.is_fail_tasks[t][b][n] = 1  # Record the failure task
            self.make_spans[t][b][n] = self.deadline_delay  # Set the make-span with deadline delay
            reward = -10 * self.deadline_delay  # Set the punishment
        else:
            reward = -self.make_spans[t][b][n]  # Set the reward
        return reward

    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_queues(self, t):
        for b_ in range(self.n_BSs):
            self.proc_queue_len[t + 1][b_] = np.max(
                [self.proc_queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.ES_capacities[b_] * self.duration, 0])


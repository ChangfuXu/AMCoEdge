from environment import OffloadEnvironment
from AdaDQN import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt


def online_AMCoEdge_algorithm(DQN_list_, NUM_EPISODE_):
    action_step = 0
    for episode_ in range(NUM_EPISODE_):
        # print('Episode: %d' % episode_)
        # Arrival task bits with a probability
        arrival_tasks = np.random.uniform(env.min_bit, env.max_bit, size=[env.n_time, env.n_BSs, env.n_tasks])
        arrival_tasks = arrival_tasks * (np.random.uniform(0, 1, size=[env.n_time, env.n_BSs, env.n_tasks]) < env.task_arrive_prob)
        # Initialize the system environment
        env.reset_env(arrival_tasks)
        # Initialize an array to storage the system state at next time slot t
        state_bnt = np.zeros([env.n_BSs, env.n_tasks, env.n_features])  # State: [D_{n,t},Q_{1,t},..., Q_{B,t}]

        # ========================================= DRL ===================================================
        # -------- Train DQN model with reinforcement learning ----------------
        # for time_index in range(0, NUM_TIME_):  # Perform action in the set (T) of time slots
        for t in range(env.n_time - 1):
            for b in range(env.n_BSs):
                for n in range(env.n_tasks):
                    state_bnt[b][n] = np.hstack([env.arrival_bits[t][b][n], env.proc_queue_len[t]])
                    if state_bnt[b][n][0] != 0:
                        all_actions[b][n] = DQN_list_[b].choose_action(state_bnt[b][n])
                        # Perform task offloading and achieve its reward
                        reward[b][n] = env.perform_offloading(t, b, n, all_actions[b][n])
            env.update_queues(t)  # Update the processing queue of all ESS
            if t > 0:  # Since the initial queue length is set to 0, the samples at time slot are dropped.
                for b in range(env.n_BSs):
                    for n in range(env.n_tasks):
                        if state_bnt[b][n][0] != 0:
                            if n == env.n_tasks - 1:
                                next_state_bnt = np.hstack([env.arrival_bits[t + 1][b][0], env.proc_queue_len[t + 1]])
                            else:
                                next_state_bnt = np.hstack([env.arrival_bits[t][b][n + 1], env.proc_queue_len[t]])
                            # Store transition tuple
                            DQN_list_[b].store_transition(state_bnt[b][n], all_actions[b][n], reward[b][n], next_state_bnt)


            action_step += 1  # Add action step (one step does not mean one store)
            # Set learning start time as bigger than 200 and frequency with each 10 steps
            if (action_step > 200) and (action_step % 10 == 0):
                for b in range(env.n_BSs):
                    DQN_list_[b].learn()

        make_spans = env.make_spans.flatten()
        make_spans = make_spans[make_spans > 0]
        aver_make_spans[episode_] = np.mean(make_spans)
        all_num_tasks = np.size(make_spans)
        aver_failure_rates[episode_] = np.sum(env.is_fail_tasks) / all_num_tasks
        #  ======================================== DQN END =================================================


if __name__ == "__main__":
    NUM_BSs = 5  # The number of Base Stations （BSs）or Edge Servers (ESs)
    NUM_TASK = 50  # The max number of task in each BS
    NUM_EPISODE = 500  # The number of episodes
    NUM_TIME_SLOTS = 61  # The number of time slot set
    DEADLINE = 1.0  # The deadline (In seconds) of task processing
    TASK_ARRIVAL_PROB = 0.3  # The task arrival probability
    # ES_capacity = np.array([10, 15, 20, 25, 30])  # The computing capacity of ES
    # ES_capacity = np.array([10, 20, 30, 40, 50])  # The computing capacity of ES
    # ES_capacity = np.array([10, 25, 40, 55, 70])  # The computing capacity of ES
    ES_capacity = np.array([10, 30, 50, 70, 90])  # The computing capacity of ES
    ALGORITHM_TYPE = "CWA"  # Set the algorithm type (i.e., CWA or HECWA) for workload allocation

    # Initial variables of actions, delays, and failure rate for experimental testing
    all_actions = np.zeros([NUM_BSs, NUM_TASK, NUM_BSs])
    aver_make_spans = np.zeros([NUM_EPISODE])
    aver_failure_rates = np.zeros([NUM_EPISODE])
    reward = np.zeros([NUM_BSs, NUM_TASK])
    # Generate offloading environment
    env = OffloadEnvironment(NUM_TASK, NUM_BSs, NUM_TIME_SLOTS, DEADLINE, ES_capacity, TASK_ARRIVAL_PROB, ALGORITHM_TYPE)
    # Distributed DQN model: Each BS generates an agent class for DRL
    DQN_list = list()
    for i in range(NUM_BSs):
        DQN_list.append(DeepQNetwork(env.n_BSs, env.n_features, env.n_time,  # GENERATE ENVIRONMENT
                                     learning_rate=0.001,
                                     reward_decay=0.9,  # discount factor
                                     e_greedy=0.99,
                                     e_greedy_increment=0.001,
                                     replace_target_iter=100,  # each 200 steps, update target net
                                     memory_size=500,  # maximum of memory
                                     batch_size=32,  # batch size of samples
                                     N_L1=20)
                        )

    online_AMCoEdge_algorithm(DQN_list, NUM_EPISODE)

    print('============ Training finished ==========')

    #  Plot loss varying the training steps
    # loss = np.array(DQN_list[0].history_loss)
    # for j in range(NUM_BSs - 1):
    #     loss = loss + np.array(DQN_list[j + 1].history_loss)
    # loss = loss / NUM_BSs
    # np.savetxt('results/Loss_AMCoEdge_tasks' + str(NUM_TASK) + '_prob' + str(np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.csv', loss, delimiter=',', fmt='%.4f')
    # plt.figure(1)
    # plt.plot(np.arange(len(loss)), loss)
    # plt.ylabel('Loss')
    # plt.xlabel('Training step')
    # plt.savefig('results/Loss_AMCoEdge_tasks' + str(NUM_TASK) + '_prob' + str(np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.png')
    # plt.close()
    # Plot the average delay varying episodes
    np.savetxt('results/AveMakeSpan_AMCoEdge_tasks' + str(NUM_TASK) + '_prob' + str(np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.csv', aver_make_spans, delimiter=',', fmt='%.4f')
    plt.figure(2)
    plt.plot(aver_make_spans)
    plt.ylabel('Average make-span')
    plt.xlabel('Episode')
    plt.savefig('results/AveMakeSpan_AMCoEdge_tasks' + str(NUM_TASK) + '_prob' + str(np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.png')
    plt.close()
    # Plot the average failure rate varying episodes
    np.savetxt('results/AveFailRate_AMCoEdge_tasks' + str(NUM_TASK) + '_prob' + str(np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.csv', aver_failure_rates, delimiter=',', fmt='%.4f')
    plt.figure(3)
    plt.plot(aver_failure_rates)
    plt.ylabel('Average failure rate')
    plt.xlabel('Episode')
    plt.savefig('results/AveFailRate_AMCoEdge_tasks' + str(NUM_TASK) + '_prob' + str(np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.png')
    plt.close()

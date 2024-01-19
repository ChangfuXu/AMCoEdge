from environment_opt import OffloadEnvironment
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    NUM_BSs = 5  # The number of Base Stations （BSs）or Edge Servers (ESs)
    NUM_TASK = 50  # The max number of task in each BS
    NUM_EPISODE = 300  # The number of episode
    NUM_TIME_SLOTS = 61  # The number of time slot set
    DEADLINE = 1.0  # The deadline of task processing
    ES_capacity = np.array([10, 20, 30, 40, 50])  # The computing capacity of ES
    TASK_ARRIVAL_PROB = 0.3  # The task arrival probability

    # Initial variables of actions, delays, and failure rate for experimental testing
    aver_make_spans = np.zeros([NUM_EPISODE])
    aver_failure_rates = np.zeros([NUM_EPISODE])
    # Generate offloading environment
    env = OffloadEnvironment(NUM_TASK, NUM_BSs, NUM_TIME_SLOTS, DEADLINE, ES_capacity, TASK_ARRIVAL_PROB)
    for episode_ in range(NUM_EPISODE):
        # print('Episode: %d' % episode_)
        # Arrival task bits with a probability
        arrival_tasks = np.random.uniform(env.min_bit, env.max_bit, size=[env.n_time, env.n_BSs, env.n_tasks])
        arrival_tasks = arrival_tasks * (np.random.uniform(0, 1, size=[env.n_time, env.n_BSs, env.n_tasks]) < env.task_arrive_prob)
        env.initialize_env(arrival_tasks)  # Initialize the system environment
        # for time_index in range(0, NUM_TIME_):  # Perform action in the set (T) of time slots
        for t in range(env.n_time - 1):
            for b in range(env.n_BSs):
                for n in range(env.n_tasks):
                    data_bnt = env.arrival_bits[t][b][n]
                    if data_bnt != 0:
                        env.opt_perform_offloading(t, b, n)
            env.update_queues(t)  # Update the processing queue of all ESs
        make_spans = env.make_spans.flatten()
        make_spans = make_spans[make_spans > 0]
        aver_make_spans[episode_] = np.mean(make_spans)
        all_num_tasks = np.size(make_spans)
        aver_failure_rates[episode_] = np.sum(env.is_fail_tasks) / all_num_tasks

    print('============ Training finished ==========')

    # Plot the average delay varying episodes
    np.savetxt('results/AveMakeSpan_Optimal_tasks' + str(NUM_TASK) + '_prob' + str(
        np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(
        ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.csv', aver_make_spans, delimiter=',', fmt='%.4f')
    plt.figure(2)
    plt.plot(aver_make_spans)
    plt.ylabel('Average make-span')
    plt.xlabel('Episode')
    plt.savefig('results/AveMakeSpan_Optimal_tasks' + str(NUM_TASK) + '_prob' + str(
        np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(
        ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.png')
    plt.close()
    # Plot the average failure rate varying episodes
    np.savetxt('results/AveFailRate_Optimal_tasks' + str(NUM_TASK) + '_prob' + str(
        np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(
        ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.csv', aver_failure_rates, delimiter=',', fmt='%.4f')
    plt.figure(3)
    plt.plot(aver_failure_rates)
    plt.ylabel('Average failure rate')
    plt.xlabel('Episode')
    plt.savefig('results/AveFailRate_Optimal_tasks' + str(NUM_TASK) + '_prob' + str(
        np.round(TASK_ARRIVAL_PROB, 1)) + '_deadline' + str(np.round(env.deadline_delay, 1)) + '_f10-' + str(
        ES_capacity[4]) + '_episode' + str(NUM_EPISODE) + '.png')
    plt.close()


This code is a implementation of our paper "Enhancing AI-Generated Content Efficiency through Adaptive Multi-Edge Collaboration", submitted to ICDCS 2024.
The function of our method mainly two stages:
1) Make the ES selection decision by the proposed adaptive DQN model in the first stage.
2) Make the workload allocation decision by the proposed Closed-form Workload Allocation (CWA) algorithm in the second stage.

To run this code, please install packages: tensorflow 1.4.0., NumPy, and matplotlib.

This code consists of three files AdaDQN.py, environment.py, and main.py.

The main.py file is the main code. User should run this code to acheive the experimental results.

The environment.py inculdes the code for MEC environment. In this file, some environment parameters such as ESs' computing capacities, task size, and transmission rate can be ajusted by user.

The AdaDQN.py includes the code for Adaptive deep reinforcement learning model. 

In addtion, some experiment results are stored in the results directory. User can run the main.py file to achieve these results again. Note that sometimes, the results may be some devivations. However, user can achieve the better results by running more times.

Here, the parameters in current code is set by default. Hence, user can ajust the parameters such as task deadline, task arrival probabiltiy, computing capacity, and the number of tasks to run the main.py to get more experimental results.

Paramters setting information: 
1) Deadline setting. User can adjust the variable value of DEADLINE in main.py. For example, when DEADLINE is set from 0.1 to 1.0 seconds.
2) Task arrival probabiltiy setting. User can adjust the variable value of TASK_ARRIVAL_PROB from 0.1 to 1 in main.py.
3) The number setting of tasks. User can adjust the variable value of NUM_TASK from 10 to 130 in main.py.
4) The setting of ESs' computing capacities. If user want to achieve the results varying different ESs' computing capacityes, user can adjust the variable value of BS_capacities in main.py and then run the main.py to achieve the experimental results.

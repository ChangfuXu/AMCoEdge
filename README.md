# Enhancing AI-Generated Content Efficiency through Adaptive Multi-Edge Collaboration
This code is a implementation of our paper "Enhancing AI-Generated Content Efficiency through Adaptive Multi-Edge Collaboration", submitted to ICDCS 2024.
The function of our AMCoEdge method mainly two stages:
1) Make the ES selection decision by the proposed adaptive DQN model in the first stage.
2) Make the workload allocation decision by the proposed Closed-form Workload Allocation (CWA) algorithm in the second stage.

To run this code, please install packages: tensorflow 1.4.0., NumPy, and matplotlib.

The code of our method mainly consists of three files: AdaDQN.py, environment.py, and main.py.

The main.py file is the main code. User should run this code to acheive the experimental results.

The environment.py inculdes the code for MEC environment. In this file, some environment parameters such as ESs' computing capacities, task size, and transmission rate can be ajusted by user.

The AdaDQN.py includes the code for Adaptive deep reinforcement learning model. 

# Results
Some experimental results of each method are stored in the corresponding results directory. User can run the corresponding main.py file to achieve these results again. Note that sometimes, the results may be some devivations. However, user can achieve the better results by running more times.

We implement four baselines (i.e., RandCoEdge, DRLCoEdge, SMCoEdge, and Optimal) in our experiments. The RandCoEdge and Optimal baselines are implemented based on the code of AMCoEdge method. The DRLCoEdge and SMCoEdge baselines are implemented based on the existing methods in [1] and [2] respectively. Note that for fairness, the wokload allocation decisions of all methods are satisfied with the same constraints in the experiment. For example, we add the CWA algorithm into the DRLCoEdge baselines. Some environment parameters in the SMCoEdge code are set with the same of our AMCoEdge method. The soure code of SMCoEdge method is released at https://github.com/ChangfuXu/SMCoEdge. User can open this website to find more information about SMCoEdge method.

[1] M. Li, J. Gao, L. Zhao, and X. Shen, “Deep reinforcement learning for collaborative edge computing in vehicular networks,” IEEE Transactions on Cognitive Communications and Networking, vol. 6, no. 4, pp. 1122–1135, 2020.
[2] C. Xu, Y. Li, X. Chu, H. Zou, W. Jia, and T. Wang, “Smcoedge: Simultaneous multi-server offloading for collaborative mobile edge computing,” in The 23rd International Conference on Algorithms and Architectures for Parallel Processing (ICA3PP). Springer, 2023, pp. 1-18, 2023. 

# Paramters setting
All the parameters in current codes are set by default. Hence, user can ajust the parameters such as task deadline, task arrival probabiltiy, computing capacity, and the number of tasks, and then run the main.py to get more experimental results.

Paramters setting information: 
1) Deadline setting. User can adjust the variable value of DEADLINE in main.py. For example, the DEADLINE can be set from 0.1 to 1.0 seconds and then run main.py to achieve the corresponding experimental results.
2) Task arrival probabiltiy setting. User can adjust the variable value (i.e., from 0.1 to 1) of TASK_ARRIVAL_PROB in main.py, thus runing main.py to achieve the corresponding experimental results.
3) The number setting of tasks. User can adjust the variable value (i.e., from 10 to 100) of NUM_TASK in main.py, thus runing main.py to achieve the corresponding experimental results.
4) The setting of ESs' computing capacities. If user want to achieve the results varying different ESs' computing capacityes, user can adjust the variable value of BS_capacities in main.py and then run the main.py to achieve the corresponding experimental results.

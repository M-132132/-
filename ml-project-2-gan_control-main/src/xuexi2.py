import argparse
import time
import json
import os

# Packages
import torch
from casadi import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Classes and helpers
from vehicleModelGarage import vehBicycleKinematic
from scenarios import trailing, simpleOvertake
from traffic import vehicleSUMO, combinedTraffic
from controllers import makeController, makeDecisionMaster
from helpers import *
from agents.rlagent import DQNAgent

np.random.seed(1)

# Constants
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_node_features = 4
n_actions = 3

# Parse command line arguments
ap = argparse.ArgumentParser(description='Train DeepQN RL agent for lane changing decisions')

ap.add_argument("-H", "--hyperparam_config", required=True,
                help=("Path to json file containing the hyperparameters for " +
                      "training the GNN. Must define the following " +
                      "hyperparameters: gamma (float), target_copy_delay " +
                      "(int), learning_rate (float), batch_size (int), " +
                      "epsilon (float), epsilon_dec (float), epsilon_min " +
                      "(float), memory_size (int)"))

ap.add_argument("-l", "--log_dir", default='../out/runs',
                help=("Directory in which to store logs. Default ../out/runs"))

ap.add_argument("-e", "--num_episodes", type=int, default=100,
                help=("Number of episodes to run simulation for. Default 100"))

ap.add_argument("-d", "--max_dist", type=int, default=10000,
                help=("Goal distance for vehicle to travel. Simulation " +
                      "terminates if this is reached. Default 10000m"))

ap.add_argument("-t", "--time_step", type=float, default=0.2,
                help=("Simulation time step. Default 0.2s"))

ap.add_argument("-f", "--controller_frequency", type=int, default=5,
                help=("Controller update frequency. Updates at each f " +
                      "timesteps. Default 5"))

ap.add_argument("-s", "--speed_limit", type=int, default=60,
                help=("Highway speed limit (km/h). Default 60"))

ap.add_argument("-N", "--horizon_length", type=int, default=12,
                help=("MPC horizon length. Default 12 "))

ap.add_argument("-T", "--simulation_time", type=int, default=100,
                help=("Maximum total simulation time (s). Default 100"))

ap.add_argument("-E", "--exp_id", default='',
                help=("Optional ID for the experiment"))

args = ap.parse_args()

# validate and parse hyperparameter configuration file
hp_file = args.hyperparam_config
assert os.path.exists(hp_file), ("ERROR: hyperparam config file not found.")
try:
    with open(hp_file, 'r', encoding='utf-8') as f:
        hps = json.load(f)
except json.JSONDecodeError:
    sys.exit("Hyperparameter config file is not a valid json file.")

REQ_HYPERPARAMS = {"gamma": "float", "target_copy_delay": "int",
                   "learning_rate": "float", "batch_size": "int",
                   "epsilon": "float", "epsilon_dec": "float",
                   "epsilon_min": "float", "memory_size": "int"}
hyperparams = {}
for hp in REQ_HYPERPARAMS.keys():
    assert (hp in hps), hp + " hyperparameter missing from config file"
    hyperparams[hp] = hps[hp]

# configure logging
log_dir = args.log_dir
assert os.path.exists(log_dir), "Invalid log directory (must already exist)"

exp_id = args.exp_id + '_' + str(time.time())

exp_log_dir = os.path.join(log_dir, exp_id)
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)
    print(f"Created experiment log directory at {exp_log_dir}")

tensorboard_log_dir = os.path.join(exp_log_dir, 'tensorboard_logs')
if not os.path.exists(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)
    print(f"Created tensorboard log directory at {tensorboard_log_dir}")

args.hyperparam_config = hyperparams

# System initialization
dt = args.time_step
f_controller = args.controller_frequency
N = args.horizon_length
ref_vx = args.speed_limit / 3.6
tsim = args.simulation_time
N_sim = int(tsim/dt)
N_episodes = args.num_episodes
dist_max = args.max_dist
N_all = N_sim * N_episodes

# Initialize RL agent object
RL_Agent = DQNAgent(device, num_node_features, n_actions, ref_vx, **hyperparams)

# Set logging
writer = SummaryWriter(log_dir=tensorboard_log_dir)

param_log_file = os.path.join(exp_log_dir, 'experiment_parameters.json')
with open(param_log_file, 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=4)
print(f"Saved input parameter log file to {param_log_file}")

# Ego Vehicle Dynamics and Controller Settings
vehicleADV = vehBicycleKinematic(dt, N)

vehWidth, vehLength, L_tract, L_trail = vehicleADV.getSize()
nx, nu, nrefx, nrefu = vehicleADV.getSystemDim()

# Integrator
int_opt = 'rk'
vehicleADV.integrator(int_opt, dt)
F_x_ADV = vehicleADV.getIntegrator()

# Set Cost parameters
Q_ADV = [0, 40, 3e2, 5, 5]
R_ADV = [5, 5]
q_ADV_decision = 50

vehicleADV.cost(Q_ADV, R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV, Lf_ADV = vehicleADV.getCost()

# Problem definition
scenarioTrailADV = trailing(vehicleADV, N, lanes=3)
scenarioADV = simpleOvertake(vehicleADV, N, lanes=3)
roadMin, roadMax, laneCenters = scenarioADV.getRoad()

# Traffic Set up
vx_init_ego = 55 / 3.6
vehicleADV.setInit([0, laneCenters[0]], vx_init_ego)

vx_ref_init = 50 / 3.6
advVeh1 = vehicleSUMO(dt, N, [30, laneCenters[1]], [0.9 * vx_ref_init, 0], type="normal")
advVeh2 = vehicleSUMO(dt, N, [45, laneCenters[0]], [0.8 * vx_ref_init, 0], type="passive")
advVeh3 = vehicleSUMO(dt, N, [100, laneCenters[2]], [0.85 * vx_ref_init, 0], type="normal")
advVeh4 = vehicleSUMO(dt, N, [-20, laneCenters[1]], [1.2 * vx_ref_init, 0], type="aggressive")
advVeh5 = vehicleSUMO(dt, N, [40, laneCenters[2]], [1.2 * vx_ref_init, 0], type="aggressive")

vehList = [advVeh1, advVeh2, advVeh3, advVeh4, advVeh5]

leadWidth, leadLength = advVeh1.getSize()
traffic = combinedTraffic(vehList, vehicleADV, ref_vx, f_controller)
traffic.setScenario(scenarioADV)
car_N = traffic.getDim()

# Formulate optimal control problem
dt_MPC = dt * f_controller
opts1 = {"version": "leftChange", "solver": "ipopt", "integrator": "rk"}
MPC1 = makeController(vehicleADV, traffic, scenarioADV, N, opts1, dt_MPC)
MPC1.setController()
changeLeft = MPC1.getFunction()

opts2 = {"version": "rightChange", "solver": "ipopt", "integrator": "rk"}
MPC2 = makeController(vehicleADV, traffic, scenarioADV, N, opts2, dt_MPC)
MPC2.setController()
changeRight = MPC2.getFunction()

opts3 = {"version": "trailing", "solver": "ipopt", "integrator": "rk"}
MPC3 = makeController(vehicleADV, traffic, scenarioTrailADV, N, opts3, dt_MPC)
MPC3.setController()
trailLead = MPC3.getFunction()

print("Initialization successful.")

decisionMaster = makeDecisionMaster(vehicleADV, traffic, [MPC1, MPC2, MPC3],
                                    [scenarioTrailADV, scenarioADV], RL_Agent)

decisionMaster.setDecisionCost(q_ADV_decision)
##
RL_feature_map = np.zeros((6, N_sim, car_N + 1))   # 特征映射（如用于RL的状态表示）
traffic_real = np.zeros((4, N_all, car_N))
traffic_ref = np.zeros((4, N_all, car_N))
traffic_real[:,0,:] = traffic.getStates()
traffic_lead = DM(car_N,N+1)               #前车状态
traffic_pre = np.zeros((5,N+1,car_N))             #环境其他车辆的预测时域内的状态


N_episodes = args.num_episodes
x_xunlian = DM(int(nx), 1)
x_xunlian[:], u_xunlian = vehicleADV.getInit()
vehicleADV.update(x_xunlian, u_xunlian)
episodes_i = 0
i_overall = 0
# Training loop
for j in range(N_episodes):
    # 初始化环境和状态
    x_iter, u_iter = vehicleADV.getInit()
    vehicleADV.update(x_iter, u_iter)
    traffic.reset()

    i = i_overall
    done = False
    while not done:

        RL_feature_map_i = createFeatureMatrix(vehicleADV, traffic)  # 创建当前状态特征
        RL_feature_map[:, i:] = RL_feature_map_i
        traffic_lead[:, :] = traffic.prediction()[0, :, :].transpose()
        traffic_pre[:2, :, ] = traffic.prediction()[:2, :, :]
        # 记录车辆速度
        writer.add_scalars('Overall/Velocity', {'Velocity': RL_feature_map_i[2][0][0],
                                                'Maximum Velocity': ref_vx}, i_overall)

        writer.add_scalars('Episode_' + str(j) + '/Velocity', {'Velocity': RL_feature_map_i[2][0][0],
                                                               'Maximum Velocity': ref_vx}, episodes_i)
        if (i - episodes_i) % f_controller == 0:
            decisionMaster.storeInput(
                [x_xunlian, refx_left_out, refx_right_out, refx_trail_out, refu_out, traffic_lead, traffic_pre])
            # 可视化当前场景
            if arg.display_simulation:
                fig = plotScene(RL_feature_map_i, scenarioADV, vehicleADV, car_list)
                plt.show()
                plt.pause(0.02)
            # 存储经验到RL回放缓冲区
            if previous_state is not None:
                reward = reward / f_controller
                reward_all = reward_all + reward
                RL_Agent.store_transition(previous_state, action, reward, RL_feature_map_i, terminal_state=False)
            # 更新参考轨迹
            refx_left_out, refx_right_out, refx_trail_out = decisionMaster.updateReference()
            writer.add_scalar('Episode_' + str(j) + '/Rewards', reward, reward_all)
            # RL代理学习
            RL_Agent.learn()
            x_test, u_test, X_out, selected_action = decisionMaster.chooseController(RL_feature_map_i)
            u_xunlian = u_test[:, 0]  # 获取控制输入（如加速度、转向角）
            previous_state, action, reward = RL_feature_map_i, selected_action, 0
        reward += RL_feature_map_i[2][0][0] / ref_vx
        # 存储当前的状态和控制输入


    print(f"Episode {episode + 1} completed.")

# 保存模型
model_path = os.path.join(exp_log_dir, 'final_model.pt')
RL_Agent.save_model(file_path=model_path)
print(f"Model saved to {model_path}")
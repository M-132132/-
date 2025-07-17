import argparse
import time
import json
import os

import matplotlib.pyplot as plt
import torch
from casadi import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from vehicleModelGarage import vehBicycleKinematic
from scenarios import trailing, simpleOvertake
from traffic import vehicleSUMO, combinedTraffic
from controllers import makeController, makeDecisionMaster
from helpers import *
from agents.rlagent import DQNAgent

np.random.seed(1)

device = torch.device('cpu')
num_node_features = 4
n_actions = 3
makeMovie = False
directory = "sim_render"

#——————命令行参数解析——————
ap = argparse.ArgumentParser(description='训练DQN智能体')
#ap.add_argument("-H","--hyperparam",required=True)
ap.add_argument("-l", "--log_dir", default='../out/runs')
ap.add_argument("-e", "--num_episodes", type=int, default=100)     #强化学习中的训练轮数，训练完所有的轮数完成一次仿真步数
ap.add_argument("-d", "--max_dist", type=int, default=10000)
ap.add_argument("-t", "--time_step", type=float, default=0.2)
ap.add_argument("-f", "--controller_frequency", type=int, default=5)
ap.add_argument("-s", "--speed_limit", type=int, default=60)
ap.add_argument("-N", "--horizon_length", type=int, default=12)
ap.add_argument("-T", "--simulation_time", type=int, default=100)
ap.add_argument("-E", "--exp_id", default='')
ap.add_argument("--display_simulation",action="store_true")
arg = ap.parse_args()
print("xinxi",arg.num_episodes)

#——————实验日志目录创建——————
log = arg.log_dir
assert os.path.isdir(log), log + " is not a valid directory"
expid = arg.exp_id+ str(time.time())
exp_dir = os.path.join(log, expid)
os.makedirs(exp_dir)
assert os.path.exists(exp_dir), exp_dir + (" is not a valid")

#——————tensorboard目录创建——————
tensorboard_dir = os.path.join(log, expid,'tensorboard_logs')
os.makedirs(tensorboard_dir)
assert os.path.exists(tensorboard_dir), tensorboard_dir + " is not a valid directory"

log_dir = arg.log_dir
exp_log_dir = os.path.join(log_dir, arg.exp_id)
model_params_path = os.path.join(exp_log_dir, 'final_model.pt')
assert os.path.exists(model_params_path), "No model parameters found"
#——————系统设置——————
dt = arg.time_step  # Simulation time step (Impacts traffic model accuracy)
f_controller = arg.controller_frequency  # Controller update frequency, i.e updates at each t = dt*f_controller
N = arg.horizon_length  # MPC Horizon length
ref_vx = arg.speed_limit / 3.6  # Highway speed limit in (m/s)
tsim = arg.simulation_time  # Maximum total simulation time in seconds
N_episodes = arg.num_episodes  # Number of scenarios run created
dist_max = arg.max_dist  # Goal distance for the vehicle to travel. If reached, epsiode terminates
RL_Agent = DQNAgent(device, num_node_features, n_actions, ref_vx, epsilon=0,
                    memory_size=0)
RL_Agent.load_model(file_path=model_params_path)


#——————tensorboard写入——————
writer = SummaryWriter(log_dir=tensorboard_dir)
tensorboard_file = os.path.join(exp_dir, 'experiment_parameters.json')
with open(tensorboard_file , 'w', encoding='utf-8') as f:
    json.dump(vars(arg), f, ensure_ascii=False, indent=4)
print(f"Saved input parameter log file to {tensorboard_file}")

#——————车辆创建——————
vehicleADV = vehBicycleKinematic(dt, N)
vehWidth, vehLength, L_tract, L_trail = vehicleADV.getSize()
nx, nu, nrefx, nrefu = vehicleADV.getSystemDim()
#——————积分设置——————
int_opt = 'rk'
vehicleADV.integrator(int_opt, dt)
F_x_ADV = vehicleADV.getIntegrator()

#——————成本函数——————
Q_ADV = [0, 40, 3e2, 5, 5]
R_ADV = [5, 5]
q_ADV_decision = 50
vehicleADV.cost(Q_ADV, R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV, Lf_ADV = vehicleADV.getCost()

#——————场景定义——————
scenarioTrailADV = trailing(vehicleADV, N, lanes=3)
scenarioADV = simpleOvertake(vehicleADV, N, lanes=3)
roadMin, roadMax, laneCenters = scenarioADV.getRoad()
#——————场景数据定义——————
vx_ego = 60 / 3.6
vehicleADV.setInit([0, laneCenters[0]], vx_ego)
#——————环境车辆定义——————
vx_env = 50 / 3.6  # (m/s)
advVeh1 = vehicleSUMO(dt, N, [30, laneCenters[1]], [0.9 * vx_env, 0], type="normal")
advVeh2 = vehicleSUMO(dt, N, [45, laneCenters[0]], [0.8 * vx_env, 0], type="passive")
advVeh3 = vehicleSUMO(dt, N, [100, laneCenters[2]], [0.85 * vx_env, 0], type="normal")
advVeh4 = vehicleSUMO(dt, N, [-20, laneCenters[1]], [1.4 * vx_env, 0], type="aggressive")
advVeh5 = vehicleSUMO(dt, N, [40, laneCenters[2]], [1.6 * vx_env, 0], type="aggressive")
car_list = [advVeh1, advVeh2, advVeh3, advVeh4, advVeh5]

#——————交通场景定义——————
leadWidth, leadLength = advVeh1.getSize()
traffic = combinedTraffic(car_list, vehicleADV, ref_vx, f_controller)
traffic.setScenario(scenarioADV)
car_N = traffic.getDim()

#——————控制器定义——————
dt_MPC = dt * f_controller
# MPC1.testSolver(traffic)x
opts1 = {"version": "leftChange", "solver": "ipopt", "integrator": "rk"}
MPC1 = makeController(vehicleADV, traffic, scenarioADV, N, opts1, dt_MPC)
MPC1.setController()
changeLeft = MPC1.getFunction()
# MPC2.testSolver(traffic)
opts2 = {"version": "rightChange", "solver": "ipopt", "integrator": "rk"}
MPC2 = makeController(vehicleADV, traffic, scenarioADV, N, opts2,dt_MPC)
MPC2.setController()
changeRight = MPC2.getFunction()
# MPC3.testSolver(traffic)
opts3 = {"version": "trailing", "solver": "ipopt", "integrator": "rk"}
MPC3 = makeController(vehicleADV, traffic, scenarioTrailADV, N, opts3, dt_MPC)
MPC3.setController()
trailLead = MPC3.getFunction()
print("load controler succesful.")
#——————主决策初始化——————
decisionMaster = makeDecisionMaster(vehicleADV, traffic, [MPC1, MPC2, MPC3],
                                    [scenarioTrailADV, scenarioADV], RL_Agent)
decisionMaster.setDecisionCost(q_ADV_decision)                                             # Sets cost of changing decision

#——————参考轨迹和控制输入——————
N_sim = int(tsim/dt)
t_list = np.arange(0, tsim, dt)
refxADV = [0,laneCenters[1],ref_vx,0,0]
refx_trail_in,refx_left_in,refx_right_in = vehicleADV.setReferences(laneCenters)
refu_in = [0,0,0]
refx_trail_out, refu_out = scenarioADV.getReference(refx_trail_in, refu_in)
refx_left_out, refu_out = scenarioADV.getReference(refx_left_in, refu_in)
refx_right_out, refu_out = scenarioADV.getReference(refx_right_in, refu_in)
refxADV_out, refuADV_out = scenarioADV.getReference(refxADV, refu_in)

#——————数据存储——————
N_all = N_sim * N_episodes
X = np.zeros((nx, N_all, 1))  #主车辆状态矩阵
U = np.zeros((nu, N_all, 1))  #主车辆控制输入矩阵
X_pre = np.zeros((nx,N+1,N_all))  #主车辆预测时域内状态
#——交通车辆——
traffic_real = np.zeros((4, N_all, car_N))
traffic_ref = np.zeros((4, N_all, car_N))
traffic_real[:,0,:] = traffic.getStates()
traffic_lead = DM(car_N,N+1)               #前车状态
traffic_pre = np.zeros((5,N+1,car_N))             #环境其他车辆的预测时域内的状态
#
RL_feature_map = np.zeros((6, N_sim, car_N + 1))   # 特征映射（如用于RL的状态表示）
i_keything = 0                                    # 关键事件计数器（如碰撞次数）
traffic.reset()                               # 重置交通环境
i_overall = 0                             # 总迭代次数计数器
#——————可视化交互——————
import matplotlib
matplotlib.use('qt5Agg')
matplotlib.pyplot.ion()# 开启交互模式(Interactive Mode) 允许在不调用plt.show()的情况下实时显示图形

#——————训练轮次迭代——————
for j in range(0,N_episodes):
    plt.close('all')
    print(f"episode : {j+1}")
    #仿真初始化
    x_xunlian = DM(int(nx),1)
    x_xunlian[:],u_xunlian =vehicleADV.getInit()
    vehicleADV.update(x_xunlian,u_xunlian)
    episodes_i = 0
    #仿真环境设置
    i = i_overall
    runSimulation = True
    previous_state = None
    action = None
    reward = 0
    reward_all = 0
#——————仿真循环——————
    while runSimulation :
        RL_feature_map_i = createFeatureMatrix(vehicleADV, traffic)  # 创建当前状态特征
        RL_feature_map[:, i:] = RL_feature_map_i
        traffic_lead[:,:] = traffic.prediction()[0,:,:].transpose()
        traffic_pre[:2,:,] = traffic.prediction()[:2,:,:]
        #记录车辆速度
        writer.add_scalars('Overall/Velocity', {'Velocity': RL_feature_map_i[2][0][0],
                                                'Maximum Velocity': ref_vx}, i_overall)

        writer.add_scalars('Episode_' + str(j) + '/Velocity', {'Velocity': RL_feature_map_i[2][0][0],
                                                               'Maximum Velocity': ref_vx}, episodes_i)
        if (i-episodes_i) % f_controller == 0 :
            decisionMaster.storeInput([x_xunlian, refx_left_out, refx_right_out, refx_trail_out, refu_out, traffic_lead, traffic_pre])
            #可视化当前场景
            if arg.display_simulation :
                fig = plotScene(RL_feature_map_i, scenarioADV, vehicleADV, car_list)
                plt.show()
                plt.pause(0.02)
            #存储经验到RL回放缓冲区
            if previous_state is not None :
                reward = reward / f_controller
            #更新参考轨迹
            refx_left_out, refx_right_out, refx_trail_out = decisionMaster.updateReference()
            writer.add_scalar('Episode_' + str(j) + '/Rewards', reward, reward_all)
            #RL代理学习

            x_test, u_test, X_out, selected_action = decisionMaster.chooseController(RL_feature_map_i)
            u_xunlian = u_test[:, 0]  # 获取控制输入（如加速度、转向角）
            previous_state, action, reward = RL_feature_map_i, selected_action, 0
        reward += RL_feature_map_i[2][0][0] / ref_vx
        # 存储当前的状态和控制输入
        X[:, i] = x_xunlian
        U[:, i] = u_xunlian
        X_pre[:, :, i] = X_out  #将选择的决策的输出赋给主车预测
        x_xunlian = F_x_ADV(x_xunlian, u_xunlian)
        #更新交通环境和车辆状态
        try:
            traffic.update()
            vehicleADV.update(x_xunlian,u_xunlian)
        except:
            print('仿真完成：发生碰撞')
            num_crashes += 1
            writer.add_scalar('Overall/Crash_1_Derail_2', 1, overall_iters)
            break
        #终止条件
        if (i-i_overall ==N_sim - 1) :
            runSimulation = False
            plt.close('all')
            print('仿真终止：达到最大仿真步数')
        elif (x_xunlian[0].full().item() > dist_max) :
            runSimulation = False
            plt.close('all')
            print('仿真终止：达到最大距离')
        elif(x_xunlian[1].full().item() > roadMax) or (x_xunlian[1].full().item() < roadMin) :
            runSimulation = False
            plt.close('all')
            print('仿真终止：达到道路边界')
        traffic.tryRespawn(x_xunlian[0])
        traffic_real[:, i, :] = traffic.getStates()
        traffic_ref[:, i, :] = traffic.getReference()
        episodes_i += 1
        i_overall += 1

    writer.add_scalar('Overall/Average_Rewards', reward_all / episodes_i, i_overall)
    writer.add_scalar('Overall/Episode_Iterations', episodes_i, j)
    writer.add_scalar('Overall/Episode_Distances', x_xunlian[0].full().item(), j)
    #total_distance += x_iter[0].full().item()
    i_overall = i

    traffic.reset()
    traffic_real[:, i, :] = traffic.getStates()
    traffic_ref[:, i, :] = traffic.getReference()

print("仿真结束")
RL_Agent.save_model(os.path.join(exp_dir, 'final_model.pt'))




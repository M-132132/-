import argparse
import time
import json
import os
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
import matplotlib

# 全局常量
np.random.seed(1)
device = torch.device('cpu')
num_node_features = 4
n_actions = 3
makeMovie = False
directory = "sim_render"


def parse_arguments():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(description='训练DQN智能体')
    ap.add_argument("-H", "--hyperparam", required=True)
    ap.add_argument("-l", "--log_dir", default='../out/runs')
    ap.add_argument("-e", "--num_episodes", type=int, default=100)
    ap.add_argument("-d", "--max_dist", type=int, default=10000)
    ap.add_argument("-t", "--time_step", type=float, default=0.2)
    ap.add_argument("-f", "--controller_frequency", type=int, default=5)
    ap.add_argument("-s", "--speed_limit", type=int, default=60)
    ap.add_argument("-N", "--horizon_length", type=int, default=12)
    ap.add_argument("-T", "--simulation_time", type=int, default=100)
    ap.add_argument("-E", "--exp_id", default='')
    return ap.parse_args()


def load_hyperparams(json_file):
    """加载超参数"""
    REQ_HYPERPARAMS = {
        "gamma": "float", "target_copy_delay": "int",
        "learning_rate": "float", "batch_size": "int",
        "epsilon": "float", "epsilon_dec": "float",
        "epsilon_min": "float", "memory_size": "int"
    }
    try:
        with open(json_file, "r", encoding='utf-8') as f:
            j_file = json.load(f)
    except FileNotFoundError:
        print("Hyperparameter config file is not a valid json file.")
        raise
    hyperparams = {}
    for hp in REQ_HYPERPARAMS.keys():
        assert (hp in j_file), hp + " hyperparameter missing from config file"
        hyperparams[hp] = j_file[hp]
    return hyperparams


def setup_directories(log_dir, exp_id):
    """设置实验目录"""
    expid = exp_id + str(time.time())
    exp_dir = os.path.join(log_dir, expid)
    os.makedirs(exp_dir)
    tensorboard_dir = os.path.join(log_dir, expid, 'tensorboard_logs')
    os.makedirs(tensorboard_dir)
    return exp_dir, tensorboard_dir


def initialize_vehicle(dt, N):
    """初始化主车辆"""
    vehicleADV = vehBicycleKinematic(dt, N)
    vehWidth, vehLength, L_tract, L_trail = vehicleADV.getSize()
    nx, nu, nrefx, nrefu = vehicleADV.getSystemDim()

    # 积分设置
    int_opt = 'rk'
    vehicleADV.integrator(int_opt, dt)
    F_x_ADV = vehicleADV.getIntegrator()

    # 成本函数
    Q_ADV = [0, 40, 3e2, 5, 5]
    R_ADV = [5, 5]
    q_ADV_decision = 50
    vehicleADV.cost(Q_ADV, R_ADV)
    vehicleADV.costf(Q_ADV)
    L_ADV, Lf_ADV = vehicleADV.getCost()

    return vehicleADV, F_x_ADV, q_ADV_decision


def initialize_scenarios(vehicleADV, N):
    """初始化场景"""
    scenarioTrailADV = trailing(vehicleADV, N, lanes=3)
    scenarioADV = simpleOvertake(vehicleADV, N, lanes=3)
    roadMin, roadMax, laneCenters = scenarioADV.getRoad()
    return scenarioADV, scenarioTrailADV, roadMin, roadMax, laneCenters


def initialize_traffic_vehicles(dt, N, laneCenters, ref_vx, f_controller, vehicleADV, scenarioADV):
    """初始化交通车辆"""
    vx_env = 50 / 3.6  # (m/s)
    advVeh1 = vehicleSUMO(dt, N, [30, laneCenters[1]], [0.9 * vx_env, 0], type="normal")
    advVeh2 = vehicleSUMO(dt, N, [45, laneCenters[0]], [0.8 * vx_env, 0], type="passive")
    advVeh3 = vehicleSUMO(dt, N, [100, laneCenters[2]], [0.85 * vx_env, 0], type="normal")
    advVeh4 = vehicleSUMO(dt, N, [-20, laneCenters[1]], [1.4 * vx_env, 0], type="aggressive")
    advVeh5 = vehicleSUMO(dt, N, [40, laneCenters[2]], [1.6 * vx_env, 0], type="aggressive")
    car_list = [advVeh1, advVeh2, advVeh3, advVeh4, advVeh5]

    leadWidth, leadLength = advVeh1.getSize()
    traffic = combinedTraffic(car_list, vehicleADV, ref_vx, f_controller)
    traffic.setScenario(scenarioADV)
    car_N = traffic.getDim()

    return traffic, car_list, car_N


def initialize_controllers(vehicleADV, traffic, scenarioADV, scenarioTrailADV, N, dt_MPC):
    """初始化控制器"""
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

    return MPC1, MPC2, MPC3


def initialize_references(vehicleADV, scenarioADV, laneCenters, ref_vx):
    """初始化参考轨迹"""
    refxADV = [0, laneCenters[1], ref_vx, 0, 0]
    refx_trail_in, refx_left_in, refx_right_in = vehicleADV.setReferences(laneCenters)
    refu_in = [0, 0, 0]
    refx_trail_out, refu_out = scenarioADV.getReference(refx_trail_in, refu_in)
    refx_left_out, refu_out = scenarioADV.getReference(refx_left_in, refu_in)
    refx_right_out, refu_out = scenarioADV.getReference(refx_right_in, refu_in)
    refxADV_out, refuADV_out = scenarioADV.getReference(refxADV, refu_in)

    return refx_left_out, refx_right_out, refx_trail_out, refu_out


def initialize_data_structures(nx, nu, car_N, N_sim, N_episodes, N):
    """初始化数据结构"""
    N_all = N_sim * N_episodes
    X = np.zeros((nx, N_all, 1))
    U = np.zeros((nu, N_all, 1))
    X_pre = np.zeros((nx, N + 1, N_all))
    traffic_real = np.zeros((4, N_all, car_N))
    traffic_ref = np.zeros((4, N_all, car_N))
    RL_feature_map = np.zeros((6, N_sim, car_N + 1))
    traffic_lead = DM(car_N, N + 1)
    traffic_pre = np.zeros((5, N + 1, car_N))

    return X, U, X_pre, traffic_real, traffic_ref, RL_feature_map, traffic_lead, traffic_pre


def main():
    args = parse_arguments()  # 解析参数
    print("开始训练，总轮次:", args.num_episodes)
    hyperparams = load_hyperparams(args.hyperparam)  # 加载超参数
    exp_dir, tensorboard_dir = setup_directories(args.log_dir, args.exp_id)  # 设置目录

    # 系统设置
    dt = args.time_step
    f_controller = args.controller_frequency
    N = args.horizon_length
    ref_vx = args.speed_limit / 3.6
    tsim = args.simulation_time
    N_episodes = args.num_episodes
    dist_max = args.max_dist
    N_sim = int(tsim / dt)

    # 初始化RL Agent
    RL_Agent = DQNAgent(device, num_node_features, n_actions, ref_vx, **hyperparams)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # 保存实验参数
    tensorboard_file = os.path.join(exp_dir, 'experiment_parameters.json')
    with open(tensorboard_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    print(f"参数文件保存到 {tensorboard_file}")

    # 初始化车辆
    vehicleADV, F_x_ADV, q_ADV_decision = initialize_vehicle(dt, N)

    # 初始化场景
    scenarioADV, scenarioTrailADV, roadMin, roadMax, laneCenters = initialize_scenarios(vehicleADV, N)

    # 设置车辆初始位置
    vx_ego = 60 / 3.6
    vehicleADV.setInit([0, laneCenters[0]], vx_ego)

    # 初始化交通车辆
    traffic, car_list, car_N = initialize_traffic_vehicles(dt, N, laneCenters, ref_vx, f_controller, vehicleADV,
                                                           scenarioADV)

    # 初始化控制器
    dt_MPC = dt * f_controller
    MPC1, MPC2, MPC3 = initialize_controllers(vehicleADV, traffic, scenarioADV, scenarioTrailADV, N, dt_MPC)
    print("控制器加载成功")

    # 初始化决策主控
    decisionMaster = makeDecisionMaster(vehicleADV, traffic, [MPC1, MPC2, MPC3],
                                        [scenarioTrailADV, scenarioADV], RL_Agent)
    decisionMaster.setDecisionCost(q_ADV_decision)

    # 初始化参考轨迹
    refx_left_out, refx_right_out, refx_trail_out, refu_out = initialize_references(
        vehicleADV, scenarioADV, laneCenters, ref_vx)
    # 初始化数据结构
    X, U, X_pre, traffic_real, traffic_ref, RL_feature_map, traffic_lead, traffic_pre = initialize_data_structures(
        vehicleADV.nx, vehicleADV.nu, car_N, N_sim, N_episodes, N)

    # 训练轮次迭代
    i_overall = 0
    for j in range(N_episodes):
        plt.close('all')
        print(f"训练轮次: {j + 1}/{N_episodes}")

        # 仿真初始化
        x_current = DM(int(vehicleADV.nx), 1)
        x_current[:], u_current = vehicleADV.getInit()
        vehicleADV.update(x_current, u_current)
        episodes_i = 0
        i = i_overall
        runSimulation = True
        previous_state = None
        action = None
        reward = 0
        reward_all = 0

        # 仿真循环
        while runSimulation:
            RL_feature_map_i = createFeatureMatrix(vehicleADV, traffic)  # 创建当前状态特征
            traffic_lead[:, :] = traffic.prediction()[0, :, :].transpose()
            traffic_pre[:2, :, ] = traffic.prediction()[:2, :, :]

            # 记录车辆速度
            writer.add_scalars('Overall/Velocity', {'Velocity': RL_feature_map_i[2][0][0],
                                                    'Maximum Velocity': ref_vx}, i_overall)
            writer.add_scalars('Episode_' + str(j) + '/Velocity', {'Velocity': RL_feature_map_i[2][0][0],
                                                                   'Maximum Velocity': ref_vx}, episodes_i)

            if (i - episodes_i) % f_controller == 0:
                decisionMaster.storeInput(
                    [x_current, refx_left_out, refx_right_out, refx_trail_out, refu_out, traffic_lead, traffic_pre])


                # 存储经验到RL回放缓冲区
                if previous_state is not None:
                    reward = reward / f_controller
                    reward_all = reward_all + reward
                    RL_Agent.store_transition(previous_state, action, reward, RL_feature_map_i, terminal_state=False)

                # 更新参考轨迹
                refx_left_out, refx_right_out, refx_trail_out = decisionMaster.updateReference()
                writer.add_scalar('Episode_' + str(j) + '/Rewards', reward, reward_all)

                # RL代理学习
                if RL_Agent.replay_buffer.mem_counter > hyperparams["batch_size"]:
                    loss = RL_Agent.learn()
                    #writer.add_scalar('Training/loss', loss, i_overall)
                    # 添加损失值检查
                    if loss is not None:
                        writer.add_scalar('Training/loss', loss, i_overall)
                    else:
                        print("警告：learn()方法返回了None损失值")

                # 选择控制器
                x_test, u_test, X_out, selected_action = decisionMaster.chooseController(RL_feature_map_i)
                u_current = u_test[:, 0]  # 获取控制输入
                previous_state, action, reward = RL_feature_map_i, selected_action, 0

            reward += RL_feature_map_i[2][0][0] / ref_vx

            # 状态转移
            x_current = F_x_ADV(x_current, u_current)

            # 更新交通环境和车辆状态
            try:
                traffic.update()
                vehicleADV.update(x_current, u_current)
            except:
                print('仿真完成：发生碰撞')
                buffer = RL_Agent.replay_buffer
                idx = buffer.mem_counter % buffer.mem_size - 1
                buffer.reward_mem[idx] = -10
                break

            # 终止条件
            if (i - i_overall == N_sim - 1):
                runSimulation = False
                plt.close('all')
                print('仿真终止：达到最大仿真步数')
            elif (x_current[0].full().item() > dist_max):
                runSimulation = False
                plt.close('all')
                print('仿真终止：达到最大距离')
            elif (x_current[1].full().item() > roadMax) or (x_current[1].full().item() < roadMin):
                runSimulation = False
                plt.close('all')
                print('仿真终止：达到道路边界')

            # 交通车重生检查
            traffic.tryRespawn(x_current[0])

            episodes_i += 1
            i_overall += 1

        # 记录训练指标
        writer.add_scalar('Overall/Average_Rewards', reward_all / episodes_i, i_overall)
        writer.add_scalar('Overall/Episode_Iterations', episodes_i, j)
        writer.add_scalar('Overall/Episode_Distances', x_current[0].full().item(), j)

        # 保存模型检查点
        if (j + 1) % 10 == 0 or j == N_episodes - 1:
            RL_Agent.save_model(os.path.join(exp_dir, f'model_checkpoint_{j + 1}.pt'))
            print(
                f"轮次 {j + 1} | 平均奖励: {reward_all / episodes_i:.2f} | 步数: {episodes_i} | ε: {RL_Agent.epsilon:.3f}")

    # 最终保存
    RL_Agent.save_model(os.path.join(exp_dir, 'final_model.pt'))
    writer.close()
    print(f"训练完成，模型已保存到: {exp_dir}")








if __name__ == "__main__":
    main()
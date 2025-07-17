import argparse
import time
import json
import os
import matplotlib
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

# 全局常量
np.random.seed(1)
device = torch.device('cpu')
num_node_features = 4
n_actions = 3
makeMovie = False
directory = "sim_render"


def parse_arguments():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(description='使用训练好的模型进行仿真实验')
    ap.add_argument("-l", "--log_dir", default='../out/runs')
    ap.add_argument("-e", "--num_episodes", type=int, default=100)
    ap.add_argument("-d", "--max_dist", type=int, default=10000)
    ap.add_argument("-t", "--time_step", type=float, default=0.2)
    ap.add_argument("-f", "--controller_frequency", type=int, default=5)
    ap.add_argument("-s", "--speed_limit", type=int, default=60)
    ap.add_argument("-N", "--horizon_length", type=int, default=12)
    ap.add_argument("-T", "--simulation_time", type=int, default=100)
    ap.add_argument("-E", "--exp_id", default='')
    ap.add_argument("--display_simulation", action="store_true")
    return ap.parse_args()


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
    print("仿真参数", args.num_episodes)

    # 设置实验目录
    exp_dir, tensorboard_dir = setup_directories(args.log_dir, args.exp_id)

    # 系统设置
    dt = args.time_step
    f_controller = args.controller_frequency
    N = args.horizon_length
    ref_vx = args.speed_limit / 3.6
    tsim = args.simulation_time
    N_episodes = args.num_episodes
    dist_max = args.max_dist
    N_sim = int(tsim / dt)

    # 初始化RL Agent (不训练，仅推理)
    RL_Agent = DQNAgent(device, num_node_features, n_actions, ref_vx, epsilon=0, memory_size=0)

    # 加载预训练模型
    model_params_path = os.path.join(args.log_dir, args.exp_id, 'final_model.pt')
    assert os.path.exists(model_params_path), "找不到预训练模型文件"
    RL_Agent.load_model(file_path=model_params_path)

    # 初始化TensorBoard (仅记录不训练)
    writer = SummaryWriter(log_dir=tensorboard_dir)
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

    # 初始化决策主控 (使用预训练模型)
    decisionMaster = makeDecisionMaster(vehicleADV, traffic, [MPC1, MPC2, MPC3],
                                        [scenarioTrailADV, scenarioADV], RL_Agent)
    decisionMaster.setDecisionCost(q_ADV_decision)

    # 初始化参考轨迹
    refx_left_out, refx_right_out, refx_trail_out, refu_out = initialize_references(
        vehicleADV, scenarioADV, laneCenters, ref_vx)

    # 初始化数据结构
    X, U, X_pre, traffic_real, traffic_ref, RL_feature_map, traffic_lead, traffic_pre = initialize_data_structures(
        vehicleADV.nx, vehicleADV.nu, car_N, N_sim, N_episodes, N)

    # 重置交通环境
    traffic.reset()
    traffic_real[:, 0, :] = traffic.getStates()

    # 可视化设置
    matplotlib.use('qt5Agg')
    matplotlib.pyplot.ion()

    # 仿真轮次迭代
    i_overall = 0
    for j in range(N_episodes):
        plt.close('all')
        print(f"仿真轮次: {j + 1}/{N_episodes}")

        # 仿真初始化
        x_current = DM(int(vehicleADV.nx), 1)
        x_current[:], u_current = vehicleADV.getInit()
        vehicleADV.update(x_current, u_current)
        episodes_i = 0
        i = i_overall
        runSimulation = True
        reward_all = 0

        # 仿真循环
        while runSimulation:
            RL_feature_map_i = createFeatureMatrix(vehicleADV, traffic)
            RL_feature_map[:, i:] = RL_feature_map_i
            traffic_lead[:, :] = traffic.prediction()[0, :, :].transpose()
            traffic_pre[:2, :, ] = traffic.prediction()[:2, :, :]

            # 记录车辆速度
            writer.add_scalars('Overall/Velocity', {
                'Velocity': RL_feature_map_i[2][0][0],
                'Maximum Velocity': ref_vx}, i_overall)

            writer.add_scalars(f'Episode_{j}/Velocity', {
                'Velocity': RL_feature_map_i[2][0][0],
                'Maximum Velocity': ref_vx}, episodes_i)

            if (i - episodes_i) % f_controller == 0:
                decisionMaster.storeInput(
                    [x_current, refx_left_out, refx_right_out,
                     refx_trail_out, refu_out, traffic_lead, traffic_pre])

                # 可视化当前场景
                if args.display_simulation:
                    fig = plotScene(RL_feature_map_i, scenarioADV, vehicleADV, car_list)
                    plt.show()
                    plt.pause(0.02)

                # 更新参考轨迹
                refx_left_out, refx_right_out, refx_trail_out = decisionMaster.updateReference()

                # 使用预训练模型进行决策 (不学习)
                x_test, u_test, X_out, _ = decisionMaster.chooseController(RL_feature_map_i)
                u_current = u_test[:, 0]

            # 存储当前的状态和控制输入
            X[:, i] = x_current
            U[:, i] = u_current
            X_pre[:, :, i] = X_out

            # 状态更新
            x_current = F_x_ADV(x_current, u_current)

            try:
                # 更新交通状态
                traffic.update()
                vehicleADV.update(x_current, u_current)
            except:
                print('仿真终止：发生碰撞')
                break

            # 终止条件检查
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
                print('仿真终止：超出道路边界')
            elif (RL_feature_map_i[2][0][0] < 1):
                runSimulation = False
                plt.close('all')
                print('仿真终止：超出道路边界')

            # 交通车重生检查
            traffic.tryRespawn(x_current[0])

            # 存储交通状态
            traffic_real[:, i, :] = traffic.getStates()
            traffic_ref[:, i, :] = traffic.getReference()

            episodes_i += 1
            i_overall += 1

        # 记录轮次统计信息
        writer.add_scalar('Overall/Episode_Iterations', episodes_i, j)
        writer.add_scalar('Overall/Episode_Distances', x_current[0].full().item(), j)
        i_overall = i
        # 重置交通状态
        traffic.reset()
        traffic_real[:, i, :] = traffic.getStates()
        traffic_ref[:, i, :] = traffic.getReference()

    # 保持绘图窗口打开
    if args.display_simulation:
        plt.ioff()
        plt.show()

    print("仿真实验结束")


if __name__ == "__main__":
    main()
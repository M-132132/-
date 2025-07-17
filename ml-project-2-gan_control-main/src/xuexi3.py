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


class DQNTrainingSimulator:
    def __init__(self):
        np.random.seed(1)
        self.device = torch.device('cpu')
        self.num_node_features = 4
        self.n_actions = 3
        self.makeMovie = False
        self.directory = "sim_render"
        self.args = self.parse_arguments()
        self.setup_directories()
        self.initialize_components()
        self.setup_data_structures()

    def parse_arguments(self):
        """解析命令行参数"""
        ap = argparse.ArgumentParser(description='训练DQN智能体')
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

    def setup_directories(self):
        """设置实验目录"""
        assert os.path.isdir(self.args.log_dir), "无效的日志目录"
        self.expid = self.args.exp_id + str(time.time())
        self.exp_dir = os.path.join(self.args.log_dir, self.expid)
        os.makedirs(self.exp_dir)

        self.tensorboard_dir = os.path.join(self.args.log_dir, self.expid, 'tensorboard_logs')
        os.makedirs(self.tensorboard_dir)

    def initialize_components(self):
        """初始化所有组件"""
        # 系统设置
        self.dt = self.args.time_step
        self.f_controller = self.args.controller_frequency
        self.N = self.args.horizon_length
        self.ref_vx = self.args.speed_limit / 3.6
        self.tsim = self.args.simulation_time
        self.N_episodes = self.args.num_episodes
        self.dist_max = self.args.max_dist
        self.N_sim = int(self.tsim / self.dt)

        # 初始化RL Agent
        self.RL_Agent = DQNAgent(self.device, self.num_node_features,
                                 self.n_actions, self.ref_vx, epsilon=0, memory_size=0)

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        with open(os.path.join(self.exp_dir, 'experiment_parameters.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(self.args), f, ensure_ascii=False, indent=4)

        # 初始化车辆
        self.vehicleADV = vehBicycleKinematic(self.dt, self.N)
        self.initialize_vehicle_params()

        # 初始化场景和交通
        self.initialize_scenario_and_traffic()

        # 初始化控制器
        self.initialize_controllers()

        # 可视化设置
        matplotlib.use('qt5Agg')
        plt.ion()

    def initialize_vehicle_params(self):
        """初始化车辆参数"""
        self.vehWidth, self.vehLength, self.L_tract, self.L_trail = self.vehicleADV.getSize()
        self.nx, self.nu, self.nrefx, self.nrefu = self.vehicleADV.getSystemDim()

        # 积分设置
        self.vehicleADV.integrator('rk', self.dt)
        self.F_x_ADV = self.vehicleADV.getIntegrator()

        # 成本函数
        Q_ADV = [0, 40, 3e2, 5, 5]
        R_ADV = [5, 5]
        self.q_ADV_decision = 50
        self.vehicleADV.cost(Q_ADV, R_ADV)
        self.vehicleADV.costf(Q_ADV)

    def initialize_scenario_and_traffic(self):
        """初始化场景和交通车辆"""
        # 场景定义
        self.scenarioTrailADV = trailing(self.vehicleADV, self.N, lanes=3)
        self.scenarioADV = simpleOvertake(self.vehicleADV, self.N, lanes=3)
        self.roadMin, self.roadMax, self.laneCenters = self.scenarioADV.getRoad()

        # 设置车辆初始位置
        vx_ego = 60 / 3.6
        self.vehicleADV.setInit([0, self.laneCenters[0]], vx_ego)

        # 环境车辆定义
        vx_env = 50 / 3.6
        self.car_list = [
            vehicleSUMO(self.dt, self.N, [30, self.laneCenters[1]], [0.9 * vx_env, 0], "normal"),
            vehicleSUMO(self.dt, self.N, [45, self.laneCenters[0]], [0.8 * vx_env, 0], "passive"),
            vehicleSUMO(self.dt, self.N, [100, self.laneCenters[2]], [0.85 * vx_env, 0], "normal"),
            vehicleSUMO(self.dt, self.N, [-20, self.laneCenters[1]], [1.4 * vx_env, 0], "aggressive"),
            vehicleSUMO(self.dt, self.N, [40, self.laneCenters[2]], [1.6 * vx_env, 0], "aggressive")
        ]

        # 交通场景定义
        self.traffic = combinedTraffic(self.car_list, self.vehicleADV, self.ref_vx, self.f_controller)
        self.traffic.setScenario(self.scenarioADV)
        self.car_N = self.traffic.getDim()

    def initialize_controllers(self):
        """初始化MPC控制器"""
        dt_MPC = self.dt * self.f_controller

        # 左变道控制器
        opts1 = {"version": "leftChange", "solver": "ipopt", "integrator": "rk"}
        self.MPC1 = makeController(self.vehicleADV, self.traffic, self.scenarioADV, self.N, opts1, dt_MPC)
        self.MPC1.setController()

        # 右变道控制器
        opts2 = {"version": "rightChange", "solver": "ipopt", "integrator": "rk"}
        self.MPC2 = makeController(self.vehicleADV, self.traffic, self.scenarioADV, self.N, opts2, dt_MPC)
        self.MPC2.setController()

        # 跟车控制器
        opts3 = {"version": "trailing", "solver": "ipopt", "integrator": "rk"}
        self.MPC3 = makeController(self.vehicleADV, self.traffic, self.scenarioTrailADV, self.N, opts3, dt_MPC)
        self.MPC3.setController()

        # 初始化决策主控
        self.decisionMaster = makeDecisionMaster(
            self.vehicleADV, self.traffic, [self.MPC1, self.MPC2, self.MPC3],
            [self.scenarioTrailADV, self.scenarioADV], self.RL_Agent
        )
        self.decisionMaster.setDecisionCost(self.q_ADV_decision)

        # 初始化参考轨迹
        self.initialize_reference_trajectories()

    def initialize_reference_trajectories(self):
        """初始化参考轨迹"""
        self.refxADV = [0, self.laneCenters[1], self.ref_vx, 0, 0]
        refx_trail_in, refx_left_in, refx_right_in = self.vehicleADV.setReferences(self.laneCenters)
        refu_in = [0, 0, 0]

        self.refx_trail_out, self.refu_out = self.scenarioADV.getReference(refx_trail_in, refu_in)
        self.refx_left_out, _ = self.scenarioADV.getReference(refx_left_in, refu_in)
        self.refx_right_out, _ = self.scenarioADV.getReference(refx_right_in, refu_in)
        self.refxADV_out, _ = self.scenarioADV.getReference(self.refxADV, refu_in)

    def setup_data_structures(self):
        """初始化数据结构"""
        self.N_all = self.N_sim * self.N_episodes

        # 主车辆数据
        self.X = np.zeros((self.nx, self.N_all, 1))
        self.U = np.zeros((self.nu, self.N_all, 1))
        self.X_pre = np.zeros((self.nx, self.N + 1, self.N_all))

        # 交通车辆数据
        self.traffic_real = np.zeros((4, self.N_all, self.car_N))
        self.traffic_ref = np.zeros((4, self.N_all, self.car_N))
        self.traffic_lead = DM(self.car_N, self.N + 1)
        self.traffic_pre = np.zeros((5, self.N + 1, self.car_N))

        # RL特征映射
        self.RL_feature_map = np.zeros((6, self.N_sim, self.car_N + 1))

        # 统计变量
        self.i_keything = 0
        self.i_overall = 0
        self.total_distance = 0
        self.num_crashes = 0

    def run_simulation(self):
        """运行主训练循环"""
        for j in range(self.N_episodes):
            self.run_episode(j)

        # 训练结束处理
        print("仿真结束")
        self.RL_Agent.save_model(os.path.join(self.exp_dir, 'final_model.pt'))
        plt.ioff()

    def run_episode(self, episode_idx):
        """运行单个训练轮次"""
        plt.close('all')
        print(f"episode : {episode_idx + 1}")

        # 初始化轮次变量
        x_xunlian = DM(int(self.nx), 1)
        x_xunlian[:], u_xunlian = self.vehicleADV.getInit()
        self.vehicleADV.update(x_xunlian, u_xunlian)

        episodes_i = 0
        i = self.i_overall
        runSimulation = True
        previous_state = None
        action = None
        reward = 0
        reward_all = 0

        # 重置交通环境
        self.traffic.reset()
        self.traffic_real[:, 0, :] = self.traffic.getStates()

        while runSimulation:
            self.simulation_step(x_xunlian, u_xunlian, episode_idx, episodes_i, i,
                                 previous_state, action, reward, reward_all)

            # 更新状态和控制输入
            x_xunlian = self.F_x_ADV(x_xunlian, u_xunlian)

            # 检查终止条件
            runSimulation = self.check_termination_conditions(x_xunlian, i, episode_idx, episodes_i)

            # 更新计数器
            episodes_i += 1
            i += 1

        # 记录轮次统计
        self.record_episode_stats(episode_idx, episodes_i, reward_all, x_xunlian)
        self.i_overall = i

    def simulation_step(self, x_xunlian, u_xunlian, episode_idx, episodes_i, i,
                        previous_state, action, reward, reward_all):
        """执行单个仿真步"""
        # 创建特征矩阵
        RL_feature_map_i = createFeatureMatrix(self.vehicleADV, self.traffic)
        self.RL_feature_map[:, i:] = RL_feature_map_i

        # 更新交通预测
        self.traffic_lead[:, :] = self.traffic.prediction()[0, :, :].transpose()
        self.traffic_pre[:2, :, ] = self.traffic.prediction()[:2, :, :]

        # 记录速度
        self.writer.add_scalars('Overall/Velocity', {
            'Velocity': RL_feature_map_i[2][0][0],
            'Maximum Velocity': self.ref_vx
        }, self.i_overall)

        self.writer.add_scalars(f'Episode_{episode_idx}/Velocity', {
            'Velocity': RL_feature_map_i[2][0][0],
            'Maximum Velocity': self.ref_vx
        }, episodes_i)

        # 控制器更新周期
        if (i - self.i_overall) % self.f_controller == 0:
            self.controller_update(x_xunlian, RL_feature_map_i, episode_idx,
                                   previous_state, action, reward, reward_all)

        # 更新奖励
        reward += RL_feature_map_i[2][0][0] / self.ref_vx

        # 存储数据
        self.X[:, i] = x_xunlian
        self.U[:, i] = u_xunlian
        self.X_pre[:, :, i] = self.X_out  # 来自controller_update

        # 更新交通状态
        try:
            self.traffic.update()
            self.vehicleADV.update(x_xunlian, u_xunlian)
        except:
            print('仿真完成：发生碰撞')
            self.num_crashes += 1
            self.writer.add_scalar('Overall/Crash_1_Derail_2', 1, self.i_overall)
            raise  # 终止当前轮次

        # 尝试重生车辆
        self.traffic.tryRespawn(x_xunlian[0])
        self.traffic_real[:, i, :] = self.traffic.getStates()
        self.traffic_ref[:, i, :] = self.traffic.getReference()

    def controller_update(self, x_xunlian, RL_feature_map_i, episode_idx,
                          previous_state, action, reward, reward_all):
        """控制器更新逻辑"""
        # 存储输入
        self.decisionMaster.storeInput([
            x_xunlian,
            self.refx_left_out,
            self.refx_right_out,
            self.refx_trail_out,
            self.refu_out,
            self.traffic_lead,
            self.traffic_pre
        ])

        # 可视化
        if self.args.display_simulation:
            fig = plotScene(RL_feature_map_i, self.scenarioADV, self.vehicleADV, self.car_list)
            plt.show()
            plt.pause(0.02)

        # 更新参考轨迹
        self.refx_left_out, self.refx_right_out, self.refx_trail_out = \
            self.decisionMaster.updateReference()

        # 记录奖励
        self.writer.add_scalar(f'Episode_{episode_idx}/Rewards', reward, reward_all)

        # 选择控制器
        _, u_test, self.X_out, selected_action = \
            self.decisionMaster.chooseController(RL_feature_map_i)

        return u_test[:, 0]  # 返回控制输入

    def check_termination_conditions(self, x_xunlian, i, episode_idx, episodes_i):
        """检查终止条件"""
        if (i - self.i_overall == self.N_sim - 1):
            print('仿真终止：达到最大仿真步数')
            return False
        elif (x_xunlian[0].full().item() > self.dist_max):
            print('仿真终止：达到最大距离')
            return False
        elif (x_xunlian[1].full().item() > self.roadMax) or \
                (x_xunlian[1].full().item() < self.roadMin):
            print('仿真终止：达到道路边界')
            return False
        return True

    def record_episode_stats(self, episode_idx, episodes_i, reward_all, x_xunlian):
        """记录轮次统计信息"""
        self.writer.add_scalar('Overall/Average_Rewards', reward_all / episodes_i, self.i_overall)
        self.writer.add_scalar('Overall/Episode_Iterations', episodes_i, episode_idx)
        self.writer.add_scalar('Overall/Episode_Distances', x_xunlian[0].full().item(), episode_idx)
        self.total_distance += x_xunlian[0].full().item()


if __name__ == "__main__":
    simulator = DQNTrainingSimulator()
    simulator.run_simulation()
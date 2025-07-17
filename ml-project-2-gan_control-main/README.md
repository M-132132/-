# 机器学习项目 2：使用图神经网络的深度强化学习实现自主变道

![](https://github.com/BorveErik/Autonomous-Truck-Sim/blob/RL_training_env/simRes.gif)

**Authors:** Arvind Satish Menon, Lars C.P.M. Quaedvlieg, and Somesh Mehra

**Supervisor:** [Erik Börve](mailto://borerik@chalmers.se)

**Group:** GAN_CONTROL

## 重要链接
- Final report: https://www.overleaf.com/read/sbtzctmpbqpy
- In-depth project description: https://docs.google.com/document/d/1_oW5013IwnaLW3alfvVjh7CEu5wkTPQh/edit

## 项目介绍

近年来，自动驾驶汽车因其在提高安全性、 效率和交通的可达性。自动驾驶的一个重要方面是能够 变道决策，这要求车辆预测其他道路使用者的意图和行为，并 评估不同行动的安全性和可行性。

在这个项目中，我们提出了一种结合强化学习 （RL） 和模型的图神经网络 （GNN） 架构 Predictive Controller 解决自主变道问题。通过使用 GNN，可以学习控件 策略，该策略考虑了车辆之间复杂而动态的关系，而不仅仅是考虑 局部特征或模式。更具体地说，我们对代理体采用带有 Graph Attention Networks 的 Deep Q-learning。

请注意，并非所有代码都是该项目组的工作。我们将使用监理提供的基础 以此为基础。有关此基础的想法，请使用 [this repository](https://github.com/BorveErik/Autonomous-Truck-Sim).
但是，我们还将在本 README 后面提到我们对每个单独文件的贡献。

来自上面链接的存储库:

>  该项目提供了在多车道高速公路场景中实现自动驾驶卡车。控制器利用 非线性最优控制来计算多个可行的轨迹，其中选择最具成本效益的轨迹。这 simulation 设置为适合 RL 训练。

## 开始

在本地计算机中克隆项目。

### 要求

找到存储库并运行:
  ```sh
  pip install -r requirements.txt
  ```

**此外**, 您需要分别安装 Pytorch 和 Pytorch Geographic。请注意，对于安装 
[Pytorch](https://pytorch.org/get-started/locally/) 和 [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html),
请参考 CPU 或 GPU 的参考安装指南，并确保兼容版本的 Pytorch 具有 已安装用于 Pytorch Geometric。

| Package            | Use                         |
|--------------------|-----------------------------|
| Pytorch (Geometric) | Automatic Differentiation   |
| Networkx           | Graph representation        |
| Casadi             | Nonlinear optimization      |
| Numpy              | Numerical computations      |
| Matplotlib         | Plotting and visualizations |
| PyQt5              | Plotting and visualizations |
| Tensorboard        | Experiment visualizations   |
| ipython            | Interactive display         |

### 用法

RL 模型可以通过 “main.py” 文件使用特定配置（请参阅下面的格式）进行训练。这也是 配置模拟，包括设计交通场景和设置最佳控制器等。



## 存储库的结构

    .
    ├── out               # 代码生成的任何输出文件
    ├── res               # 代码中可以或正在使用的任何资源（例如配置）
    ├── src               # 项目源代码
    ├── .gitignore        # 配置设置
    ├── README.md         # 存储库描述
    └── requirements.txt  # 运行代码前要安装的Python包

### 输出文件
    .
    ├── out                   
    │   └── runs          # 包含实验日志和模型等文件夹。
    │       └── example   # 包含训练模型的示例实验文件夹
    └── ...

> 这是存储从代码生成的所有输出文件的目录

### 资源文件
    .
    ├── ...
    ├── res                   
    │   ├── model_hyperparams             # 存储RL模型超参数配置的目录
    │   │   └── example_hyperparams.json  # 超参数配置文件示例
    │   ├── ex2.csv                       # 模拟中生成的原始数据示例文件（仅用于直观）
    │   ├── metaData_ex2.txt              # ex2.csv的源数据
    │   └── simRes.gif                    # 模拟中一轮次的gif示例
    └── ...

> 这是代码中可以或正在使用的任何资源（例如配置）的目录

### 程序文件
    .
    ├── ...
    ├── src  
    │   ├── agents                 # 为存储与 RL 代理有关的任何内容而创建的包
    │   │   ├── __init__.py        # 创建 Python 包
    │   │   ├── graphs.py          # 包含创建图数据结构类的文件
    │   │   └── rlagent.py         # 包含从网络架构到深度 Q 学习代理的任何内容的文件
    │   ├── controllers.py         # 根据指定的场景生成最优控制器，并优化轨迹选择，返回最优策略
    │   ├── helpers.py             # 包含辅助函数
    │   ├── inference.py           # Performing inference with a trained RL agent
    │   ├── main.py                # 项目运行的文件，设置并运行模拟以训练 RL 代理
    │   ├── scenarios.py           # 为最优控制器中考虑的不同场景制定约束条件
    │   ├── traffic.py             # 用于与交通场景中的所有车辆通信，以指定的起始位置、速度和等级创建车辆。
    │   ├── vehicleModelGarage.py  # 包含可在模拟中使用的主车辆模型
    │   ├──xuexi_re.py             # main文件重构后的程序
    │   ├──xuexi_test_re.py        # 仿真实验程序脚本
    │   ├──xuexi_train.py          #模型训练脚本
    │   ├──plot_result.py          #实验结果绘制脚本
    └── ...

> 这是包含此项目的所有源代码的目录

### 执行文件的参数

#### main.py

要运行此脚本，您至少必须提供 json 格式的超参数配置文件，该文件使用 -H 标志指定。示例文件如下，其中包含用于最终模型的超参数，可以使用以下命令直接从 src 文件夹内运行脚本：

```bash
$ python main.py -H ../res/model_hyperparams/example_hyperparams.json
```

注意：这假设 out/runs 文件夹存在于存储库中（克隆后应该如此）。如果没有，您可以使用 -l 标志指定备用日志目录，所有输出都将保存在其中。

此外，还建议（但不是必需的）提供带有 -E 标志的实验 ID，以便更轻松地在日志目录中找到实验结果。结果保存在日志目录中名为 {experiment_ID}_{timestamp} 的文件夹中。

此脚本的输入参数的完整列表如下所示，其余参数具有默认值，但可用于更改各种仿真参数。

```bash
$ python main.py -h

usage: main.py [-h] -H HYPERPARAM_CONFIG [-l LOG_DIR] [-e NUM_EPISODES]
               [-d MAX_DIST] [-t TIME_STEP] [-f CONTROLLER_FREQUENCY]
               [-s SPEED_LIMIT] [-N HORIZON_LENGTH] [-T SIMULATION_TIME]
               [-E EXP_ID] [--display_simulation]

训练 DeepQN RL 代理做出变道决策

options:
  -h, --help            显示此帮助信息并退出
  -H HYPERPARAM_CONFIG, --hyperparam_config HYPERPARAM_CONFIG
                        训练 GNN 的超参数的 json 文件路径。
                        hyperparameters: gamma (float), target_copy_delay
                        (int), learning_rate (float), batch_size (int),
                        epsilon (float), epsilon_dec (float), epsilon_min
                        (float), memory_size (int)
  -l LOG_DIR, --log_dir LOG_DIR
                        存储日志的目录。默认 ../out/runs
  -e NUM_EPISODES, --num_episodes NUM_EPISODES
                        模拟运行的轮次数。 默认 100
  -d MAX_DIST, --max_dist MAX_DIST
                        车辆行驶的目标距离。默认 10000m
  -t TIME_STEP, --time_step TIME_STEP
                        模拟时间步长。默认 0.2 秒
  -f CONTROLLER_FREQUENCY, --controller_frequency CONTROLLER_FREQUENCY
                        控制器更新频率。 默认 5
  -s SPEED_LIMIT, --speed_limit SPEED_LIMIT
                        限速 (km/h). 默认 60
  -N HORIZON_LENGTH, --horizon_length HORIZON_LENGTH
                         MPC 水平线长度。 默认 12
  -T SIMULATION_TIME, --simulation_time SIMULATION_TIME
                        最大总模拟时间（秒）。 默认 100
  -E EXP_ID, --exp_id EXP_ID
                        实验名
  --display_simulation  显示仿真，将在每个时间步绘制仿真图并显示
```

_运行项目需向脚本提供相同的超参数配置和输入参数。对于每个实验，所有实验参数（包括超参数配置）保存在实验日志文件夹中名为 experiment_parameters.json 的文件中
例：-H ../res/model_hyperparams/example_hyperparams.json -l ../out/runs -e 2 -s 60 --display_simulation -E test4 -T 1 -d 500
#### xuexi_re.py
对main文件重构后的程序，运行脚本配置同main.py
#### xuexi_test_re.py
只实现仿真实验功能，运行脚本配置同main.py相似，提供的实验名需是已经存在的日志文件，该文件中需要包含final_model.pt模型文件
#### xuexi_train.py
只实现模型训练功能，运行脚本配置同main.py
#### plot_result.py
只实现结果绘制功能，运行需提供已存在的日志文件路径，例：--log_dir E:\edge_downloads_file\ml-project-2-gan_control-main\ml-project-2-gan_control-main\out\runs\example





# D435i 近距离抛球录制与接球能力分析

本流程用于回答三个独立问题：

1. D435i 在近距离高速球场景下是否连续输出可靠 RGB-D 球心；
2. `catch_controller` 是否足够早、足够准确地预测球心接触平面落点；
3. 在当前速度和加速度限制下，底盘是否来得及到达目标。

视觉录像只用于人工复核。所有速度、延迟和撞击时刻分析均使用 ROS 消息
`header.stamp`，不能使用 MJPG 视频帧率推算时间。

## 1. 构建和环境

RDK 上先同步最新 `ros2_ws`，再构建分析器：

```bash
source /opt/ros/humble/setup.bash
cd ~/RDK-S100p-BPU-yolo/ros2_ws
colcon build --packages-select volleyball_interfaces volleyball_catch_controller
source install/setup.bash
```

录制脚本位于 `USB2CAN_motor` 的 `rviz_bag_tools`：

```bash
cd ~/USB2CAN_motor
colcon build --packages-select rviz_bag_tools
source install/setup.bash
```

所有终端使用相同的 DDS 环境：

```bash
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
source ~/USB2CAN_motor/install/setup.bash
source ~/RDK-S100p-BPU-yolo/ros2_ws/install/setup.bash
```

## 2. 原始感知录制

固定底盘并保持 RC 急停。启动视觉程序：

```bash
mkdir -p ~/test_data
cd ~/RDK-S100p-BPU-yolo/yolov8/build
./yolov8_usb_camera \
  --config=../config/yolo26_d435i_848x480_60fps.yaml \
  --display=false \
  --headless=true \
  --save_video="$HOME/test_data/d435_raw_001.avi" \
  2>&1 | tee ~/test_data/d435_raw_001.log
```

另一个终端开始 rosbag：

```bash
cd ~/USB2CAN_motor
ros2 run rviz_bag_tools bag_record.py \
  --config src/rviz_bag_tools/config/d435_throw_bag.yaml \
  --output ~/test_data/d435_raw_001
```

配置中包含后续控制和机械臂话题。感知测试时这些话题没有发布者不会影响
`/d435/ball/observation` 和 `/camera/camera/imu` 的录制。

## 3. 影子控制录制

影子控制只向测试话题发布速度，不允许遥控节点转发到底盘：

```bash
ros2 launch volleyball_catch_controller d435_only_chase.launch.py \
  odom_topic:=/odom \
  cmd_vel_topic:=/vision/cmd_vel_test
```

开始前必须确认：

```bash
ros2 run volleyball_catch_controller check_chassis_odom \
  --ros-args -p topic:=/odom
ros2 topic info /vision/cmd_vel_test --verbose
```

录制命令与原始感知阶段相同，换一个输出目录即可。分析器会自动读取
`/ball/landing`、`/catch/state` 和 `/auto/goal_pose`。

## 4. 抛球真值 CSV

复制模板并删除示例行：

```bash
cp ~/RDK-S100p-BPU-yolo/ros2_ws/src/volleyball_catch_controller/config/d435_throw_ground_truth.csv \
  ~/test_data/d435_shadow_001_truth.csv
```

格式：

```csv
trial_id,start_s,impact_s,impact_x_m,impact_y_m
throw_001,2.500,3.420,1.135,-0.082
```

- 默认 `start_s`、`impact_s` 是相对 bag 第一条消息的秒数；
- 使用绝对 ROS 时间时，分析命令增加 `--time-reference absolute`；
- `impact_x_m`、`impact_y_m` 必须是 `odom` 坐标系下球心真实接触位置；
- 固定底盘测试可使用地面/接球盘坐标网格；移动底盘测试需要外部相机或撞击传感器；
- 只填时间、不填坐标仍可分析感知和撞击时间，但落点能力保持 `INCOMPLETE`。

外部高速相机至少应记录释放、第一次稳定预测对应阶段和真实接触帧。建议通过画面中
可见的同步动作对齐 bag 导出的 `relative_time_s`。

## 5. 分析命令

只有原始感知、没有真值时：

```bash
ros2 run volleyball_catch_controller analyze_d435_throw_bag \
  --bag ~/test_data/d435_raw_001 \
  --output ~/test_data/d435_raw_001_report.json \
  --csv-dir ~/test_data/d435_raw_001_csv
```

有真实落点时，还必须给出接球盘允许的球心误差。它不是接球盘半径，而是：

```text
有效球心容差 = 接球盘有效半径 - 排球半径 0.1075 m - 安全余量
```

```bash
ros2 run volleyball_catch_controller analyze_d435_throw_bag \
  --bag ~/test_data/d435_shadow_001 \
  --ground-truth ~/test_data/d435_shadow_001_truth.csv \
  --target-radius 0.10 \
  --output ~/test_data/d435_shadow_001_report.json \
  --csv-dir ~/test_data/d435_shadow_001_csv
```

`--target-radius 0.10` 只是命令示例，必须替换为实测机械容差。

## 6. 报告内容

每个试次输出：

- 检测率、RGB-D 有效率、时间映射有效率；
- 帧间隔 P50/P99、采集到 rosbag 的延迟 P50/P95；
- 检测置信度 P10、有效深度离散度 P90；
- 第一次稳定落点距离真实撞击的提前量；
- 落点误差 P50/P90；
- 撞击时间误差 P50 和绝对误差 P90；
- TTI=1.0/0.8/0.6/0.4/0.25 秒附近的落点误差；
- 当前速度、加速度和刹车余量下，从静止出发的可达距离。

判定含义：

- `PASS`：该试次所有已配置门限通过；
- `FAIL`：至少一个已配置门限失败；
- `INCOMPLETE`：缺少真值、预测、目标有效半径或其他必需数据；
- `NO_DATA`：门限已配置，但 bag 中没有对应样本；
- `NOT_CONFIGURED`：例如尚未提供接球盘有效半径。

默认门限位于
`volleyball_catch_controller/config/d435_throw_analysis.yaml`。先保留原始报告，再根据实测机械
容差修改配置，禁止为了让旧数据通过而反向放宽门限。

## 7. 推荐试次

1. 静态球：0.5–3.0 m，中心和画面边缘各 5–10 秒；
2. 固定底盘抛球：2 m、3 m、左右偏置、高弧、低弧各至少 10 次；
3. 影子控制：输出到 `/vision/cmd_vel_test`，核对方向和可达性；
4. 架空轮：恢复 `/vision/cmd_vel`，验证急停和失联归零；
5. 低速地面：先限制活动区域，再逐步增加抛球范围；
6. 参数冻结后至少 60 次代表性抛球，再评估 95% 成功目标。

击球能力不能只由落点报告证明。还需同步录制 GO8010 命令/状态和真实 `/catch/event`，
分别测量机械臂触发延迟、接触时刻误差和重复性。

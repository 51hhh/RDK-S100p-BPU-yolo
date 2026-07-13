# RDK ROS2 工作区

## 构建

```bash
source /opt/ros/humble/setup.bash
cd ~/RDK-S100p-BPU-yolo/ros2_ws
colcon build
source install/setup.bash
```

先构建并 source 本工作区，再配置 `yolov8`，这样 CMake 才能找到
`volleyball_interfaces` 并启用 `/d435/ball/observation` 发布。

```bash
cd ~/RDK-S100p-BPU-yolo/yolov8
rm -rf build
mkdir build && cd build
cmake ..
make -j4
```

## 启动顺序

单 RDK + D435i 模式请直接参考[单 D435i 追球说明](../wiki/单D435i追球.md)。该模式不启动
NX 订阅和 NX 时间同步，控制输出由 `/auto/goal_pose` 经过安全限速节点转换为
`/vision/cmd_vel`。

1. 启动舵轮底盘里程计。当前阶段直接使用底盘`nav_msgs/Odometry`，不在本仓库重复计算
   码盘里程计，也暂不融合D435i IMU。先执行契约检查：

```bash
ros2 run volleyball_catch_controller check_chassis_odom \
  --ros-args -p topic:=/odom
```

要求`frame_id=odom`、`child_frame_id=base_link`、时间戳属于RDK系统时间域、频率至少20 Hz，
推荐50–100 Hz。若底盘话题为`/chassis/odom`，检查和启动时都传入该话题。
2. 启动控制节点：

```bash
ros2 launch volleyball_catch_controller catch_controller.launch.py odom_topic:=/odom

# 底盘使用其他话题时：
# ros2 launch volleyball_catch_controller catch_controller.launch.py \
#   odom_topic:=/chassis/odom
```

3. 启动 D435i 视觉：

```bash
cd ~/RDK-S100p-BPU-yolo/yolov8/build
./yolov8_usb_camera --config=../config/yolo26_d435i_848x480_60fps.yaml
```

4. 启动 NX 远场节点，发布原始双目观测 `/nx/ball/observation`。

NX侧先准备Chrony和共享`source_epoch`，再启动同步发布器：

```bash
source ros2_ws/scripts/use_nx_transport.sh
sudo install -d -o "$USER" -g "$(id -gn)" /run/volleyball
# 每次联合启动生成新epoch；NX观测进程必须读取同一文件或环境变量。
export NX_SOURCE_EPOCH=$(ros2 run volleyball_catch_controller prepare_nx_epoch)
ros2 launch volleyball_catch_controller nx_time_sync.launch.py
```

同步发布器读取`chronyc -c tracking`，发布测量时刻、序号、offset、uncertainty、servo状态
和同步源。默认要求测量与接收状态均新鲜、servo锁定、偏差不超过20 ms、估计不确定度
不超过2 ms，否则远场观测不会进入控制；偏差5–20 ms时降低NX观测权重。

D435观测同时发布RGB与Depth采集时间差、librealsense时间域和映射不确定度。默认只允许
GLOBAL_TIME/SYSTEM_TIME更新近场滤波；RGB/Depth差超过2 ms或映射无效时不能接管。

实车启用前必须填写
`volleyball_catch_controller/config/catch_controller.yaml` 中两个相机外参，并分别将
`nx_extrinsics_calibrated`、`d435_extrinsics_calibrated` 设置为 `true`。
默认 `require_calibrated_extrinsics=true`，未显式确认的相机不会产生有效控制输出。

## 检查唯一控制发布者

```bash
ros2 topic info /auto/goal_pose --verbose
ros2 topic echo /catch/state
ros2 topic echo /auto/goal_valid
```

`/auto/goal_pose` 必须只有 `catch_controller` 一个发布者。

## 球轨迹输出

控制节点先用观测采集时刻的 `/odom` 将 NX/D435 相机坐标转换到连续 `odom` 坐标，
再分别更新远场和近场 Student-t 阻力 EKF。物理参数和 RK4 落点模型与 NX 当前主路径
保持一致。当前活动滤波器发布：

- `/ball/filtered_position`: `odom` 下滤波球位置；
- `/ball/filtered_velocity`: `odom` 下滤波球速度；
- `/ball/predicted_path`: 受重力作用直到落地的预测轨迹；
- `/ball/landing`: `odom` 下预测落点。

三个滤波诊断话题使用 Best Effort，并由 `diagnostics_rate_hz` 限频；它们不参与底盘控制。

## 通信与断网恢复诊断

```bash
ros2 topic echo /diagnostics/transport
ros2 run volleyball_catch_controller transport_recovery_test
```

第二条命令会自动验证DDS deadline、断连、旧NX/同步消息重放以及新epoch恢复。双板部署前
还应分别运行`ros2_ws/scripts/check_transport.sh nx eth0`和`... rdk eth0`。

`/diagnostics/transport`中的`odom_online`仅表示话题仍在收包；`odom_valid`还要求最近存在
通过frame、时间戳和单调性检查的样本。`odom_rate_hz`和`odom_rejected`用于现场定位底盘
里程计频率不足、旧时间戳或坐标系错误。

NX 发送的是 `HybridDepthEstimator` 生成的当前帧混合深度原始观测；RDK 不复用 NX 的
9D 深度滤波状态，而是在 `odom` 下独立运行接球所需的 Student-t 弹道滤波。

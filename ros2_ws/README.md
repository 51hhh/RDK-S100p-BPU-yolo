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

1. 启动底盘 `/odom`。
2. 启动控制节点：

```bash
ros2 launch volleyball_catch_controller catch_controller.launch.py
```

3. 启动 D435i 视觉：

```bash
cd ~/RDK-S100p-BPU-yolo/yolov8/build
./yolov8_usb_camera --config=../config/yolo26_d435i_848x480_60fps.yaml
```

4. 启动 NX 远场节点，发布原始双目观测 `/nx/ball/observation`。

NX 同时必须发布 `/diagnostics/time_sync` (`TimeSyncStatus`)。默认配置要求时钟状态新鲜、
偏差不超过 20 ms、估计不确定度不超过 2 ms，否则远场观测不会进入控制。

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

注意：Student-t EKF、阻力物理、RK4 和控制门控已与 NX 对齐；要让真实数据流逐帧复现
NX 本机旧轨迹结果，NX 传输消息还需携带旧轨迹模块实际选中的观测及其最终测量噪声，
不能只依赖当前 raw position 的保守传输协方差。

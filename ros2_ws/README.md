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

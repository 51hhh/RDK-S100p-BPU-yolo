# RDK S100P + D435i + YOLO26 RGB-D 使用说明

本文记录 RDK S100P 上的编译、启动、配置、可视化、深度图保存和常见故障处理方式。

## 1. 当前数据流

```text
Intel RealSense D435i
  ├─ Color 848x480@60 BGR8
  └─ Depth 848x480@60 Z16
          │
          ├─ librealsense 同步取流
          ├─ rs2::align 深度对齐到彩色坐标
          └─ 可选空间深度滤波
                  │
                  ├─ 彩色图 → NV12 → RDK BPU YOLO26
                  └─ 对齐深度图 + bbox → 球面多点测距
```

D435i 内部 D4 ASIC 已计算好双目深度，并向主机输出 Z16 深度图。本项目不重新计算双目视差。本地 CPU 负责深度和彩色对齐、滤波、球面采样、鲁棒统计、去畸变和单目测距；BPU 只负责 YOLO 推理。

## 2. 环境要求

- RDK S100P，AArch64 Ubuntu
- D-Robotics UCP SDK
- OpenCV 4.x
- yaml-cpp
- librealsense2 2.58.x
- Intel RealSense D435i

板端检查命令：

```bash
pkg-config --modversion realsense2
rs-enumerate-devices -s
lsusb | grep -i RealSense
```

正常情况下应能看到：

```text
Intel RealSense D435I
Serial Number: 035422071550
```

注意以下两个编号用途不同：

| 编号 | 用途 |
|------|------|
| `035422071550` | librealsense `enable_device()` 使用的 Serial Number |
| `035323051042` | ASIC/Firmware Update ID，也是当前 V4L by-id 路径中的编号 |

## 3. 编译

在 RDK 上执行：

```bash
cd /home/sunrise/RDK-S100p-BPU-yolo/yolov8
mkdir -p build
cd build
cmake ..
cmake --build . -j4
```

CMake 输出中应包含：

```text
-- RealSense RGB-D: enabled
```

生成程序：

```text
/home/sunrise/RDK-S100p-BPU-yolo/yolov8/build/yolov8_usb_camera
```

## 4. 启动方式

### 4.1 前台启动

适合本地桌面调试：

```bash
cd /home/sunrise/RDK-S100p-BPU-yolo/yolov8/build

./yolov8_usb_camera \
    --config=../config/yolo26_d435i_848x480_60fps.yaml
```

窗口中按 `q` 退出。

### 4.2 SSH 后台启动

```bash
cd /home/sunrise/RDK-S100p-BPU-yolo/yolov8/build

nohup ./yolov8_usb_camera \
    --config=../config/yolo26_d435i_848x480_60fps.yaml \
    --display=true \
    --fullscreen=false \
    > /tmp/yolo26_rgbd.log 2>&1 < /dev/null &
```

查看进程和日志：

```bash
pgrep -af yolov8_usb_camera
tail -f /tmp/yolo26_rgbd.log
```

停止时先取得 PID，再显式终止：

```bash
kill <PID>
```

### 4.3 无界面运行

适合远程测试或没有显示器的环境：

```bash
./yolov8_usb_camera \
    --config=../config/yolo26_d435i_848x480_60fps.yaml \
    --display=false \
    --headless=true
```

### 4.4 无界面运行并保存彩色和深度图

```bash
./yolov8_usb_camera \
    --config=../config/yolo26_d435i_848x480_60fps.yaml \
    --display=false \
    --headless=true \
    --snapshot_dir=/tmp/yolo26_snapshots \
    --snapshot_interval=30
```

输出目录：

```text
/tmp/yolo26_snapshots/
├── frame_000030.jpg
├── frame_000060.jpg
└── depth_mm/
    ├── depth_000030.png
    └── depth_000060.png
```

`depth_mm/*.png` 是 16 位单通道 PNG，每个像素值表示毫米。值为 `0` 表示无有效深度。

### 4.5 从开发机远程启动

```bash
sshpass -p '<password>' ssh sunrise@10.42.0.225 \
    'cd /home/sunrise/RDK-S100p-BPU-yolo/yolov8/build && \
     ./yolov8_usb_camera \
       --config=../config/yolo26_d435i_848x480_60fps.yaml'
```

建议使用 SSH 密钥代替在命令行中保存密码。

## 5. 配置文件

RGB-D 默认配置：

```text
yolov8/config/yolo26_d435i_848x480_60fps.yaml
```

命令行显式参数优先于 YAML。例如：

```bash
./yolov8_usb_camera \
    --config=../config/yolo26_d435i_848x480_60fps.yaml \
    --score_thres=0.10 \
    --depth_show_colormap=true
```

### 5.1 camera

```yaml
camera:
  width: 848
  height: 480
  fps: 60
  auto_exposure: false
  exposure_us: 15000
  gain: 128
  auto_white_balance: false
  white_balance_temperature: 4600
  warmup_frames: 120
  strict: true
```

| 参数 | 说明 |
|------|------|
| `width/height/fps` | 彩色和深度共同使用的 profile |
| `exposure_us` | 曝光时间，配置单位为微秒；程序会转换为 D435i RGB 的 100µs 控制单位 |
| `gain` | D435i RGB 增益，范围 0–128；当前使用最大值 128 |
| `strict` | 控件不支持或读回不一致时退出 |

### 5.2 depth

```yaml
depth:
  enabled: true
  serial: "035422071550"
  object_diameter_m: 0.215
  bbox_scale: 0.95
  min_m: 0.20
  max_m: 12.0
  sample_radius_scale: 0.68
  sample_rings: 3
  samples_per_ring: 12
  patch_radius: 1
  min_samples: 6
  mad_scale: 3.0
  max_spread_m: 0.12
  mono_consistency_relative: 0.45
  mono_consistency_absolute_m: 0.75
  spatial_filter: true
  temporal_filter: false
  hole_filling_filter: false
  draw_samples: true
  show_colormap: false
```

| 参数 | 说明 |
|------|------|
| `object_diameter_m` | 排球实际直径，决定球心反解和单目距离尺度 |
| `bbox_scale` | 检测框相对真实投影直径的标定系数 |
| `sample_radius_scale` | 采样椭圆相对 bbox 半径的比例；小于 1 可减少边缘背景污染 |
| `sample_rings` | 同心采样环数量，另外固定采样中心点 |
| `samples_per_ring` | 每个环上的采样点数量 |
| `patch_radius` | 每个采样位置局部中值半径；1 表示 3×3 |
| `min_samples` | 输出 RGB-D 距离所需的最少有效点数 |
| `mad_scale` | Median Absolute Deviation 离群点门限倍数 |
| `max_spread_m` | 球心候选允许的最大鲁棒离散度 |
| `mono_consistency_*` | RGB-D 候选相对单目尺度先验的背景剔除门限 |
| `spatial_filter` | SDK 空间滤波，默认开启 |
| `temporal_filter` | SDK 时域滤波；高速运动容易产生拖尾，默认关闭 |
| `hole_filling_filter` | 深度孔洞填充；可能引入背景值，默认关闭 |
| `draw_samples` | 在彩色图上画出通过筛选的黄色采样点 |
| `show_colormap` | 显示对齐深度伪彩窗口 |

### 5.3 detector

```yaml
detector:
  model_path: "../model/yolo26n_nashm_640x640_nv12.hbm"
  label_file: "../model/yolo26_classes.names"
  score_threshold: 0.05
  nms_threshold: 0.45
```

`score_threshold` 越低越容易检出远距离或模糊目标，也会增加误检。

### 5.4 display

```yaml
display:
  enabled: true
  fullscreen: false
  width: 1280
  height: 720
```

### 5.5 recording

```yaml
recording:
  video_path: ""
  snapshot_dir: ""
  snapshot_interval: 0
```

`snapshot_interval=30` 表示每 30 帧保存一张带标注彩色图和一张对应的毫米深度图。

## 6. 可视化说明

### 6.1 状态面板

画面左上角显示：

| 字段 | 说明 |
|------|------|
| `Detections` | 当前检测数量 |
| `Camera` | RGB-D 实际处理帧率 |
| `Pipeline` | YOLO 预处理、推理和后处理理论帧率 |
| `Max candidate` | 当前帧最大的分类候选置信度 |
| `E2E` | 取流到检测及测距完成的端到端耗时 |
| `Pre/Infer/Post` | YOLO 各阶段耗时 |

### 6.2 检测框深度面板

深度信息显示在检测框下方；空间不足时自动移到检测框上方。

| 字段 | 含义 |
|------|------|
| `RGB-D Zc` | RGB-D 估计的球心光轴深度，单位米 |
| `RGB-D Zs` | 沿球心射线最近球表面的光轴深度 |
| `XYZ` | 彩色相机坐标系中的球心位置，单位米 |
| `n` | 通过球体几何和鲁棒筛选的有效采样点数 |
| `Mono center` | 去畸变 bbox 角直径计算的单目球心深度 |
| `Mono surface` | 单目球表面深度 |

相机坐标系定义：

```text
X：向右
Y：向下
Z：向前
```

黄色点表示最终参与 RGB-D 球心估计的球面采样位置。

## 7. 深度计算方法

每个检测框采用以下步骤：

1. 对 bbox 中心和边缘点进行 Brown-Conrady 去畸变。
2. 在缩小椭圆内采样中心点及多层同心环。
3. 每个位置读取对齐深度图的局部中值。
4. 将像素和深度反投影为相机坐标系 3D 球面点。
5. 根据已知球半径和 bbox 中心射线反解球心距离候选。
6. 用单目距离先验排除明显背景点。
7. 使用 Median/MAD 排除飞点并计算最终球心。

单目测距使用球体角直径模型：

```text
center_range = radius / sin(angular_diameter / 2)
```

它比小角度近似 `f × diameter / bbox_width` 在近距离更准确。

## 8. 当前板端实测

848×480@60、空间滤波开启时：

| 项目 | 实测 |
|------|------|
| BPU 推理 | 约 1.4 ms |
| YOLO 预处理+推理+后处理 | 约 2.7–3.1 ms |
| librealsense 取流、对齐和滤波 | 约 18–20 ms |
| 端到端 | 约 21–23 ms |
| 实际处理帧率 | 约 40–44 FPS |
| 球面有效采样点 | 约 36–37 点 |
| 球心候选鲁棒离散度 | 约 0–1 cm |

60 FPS 目标的主要优化对象是 `rs2::align`、空间滤波和彩色帧复制，不是 BPU 推理。

## 9. 常见故障

### `No device connected`

```bash
rs-enumerate-devices -s
lsusb | grep -i RealSense
```

若 USB 能看到但 SDK 暂时无法枚举，重新插拔 D435i 后再测试。

同时确认配置使用 librealsense Serial Number：

```yaml
serial: "035422071550"
```

### `out of range value for argument value`

检查曝光、增益和白平衡范围：

```bash
rs-enumerate-devices -o
```

D435i RGB 常用范围：

```text
Exposure:      1–10000，控制单位 100µs
Gain:          0–128
White Balance: 2800–6500，步长 10K
```

YAML 中 `exposure_us` 使用微秒，程序内部会除以 100 后写入 SDK。

### CMake 没有启用 RealSense

确认：

```bash
pkg-config --modversion realsense2
ls /usr/local/include/librealsense2/rs.hpp
ldconfig -p | grep realsense
```

重新运行：

```bash
cd yolov8/build
cmake ..
cmake --build . -j4
```

### RGB-D 距离 invalid

依次检查：

1. bbox 内是否有黄色有效采样点。
2. 深度 PNG 中球区域是否为非零值。
3. `min_samples` 是否过大。
4. `max_spread_m` 是否过小。
5. 球被手或其他物体大面积遮挡时，是否仍有足够可见球面。

### RGB-D 与单目距离差异较大

- 先确认排球实际直径并修改 `object_diameter_m`。
- 在多个已知距离处标定 `bbox_scale`。
- 检查 bbox 是否包含手部或大量背景。
- RGB-D 通常作为主测量，单目结果用于对照和深度失效回退。

## 10. 标定建议

将球放置在 0.5m、1m、2m、3m 等已知距离，每个位置记录至少 100 帧：

1. 计算 RGB-D `Zc` 的均值、偏差和标准差。
2. 调整 `object_diameter_m` 为实际测量直径。
3. 调整 `bbox_scale`，使单目结果与真值一致。
4. 检查 `n` 和 spread，确认采样稳定。
5. 测试球位于画面中心、边缘以及部分遮挡的情况。

完成标定前，当前结果可用于功能验证和相对运动测量，但不应直接声明为计量级绝对精度。
# 轨迹滤波与落点发布

程序在逐帧 RGB-D 球心测量之后增加两层状态估计：第一层是 9 维恒加速度
Kalman，负责短时深度缺失和连续状态；第二层是 6 维 Student-t 鲁棒弹道
EKF，使用重力和二次空气阻力，并通过 RK4 与配置地面平面求交。

观测优先级为 `RGB-D -> 1~3帧Kalman预测 -> 单目球尺寸测距`。单目观测噪声
更大，默认只维持轨迹，不允许触发控制落点。可视化底部的 `RGB-D/PRED/MONO`
显示当前来源，随后显示速度、预计落地时间和门控原因。

若构建环境中存在 ROS2 `rclcpp` 和 `volleyball_interfaces`，视觉程序发布
`/d435/ball/observation`。消息位于 `camera_color_optical_frame`（x 向右、y 向下、
z 向前），时间戳优先使用 RealSense 帧采集时间。

轨迹状态仅用于本地诊断显示。相机到 `base_link` 的坐标变换、正式轨迹预测和
`/auto/goal_pose` 发布均由独立 `catch_controller` 完成，视觉程序不直接发布控制目标。
没有 ROS2 时，检测、测距、日志和可视化仍正常工作。

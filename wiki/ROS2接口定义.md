# ROS2 接口定义

最后确认：2026-07-13

建议创建独立接口包 `volleyball_interfaces`，NX 和 RDK 使用同一版本。所有相机数据的
`header.stamp` 均表示图像采集时刻，不是推理完成、发送或接收时刻。

## NxBallObservation.msg

话题：`/nx/ball/observation`

```text
std_msgs/Header header
uint32 source_epoch
uint32 frame_id
int32 track_id
int32 class_id
float32 detection_confidence
float32[4] bbox_left_xyxy
float32[4] bbox_right_xyxy
geometry_msgs/Point position
float64[9] position_covariance
float32 depth_m
float32 disparity_px
float32 depth_sigma_m
float32 match_confidence
int32 depth_method
bool detection_valid
bool stereo_valid
bool fallback_observation
uint64 left_device_timestamp
uint64 right_device_timestamp
int64 stereo_timestamp_delta_ns
```

约定：

- `header.frame_id=nx_left_rectified_optical_frame`，表示按当前双目标定 `R1/P1` 校正后的左目光学坐标系。
- `position` 必须是未经跨帧 Kalman 的当前帧双目三维观测。
- NX 只做同帧双目候选选择、一致性检查和反投影，不发布速度或落点。
- `source_epoch` 由 NX 观测和时间同步节点共享的 `/run/volleyball/nx_source_epoch` 提供；`catch_id` 由 RDK 控制节点生成。
- `track_id` 在同一颗球的连续观测期间保持不变；切换目标时必须分配新值。
- `fallback_observation=true` 或 `stereo_valid=false` 默认不能更新远场控制轨迹。

## D435BallObservation.msg

话题：`/d435/ball/observation`

```text
std_msgs/Header header
uint32 source_epoch
uint32 frame_id
int64 rgb_depth_timestamp_delta_ns
uint64 timestamp_uncertainty_ns
uint8 timestamp_domain
bool timestamp_mapping_valid
int32 class_id
float32 detection_confidence
float32[4] bbox_xyxy

geometry_msgs/Point position
float64[9] position_covariance
uint16 valid_depth_samples
float32 depth_spread_m
float32 mono_center_z_m

bool detection_valid
bool rgbd_valid
bool mono_valid

uint8 TIMESTAMP_UNKNOWN=0
uint8 TIMESTAMP_HARDWARE_CLOCK=1
uint8 TIMESTAMP_SYSTEM_TIME=2
uint8 TIMESTAMP_GLOBAL_TIME=3
```

约定：

- `header.frame_id=camera_color_optical_frame`。
- `source_epoch` 在 D435 视觉进程启动时随机生成，用于识别进程重启和帧号回绕。
- `position` 是对齐到彩色坐标的球心 RGB-D 三维坐标。
- `header.stamp`使用彩色帧采集时刻，`rgb_depth_timestamp_delta_ns=depth-color`。
- 实车默认只接受`timestamp_mapping_valid=true`且RGB/Depth时间差不超过2 ms的观测。
- `TIMESTAMP_HARDWARE_CLOCK`表示使用设备时钟软件映射；默认仅用于诊断，不能接管。
- `detection_valid` 可触发远近场切换；`rgbd_valid` 才能作为高质量近场测量。
- 没有有效 RGB-D 时，`position` 填 NaN，不允许用零表示无效。
- 协方差按行排列；无可靠估计时用配置的保守方差，不能全部填零。

## CatchEvent.msg

话题：`/catch/event`

```text
std_msgs/Header header
uint32 source_epoch
uint32 catch_id
uint8 event

uint8 RETURNED=1
uint8 CAUGHT=2
uint8 FAILED=3
```

控制节点只处理当前 `source_epoch + catch_id` 对应的事件。当前需求中 `RETURNED`
立即触发复位；`CAUGHT/FAILED` 保留给后续机构逻辑，也建议复位。

## TimeSyncStatus.msg

话题：`/diagnostics/time_sync`

```text
std_msgs/Header header
uint32 source_epoch
uint64 sequence
int64 offset_ns
uint64 uncertainty_ns
bool synchronized
uint8 servo_state
string clock_source

uint8 SERVO_UNKNOWN=0
uint8 SERVO_UNLOCKED=1
uint8 SERVO_LOCKED=2
uint8 SERVO_HOLDOVER=3
```

`header.stamp`是NX侧同步偏差的实际测量时刻；`offset_ns = NX时钟 - RDK时钟`。
RDK将NX图像时间减去该偏差后再查询历史odom。RDK同时检查测量年龄、接收年龄、
`source_epoch`、单调`sequence`和servo锁定状态；已退役epoch和旧序号不能在重连后重新生效。
偏差超过5 ms时逐步降低NX观测权重，超过20 ms或不确定度超过配置阈值时拒绝远场观测。

## TransportDiagnostics.msg

话题：`/diagnostics/transport`

发布5 Hz通信状态，包括NX、D435、时间同步及odom的在线/超时状态、最后接收年龄、
接收消息数、DDS deadline miss计数、应用层拒绝数和相机`source_epoch`切换计数。
其中`odom_valid`表示最近存在通过frame、采集时间和单调性检查的底盘样本，
`odom_rate_hz`是最近样本时间间隔的中位频率。
该话题使用Best Effort，只用于诊断，不参与控制判定。

## CatchState.msg

话题：`/catch/state`

```text
std_msgs/Header header
uint32 source_epoch
uint32 catch_id
uint8 state
uint8 active_source
uint8 reset_reason
bool base_arrived_latched
bool near_filter_initialized
bool goal_valid
float32 distance_to_landing_m
float32 time_to_impact_s
string detail

uint8 IDLE=0
uint8 FAR_NX=1
uint8 WAIT_D435=2
uint8 NEAR_D435=3

uint8 SOURCE_NONE=0
uint8 SOURCE_NX=1
uint8 SOURCE_D435=2

uint8 RESET_NONE=0
uint8 RESET_RETURNED=1
uint8 RESET_IMPACT_TIME=2
uint8 RESET_BALL_LOST=3
```

## 标准消息接口

| 话题 | 类型 | 发布方 | 说明 |
|---|---|---|---|
| `/odom` | `nav_msgs/Odometry` | 舵轮底盘 | 直接使用底盘里程计，当前阶段不融合D435i IMU |
| `/auto/goal_pose` | `geometry_msgs/PoseStamped` | 控制节点 | `base_link` 下当前接球目标 |
| `/auto/goal_valid` | `std_msgs/Bool` | 控制节点 | 目标是否可执行 |
| `/ball/landing` | `geometry_msgs/PointStamped` | 控制节点 | `odom` 下当前选中落点 |
| `/ball/filtered_position` | `geometry_msgs/PointStamped` | 控制节点 | `odom` 下当前活动卡尔曼滤波位置 |
| `/ball/filtered_velocity` | `geometry_msgs/Vector3Stamped` | 控制节点 | `odom` 下当前活动卡尔曼滤波速度 |
| `/ball/predicted_path` | `nav_msgs/Path` | 控制节点 | `odom` 下从当前状态到预测落点的弹道 |

`/auto/goal_pose` 只允许一个发布者。底盘必须同时检查 `goal_valid` 和目标消息年龄，
建议超过 200 ms 未更新自动失效。

## QoS 确定值

| 话题 | Reliability | History/depth | Durability | Deadline建议 |
|---|---|---:|---|---:|
| `/nx/ball/observation` | Best effort | Keep last 1 | Volatile | 50 ms |
| `/d435/ball/observation` | Best effort | Keep last 1 | Volatile | 40 ms |
| `/diagnostics/time_sync` | Reliable | Keep last 1 | Volatile | 500 ms |
| `/diagnostics/transport` | Best effort | Keep last 1 | Volatile | 无 |
| `/odom` | Best effort | Keep last 20 | Volatile | 20 ms |
| `/catch/event` | Reliable | Keep last 10 | Volatile | 无 |
| `/auto/goal_pose` | Reliable | Keep last 1 | Volatile | 100 ms |
| `/auto/goal_valid` | Reliable | Keep last 1 | Transient local | 无 |
| `/catch/state` | Reliable | Keep last 1 | Transient local | 无 |
| `/ball/landing` | Reliable | Keep last 3 | Volatile | 无 |
| `/ball/filtered_position` | Best effort | Keep last 1 | Volatile | 无 |
| `/ball/filtered_velocity` | Best effort | Keep last 1 | Volatile | 无 |
| `/ball/predicted_path` | Best effort | Keep last 1 | Volatile | 无 |

视觉数据使用 Best Effort 是为了丢弃旧帧而不是积压重传；控制事件、状态和有效性必须
可靠。订阅回调只写入无锁/短锁缓存，轨迹和控制在独立定时线程运行，网络回调中不得做
图像处理或阻塞计算。

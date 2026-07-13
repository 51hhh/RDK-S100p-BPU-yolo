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

- `header.frame_id=nx_camera_optical_frame`。
- `position` 必须是未经跨帧 Kalman 的当前帧双目三维观测。
- NX 只做同帧双目候选选择、一致性检查和反投影，不发布速度或落点。
- `source_epoch` 在 NX 进程启动时随机生成；`catch_id` 由 RDK 控制节点生成。
- `track_id` 在同一颗球的连续观测期间保持不变；切换目标时必须分配新值。
- `fallback_observation=true` 或 `stereo_valid=false` 默认不能更新远场控制轨迹。

## D435BallObservation.msg

话题：`/d435/ball/observation`

```text
std_msgs/Header header
uint32 frame_id
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
```

约定：

- `header.frame_id=camera_color_optical_frame`。
- `position` 是对齐到彩色坐标的球心 RGB-D 三维坐标。
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
| `/odom` | `nav_msgs/Odometry` | 底盘 | 高频精准里程计 |
| `/auto/goal_pose` | `geometry_msgs/PoseStamped` | 控制节点 | `base_link` 下当前接球目标 |
| `/auto/goal_valid` | `std_msgs/Bool` | 控制节点 | 目标是否可执行 |
| `/ball/landing` | `geometry_msgs/PointStamped` | 控制节点 | `odom` 下当前选中落点 |

`/auto/goal_pose` 只允许一个发布者。底盘必须同时检查 `goal_valid` 和目标消息年龄，
建议超过 200 ms 未更新自动失效。

## QoS 确定值

| 话题 | Reliability | History/depth | Durability | Deadline建议 |
|---|---|---:|---|---:|
| `/nx/ball/observation` | Best effort | Keep last 1 | Volatile | 50 ms |
| `/d435/ball/observation` | Best effort | Keep last 1 | Volatile | 40 ms |
| `/odom` | Best effort | Keep last 20 | Volatile | 20 ms |
| `/catch/event` | Reliable | Keep last 10 | Volatile | 无 |
| `/auto/goal_pose` | Reliable | Keep last 1 | Volatile | 100 ms |
| `/auto/goal_valid` | Reliable | Keep last 1 | Transient local | 无 |
| `/catch/state` | Reliable | Keep last 1 | Transient local | 无 |
| `/ball/landing` | Reliable | Keep last 3 | Volatile | 无 |

视觉数据使用 Best Effort 是为了丢弃旧帧而不是积压重传；控制事件、状态和有效性必须
可靠。订阅回调只写入无锁/短锁缓存，轨迹和控制在独立定时线程运行，网络回调中不得做
图像处理或阻塞计算。

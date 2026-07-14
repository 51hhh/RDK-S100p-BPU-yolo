# RDK S100p BPU YOLO 推理框架

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-RDK%20S100p-orange.svg)]()

在 **D-Robotics 地瓜派 RDK S100p** 开发板上，利用 **BPU（神经网络处理单元）** 硬件加速，实现 YOLOv8 / YOLOv11 系列目标检测模型的高性能实时推理。

## 特性

- 基于 D-Robotics UCP（统一计算平台）API，充分利用 BPU 硬件加速
- 支持 **YOLOv8** 与 **YOLOv11** 系列模型，包含多种自定义注意力机制变体
- 完整的推理 Pipeline：Letterbox 前处理 → BPU 推理 → DFL 解码 + NMS 后处理
- USB 摄像头实时目标检测，支持自动探测摄像头设备
- Intel RealSense D435i 彩色/深度同步取流，深度对齐到彩色检测框
- 球面多环深度采样、局部中值与 Median/MAD 离群点剔除
- 基于相机畸变参数和已知球直径的单目 BBox 测距
- NV12 色彩空间硬件友好输入格式，INT8 量化模型推理
- 内置详细性能计时统计（取流 / 前处理 / 推理 / 后处理 / 绘制）
- 包含 HBM 模型诊断工具 `test_hbm`，辅助调试和验证
- Apache 2.0 开源许可证

## 项目结构

```
RDK-S100p-BPU-yolo/
├── yolov8/                         # 核心推理模块
│   ├── src/
│   │   ├── ultralytics_yolo.hpp    # YOLO 推理类接口
│   │   ├── ultralytics_yolo.cc     # YOLO 推理实现（模型加载/前处理/推理/后处理）
│   │   └── main.cc                 # USB 摄像头实时推理主程序
│   ├── utils/
│   │   ├── inc/                    # 头文件
│   │   │   ├── common_utils.hpp    # 公共数据结构、绘制函数、IO 工具
│   │   │   ├── preprocess_utils.hpp
│   │   │   ├── postprocess_utils.hpp
│   │   │   ├── draw_utils.hpp
│   │   │   └── multimedia_utils.hpp
│   │   └── src/                    # 实现文件
│   │       ├── common_utils.cc     # 图像加载、标签解析、检测框绘制
│   │       ├── preprocess_utils.cc # Letterbox 缩放、BGR→NV12 转换、张量对齐
│   │       ├── postprocess_utils.cc# DFL 解码、NMS、坐标逆映射、反量化
│   │       └── multimedia_utils.cc # 硬件显示叠加层支持
│   ├── model/                      # 预训练量化模型
│   │   ├── best.hbm                # YOLOv8/v11 BPU 量化模型
│   │   ├── bestall.hbm             # 增强版模型
│   │   ├── classes.names           # COCO 80 类标签
│   │   ├── yolo11_BPU/             # YOLOv11 标准 BPU 优化版
│   │   ├── yolo11_C2PSAtoCoordAtt/ # CoordAtt 注意力机制版
│   │   ├── yolo11_C2PSAtoCBAM-RGB/ # CBAM 注意力机制版
│   │   ├── yolo11_C2PSAtoC3k2-RGB/ # C3k2 深度卷积版
│   │   └── yolo11_C2PSAtoC3k2_DWConvblock_CBAM-RGB/  # DWConv+CBAM 融合版
│   └── CMakeLists.txt
├── example/                        # YOLOv11 兼容示例
│   ├── ultralytics_yolo11.hpp
│   ├── ultralytics_yolo11.cc
│   └── main.cc
├── test_hbm/                       # HBM 模型诊断工具
│   ├── test_hbm_info.cc
│   ├── CMakeLists.txt
│   └── README.md
└── example.md                      # 详细 API 教程文档
```

## 推理 Pipeline

```
输入 (USB摄像头 1920x1080 BGR)
  │
  ▼
┌────────────────────────────────┐
│  前处理 (CPU)                  │
│  Letterbox 等比缩放 → 640x640  │
│  BGR → I420 → NV12 色彩转换    │
│  张量内存对齐 + 缓存刷新       │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│  BPU 推理                      │
│  hbDNNInferV2 创建任务         │
│  hbUCPSubmitTask 提交至 BPU    │
│  hbUCPWaitTaskDone 等待完成    │
│  输出: 6 个张量 (3尺度×2类型)  │
└──────────────┬─────────────────┘
               ▼
┌────────────────────────────────┐
│  后处理 (CPU)                  │
│  DFL 解码 (16-bin Softmax)     │
│  多尺度融合 (stride 8/16/32)   │
│  置信度过滤 (Logit 空间比较)   │
│  类别级 NMS (IoU 阈值抑制)     │
│  Letterbox 坐标逆映射          │
└──────────────┬─────────────────┘
               ▼
输出 (检测框 + 类别 + 置信度 → 可视化显示)
```

## 环境要求

### 硬件

- **D-Robotics RDK S100p 开发板**（必需）
- USB 摄像头（用于实时推理，可选）

### 软件

| 依赖 | 说明 |
|------|------|
| RDK UCP SDK | `libdnn.so`、`libhbucp.so`（`/usr/hobot/lib`） |
| OpenCV | >= 4.0 |
| librealsense2 | D435i RGB-D同步、深度对齐和相机内参读取 |
| CMake | >= 3.0 |
| OpenMP | 并行计算支持 |
| gflags | 命令行参数解析 |
| fmt | 格式化输出库 |
| GCC | >= 7.0（C++17 支持） |

## 快速开始

RDK S100P + D435i + YOLO26 的完整 RGB-D 启动、配置、可视化和深度图说明见：

- [RGB-D 使用说明](docs/RGBD_USAGE.md)
- [D435i 近距离抛球录制与接球能力分析](docs/D435_THROW_VALIDATION.md)
- [远近场接球系统 Wiki](wiki/Home.md)

RDK ROS2 接口包和独立控制节点位于 `ros2_ws/src/`。D435i视觉进程发布
`/d435/ball/observation`和`/camera/camera/imu`，控制节点是`/auto/goal_pose`的唯一发布者。

### 1. 克隆仓库

```bash
git clone https://github.com/51hhh/RDK-S100p-BPU-yolo.git
cd RDK-S100p-BPU-yolo
```

### 2. 编译

```bash
cd yolov8
mkdir -p build && cd build
cmake ..
make -j4
```

编译产物：
- `yolov8_usb_camera` — USB 摄像头实时检测主程序
- `ultralytics_yolo11_example` — YOLOv11 示例程序（`example/` 目录存在时自动编译）

### 3. 运行

**USB 摄像头实时推理（自动探测摄像头）：**

```bash
./yolov8_usb_camera \
    --config=../config/yolo26_d435i_848x480_60fps.yaml \
    --model_path=../model/best.hbm \
    --label_file=../model/classes.names
```

**完整参数：**

```bash
./yolov8_usb_camera \
    --model_path=../model/best.hbm \
    --label_file=../model/classes.names \
    --video_device=/dev/video0 \
    --score_thres=0.25 \
    --nms_thres=0.45 \
    --camera_width=1920 \
    --camera_height=1080 \
    --camera_fps=30 \
    --display=true \
    --fullscreen=false \
    --display_width=1280 \
    --display_height=720
```

**保存实时可视化结果（无显示器/远程调试时使用）：**

```bash
./yolov8_usb_camera \
    --model_path=../model/best.hbm \
    --label_file=../model/classes.names \
    --display=false \
    --save_video=./runs/vis.avi \
    --snapshot_dir=./runs/snapshots \
    --snapshot_interval=30
```

### 4. 模型诊断工具（可选）

```bash
cd test_hbm
mkdir -p build && cd build
cmake .. && make
./test_hbm_info ../../yolov8/model/best.hbm --all
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `../model/best.hbm` | HBM 量化模型路径 |
| `--label_file` | `../model/classes.names` | 类别标签文件（每行一个类别名） |
| `--video_device` | 自动探测 | USB 摄像头设备路径（如 `/dev/video0`） |
| `--score_thres` | `0.25` | 置信度阈值（0.0 ~ 1.0，越高检测越严格） |
| `--nms_thres` | `0.45` | NMS IoU 阈值（0.0 ~ 1.0，越低去重越严格） |
| `--camera_width` | `1920` | 摄像头采集宽度 |
| `--camera_height` | `1080` | 摄像头采集高度 |
| `--camera_fps` | `30` | 摄像头采集帧率 |
| `--display` | `true` | 是否显示实时检测窗口 |
| `--fullscreen` | `true` | 全屏显示 |
| `--display_width` | `1280` | 非全屏窗口最大宽度 |
| `--display_height` | `720` | 非全屏窗口最大高度 |
| `--save_video` | 空 | 保存带检测框的视频路径 |
| `--snapshot_dir` | 空 | 定时保存可视化帧的目录 |
| `--snapshot_interval` | `0` | 每隔多少帧保存一张可视化帧，0 表示不保存 |

## 预训练模型

项目包含多个基于 YOLOv11 的 BPU 量化模型变体，采用不同的注意力机制和网络结构优化：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| `best.hbm` | 基础版本 | 通用目标检测 |
| `bestall.hbm` | 增强版本 | 精度要求较高的场景 |
| `yolo11_BPU/` | YOLOv11 标准 BPU 优化 | 通用检测 |
| `yolo11_C2PSAtoCoordAtt/` | CoordAtt 坐标注意力 | 空间位置敏感场景 |
| `yolo11_C2PSAtoCBAM-RGB/` | CBAM 通道+空间注意力 | 精度优先场景 |
| `yolo11_C2PSAtoC3k2-RGB/` | C3k2 深度卷积 | 速度与精度平衡 |
| `yolo11_C2PSAtoC3k2_DWConvblock_CBAM-RGB/` | DWConv + CBAM 融合 | 轻量化高精度 |

所有模型支持 COCO 80 类目标检测，输入分辨率 640x640。

## 技术细节

### DFL 解码

YOLOv8/v11 使用 DFL（Distribution Focal Loss）进行边界框回归，将每条边的偏移编码为 16 个 bin 的概率分布，通过 Softmax 归一化后取期望值，相比直接回归具有更好的数值稳定性和检测精度。

### 量化推理

模型采用 INT8 量化部署在 BPU 上，分类输出通常为浮点格式，边框输出为 INT32 量化格式（per-channel scale），后处理中自动检测并完成反量化。

### 内存管理

- 使用 `hbUCPMallocCached` 分配 CPU-BPU 共享缓存内存
- 严格处理硬件要求的字节对齐（32/64 字节）
- 通过 `hbUCPMemFlush` 管理 CPU-BPU 缓存一致性
- RAII 模式确保资源正确释放

### 性能统计

主程序每 30 帧输出一次详细的性能统计：

```
[取流]   USB摄像头取流:              xx.xx ms
[预处理] Letterbox+NV12转换+内存拷贝: xx.xx ms
[推理]   BPU推理:                    xx.xx ms
[后处理] DFL解码+NMS+内存拷贝:        xx.xx ms
[绘制]   目标框绘制:                  xx.xx ms
[流水线] 预处理+推理+后处理:          xx.xx ms (XX.X FPS)
[端到端] 取流到处理完成延迟:          xx.xx ms
```

## 自定义模型部署

如需部署自己训练的 YOLO 模型：

1. **导出 ONNX**：使用 Ultralytics 导出 YOLOv8/v11 的 ONNX 模型
2. **量化编译**：使用 D-Robotics Horizon 工具链将 ONNX 转换为 HBM 格式
3. **修改标签**：编辑 `classes.names` 文件，每行一个类别名
4. **运行推理**：指定新模型路径即可

```bash
./yolov8_usb_camera \
    --model_path=path/to/your_model.hbm \
    --label_file=path/to/your_classes.names \
    --score_thres=0.3
```

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| 模型加载失败 | 检查 `.hbm` 文件路径和读取权限 |
| 摄像头找不到 | 确认设备已连接：`ls /dev/video*`，检查权限 |
| 帧率过低 | 降低摄像头分辨率或帧率参数 |
| 无检测结果 | 降低 `--score_thres`（如 0.1），确认模型与标签匹配 |
| 编译找不到 SDK | 确认 RDK UCP SDK 已安装至 `/usr/hobot/` |

## 致谢

- [D-Robotics](https://developer.d-robotics.cc/) — RDK 硬件平台与 UCP SDK
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO 系列模型

## 许可证

本项目基于 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 开源。

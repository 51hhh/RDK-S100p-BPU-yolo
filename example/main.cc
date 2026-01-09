/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>
#include <chrono>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo11.hpp"
#include "common_utils.hpp"

DEFINE_string(video_device, "", "Path to video device (e.g. /dev/video0)");
DEFINE_string(model_path, "../model/best.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(label_file, "../model/classes.names",
              "Path to load ImageNet label mapping file.");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.45, "IoU threshold for Non-Maximum Suppression.");

namespace fs = std::filesystem;

/**
 * @brief 快速探测 /dev/video* 节点是否可以作为摄像头打开。
 * @param[in] device_path 视频设备的文件系统路径 (例如 "/dev/video0")。
 * @return bool           如果设备可以作为摄像头打开则返回 true，否则返回 false。
 */
bool is_usb_camera(const std::string& device_path) {
    cv::VideoCapture cap(device_path, cv::CAP_V4L2);     // 尝试 V4L2 后端
    if (!cap.isOpened()) {
        return false;
    }
    cap.release();                                       // 立即关闭
    return true;
}

/**
 * @brief 在 /dev/video* 下查找第一个可打开的 USB 摄像头。
 * @return std::string  成功时返回第一个匹配的设备路径，失败时返回空字符串。
 */
std::string find_first_usb_camera() {
    std::regex video_regex("^video[0-9]+$");             // 匹配 "video0", "video1", ...
    for (const auto& entry : fs::directory_iterator("/dev")) {
        if (std::regex_match(entry.path().filename().string(), video_regex)) {
            std::string device_path = entry.path().string();
            if (is_usb_camera(device_path)) {
                return device_path;
            }
        }
    }
    return "";
}

/**
 * @brief 在 USB 摄像头上运行 YOLOv11 目标检测的主入口。
 *
 * 流程:
 * 1) 解析命令行标志
 * 2) 打开 USB 摄像头
 * 3) 加载量化的 YOLOv11 模型
 * 4) 循环:
 *    a) 获取帧
 *    b) 预处理
 *    c) 推理
 *    d) 后处理
 *    e) 可视化
 *
 * @param argc [in] 命令行参数的数量。
 * @param argv [in] 命令行参数字符串数组。
 * @return int  [out] 进程退出代码 (0 表示成功)。
 */
int main(int argc, char **argv)
{
    // 解析命令行参数
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    std::string video_device = FLAGS_video_device;

    // 如果未指定，自动检测第一个可用的摄像头
    if (video_device.empty()) {
        video_device = find_first_usb_camera();
        if (video_device.empty()) {
            std::cerr << "No USB camera found under /dev/video*.\n";
            return 1;
        }
        std::cout << "Auto-detected USB camera: " << video_device << "\n";
    }

    // 打开摄像头 (首选默认后端; V4L2 已探测)
    cv::VideoCapture cap(video_device);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video device: " << video_device << std::endl;
        return -1;
    }
    std::cout << "Open USB camera successfully" << std::endl;

    // 配置捕获属性
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); // 尝试 MJPG 以获得更高 FPS
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    std::cout << "Place the mouse in the display window and press 'q' to quit" << std::endl;
    cv::namedWindow("YOLOv11 Detection", cv::WND_PROP_FULLSCREEN);
    cv::setWindowProperty("YOLOv11 Detection", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // 步骤 1: 加载模型并创建运行时封装
    YOLO11 yolo11 = YOLO11(FLAGS_model_path);

    // 步骤 2: 加载类别名称
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);

    double total_time_ms = 0.0;
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {                 // 获取下一帧
            std::cerr << "Failed to get image from USB camera" << std::endl;
            break;
        }

        int img_w = frame.cols;                                  // 缓存原始尺寸
        int img_h = frame.rows;

        auto start = std::chrono::high_resolution_clock::now();
        
        // 预处理 -> NV12 张量 (Letterbox 缩放到模型输入尺寸)
        yolo11.pre_process(frame);

        // 步骤 3: 在 BPU 上运行推理
        yolo11.infer();
        
        // 步骤 4: 解码和过滤预测，NMS，并重新缩放到原始尺寸
        auto results = yolo11.post_process(FLAGS_score_thres, FLAGS_nms_thres, img_w, img_h);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        total_time_ms += duration;
        frame_count++;

        if (frame_count % 10 == 0) {
            double avg_time = total_time_ms / 10.0;
            std::cout << "Average processing time (last 10 frames): " << avg_time << " ms (" << 1000.0 / avg_time << " FPS)" << std::endl;
            total_time_ms = 0.0;
        }

        // 步骤 5: 在图像上绘制检测结果
        draw_boxes(frame, results, class_names, rdk_colors);

        // 显示当前帧
        cv::imshow("YOLOv11 Detection", frame);

        // 按 'q' 退出
        int key = cv::waitKey(1);
        if ((key & 0xFF) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

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

/**
 * @file main.cc
 * @brief USB摄像头 YOLOv8/YOLOv11 实时目标检测示例
 * 
 * 本示例展示如何在地瓜派(RDK)开发板上使用USB摄像头进行实时目标检测：
 * 1. 自动探测并打开USB摄像头
 * 2. 加载BPU量化模型(.hbm)
 * 3. 实时推理并显示检测结果
 * 
 * 使用方法:
 *   ./yolov8_usb_camera --model_path=model/yolov8n.hbm --label_file=model/coco.names
 *   ./yolov8_usb_camera --video_device=/dev/video0  # 指定摄像头设备
 */

#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>
#include <chrono>
#include <iomanip>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>

#include "ultralytics_yolo.hpp"
#include "common_utils.hpp"

// ==================== 命令行参数定义 ====================
DEFINE_string(video_device, "", "USB摄像头设备路径 (例如 /dev/video0)，留空则自动检测");
DEFINE_string(model_path, "../model/best.hbm",
              "BPU量化模型文件路径 (*.hbm)");
DEFINE_string(label_file, "../model/classes.names",
              "类别标签文件路径，每行一个类别名");
DEFINE_double(score_thres, 0.25, "置信度阈值，过滤低于此分数的检测结果");
DEFINE_double(nms_thres, 0.45, "NMS的IoU阈值，用于去除重叠框");
DEFINE_int32(camera_width, 1920, "摄像头采集宽度");
DEFINE_int32(camera_height, 1080, "摄像头采集高度");
DEFINE_int32(camera_fps, 30, "摄像头帧率");
DEFINE_bool(fullscreen, true, "是否全屏显示");

namespace fs = std::filesystem;

/**
 * @brief 探测指定的视频设备是否可以作为摄像头打开
 * @param[in] device_path 视频设备的文件系统路径 (例如 "/dev/video0")
 * @return bool 如果设备可以作为摄像头打开则返回 true，否则返回 false
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
 * @brief 在 /dev/video* 下查找第一个可用的USB摄像头
 * @return std::string 成功时返回第一个匹配的设备路径，失败时返回空字符串
 */
std::string find_first_usb_camera() {
    std::regex video_regex("^video[0-9]+$");             // 匹配 "video0", "video1", ...
    
    std::vector<std::string> video_devices;
    for (const auto& entry : fs::directory_iterator("/dev")) {
        if (std::regex_match(entry.path().filename().string(), video_regex)) {
            video_devices.push_back(entry.path().string());
        }
    }
    
    // 排序以确保优先尝试 video0, video1...
    std::sort(video_devices.begin(), video_devices.end());
    
    for (const auto& device_path : video_devices) {
        if (is_usb_camera(device_path)) {
            return device_path;
        }
    }
    return "";
}

/**
 * @brief 打印使用帮助信息
 */
void print_usage() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "USB摄像头 YOLO 目标检测 - RDK BPU 推理" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n使用方法:" << std::endl;
    std::cout << "  ./yolov8_usb_camera [options]" << std::endl;
    std::cout << "\n常用参数:" << std::endl;
    std::cout << "  --model_path      模型文件路径 (.hbm)" << std::endl;
    std::cout << "  --label_file      类别标签文件路径" << std::endl;
    std::cout << "  --video_device    摄像头设备 (留空自动检测)" << std::endl;
    std::cout << "  --score_thres     置信度阈值 (默认: 0.25)" << std::endl;
    std::cout << "  --nms_thres       NMS阈值 (默认: 0.45)" << std::endl;
    std::cout << "  --camera_width    采集宽度 (默认: 1920)" << std::endl;
    std::cout << "  --camera_height   采集高度 (默认: 1080)" << std::endl;
    std::cout << "  --fullscreen      全屏显示 (默认: true)" << std::endl;
    std::cout << "\n操作说明:" << std::endl;
    std::cout << "  按 'q' 键退出程序" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

/**
 * @brief 主函数 - USB摄像头YOLO目标检测
 *
 * 流程:
 * 1) 解析命令行参数
 * 2) 打开USB摄像头
 * 3) 加载量化的YOLO模型
 * 4) 主循环:
 *    a) 获取帧
 *    b) 预处理 (Letterbox + BGR->NV12)
 *    c) BPU推理
 *    d) 后处理 (DFL解码 + NMS)
 *    e) 绘制检测框并显示
 *
 * @param argc [in] 命令行参数数量
 * @param argv [in] 命令行参数数组
 * @return int 进程退出代码 (0表示成功)
 */
int main(int argc, char **argv)
{
    // 解析命令行参数
    gflags::SetUsageMessage("USB摄像头 YOLO 目标检测 - RDK BPU 推理示例");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    print_usage();
    std::cout << "启动参数: " << gflags::GetArgv() << std::endl;

    std::string video_device = FLAGS_video_device;

    // ==================== Step 1: 打开摄像头 ====================
    if (video_device.empty()) {
        std::cout << "[INFO] 未指定摄像头设备，正在自动检测..." << std::endl;
        video_device = find_first_usb_camera();
        if (video_device.empty()) {
            std::cerr << "[ERROR] 未找到可用的USB摄像头 (/dev/video*)" << std::endl;
            std::cerr << "[提示] 请检查摄像头是否正确连接，或使用 --video_device 手动指定" << std::endl;
            return 1;
        }
        std::cout << "[INFO] 自动检测到摄像头: " << video_device << std::endl;
    }

    cv::VideoCapture cap(video_device);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] 无法打开摄像头设备: " << video_device << std::endl;
        return -1;
    }
    std::cout << "[INFO] 摄像头打开成功: " << video_device << std::endl;

    // 配置摄像头参数
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); // MJPG格式以获得更高帧率
    cap.set(cv::CAP_PROP_FPS, FLAGS_camera_fps);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FLAGS_camera_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FLAGS_camera_height);
    
    // 打印实际摄像头参数
    std::cout << "[INFO] 摄像头配置:" << std::endl;
    std::cout << "       分辨率: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "       帧率: " << cap.get(cv::CAP_PROP_FPS) << " FPS" << std::endl;

    // ==================== Step 2: 加载模型 ====================
    std::cout << "[INFO] 正在加载模型: " << FLAGS_model_path << std::endl;
    UltralyticsYOLO yolo(FLAGS_model_path);

    // ==================== Step 3: 加载类别标签 ====================
    std::cout << "[INFO] 正在加载类别标签: " << FLAGS_label_file << std::endl;
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    std::cout << "[INFO] 已加载 " << class_names.size() << " 个类别" << std::endl;

    // ==================== Step 4: 创建显示窗口 ====================
    const std::string window_name = "YOLO Detection - RDK BPU";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    if (FLAGS_fullscreen) {
        cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }
    std::cout << "\n[INFO] 按 'q' 键退出程序\n" << std::endl;

    // ==================== Step 5: 主推理循环 ====================
    // 性能统计变量
    double total_capture_ms = 0.0;      // USB取流累计时间
    double total_preprocess_ms = 0.0;   // 预处理累计时间
    double total_infer_ms = 0.0;        // BPU推理累计时间
    double total_postprocess_ms = 0.0;  // 后处理累计时间
    double total_draw_ms = 0.0;         // 绘制累计时间
    double total_e2e_ms = 0.0;          // 端到端延迟累计时间
    int frame_count = 0;
    cv::Mat frame;
    
    auto loop_start = std::chrono::high_resolution_clock::now();

    while (true) {
        // 端到端计时起点 (从取流开始)
        auto e2e_start = std::chrono::high_resolution_clock::now();
        
        // 5.1 获取一帧图像 (USB取流)
        auto capture_start = std::chrono::high_resolution_clock::now();
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[WARN] 获取图像帧失败，尝试重新获取..." << std::endl;
            continue;
        }
        auto capture_end = std::chrono::high_resolution_clock::now();
        double capture_ms = std::chrono::duration<double, std::milli>(capture_end - capture_start).count();

        int img_w = frame.cols;
        int img_h = frame.rows;

        // 5.2 预处理: Letterbox缩放 + BGR转NV12 (CPU -> BPU内存拷贝在此阶段)
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        yolo.pre_process(frame);
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        double preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

        // 5.3 BPU推理
        auto infer_start = std::chrono::high_resolution_clock::now();
        yolo.infer();
        auto infer_end = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

        // 5.4 后处理: DFL解码 + NMS + 坐标还原 (BPU -> CPU内存拷贝在此阶段)
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        auto results = yolo.post_process(FLAGS_score_thres, FLAGS_nms_thres, img_w, img_h);
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        double postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

        // 5.5 在图像上绘制检测结果
        auto draw_start = std::chrono::high_resolution_clock::now();
        draw_boxes(frame, results, class_names, rdk_colors);
        auto draw_end = std::chrono::high_resolution_clock::now();
        double draw_ms = std::chrono::duration<double, std::milli>(draw_end - draw_start).count();

        // 端到端计时终点 (处理完成，不含显示)
        auto e2e_end = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();

        // 累计统计
        total_capture_ms += capture_ms;
        total_preprocess_ms += preprocess_ms;
        total_infer_ms += infer_ms;
        total_postprocess_ms += postprocess_ms;
        total_draw_ms += draw_ms;
        total_e2e_ms += e2e_ms;
        frame_count++;

        // 推理流水线时间 (预处理+推理+后处理)
        double pipeline_ms = preprocess_ms + infer_ms + postprocess_ms;

        // 每30帧打印一次详细性能统计
        if (frame_count % 30 == 0) {
            auto loop_now = std::chrono::high_resolution_clock::now();
            double loop_elapsed = std::chrono::duration<double, std::milli>(loop_now - loop_start).count();
            double actual_fps = 30.0 * 1000.0 / loop_elapsed;  // 实际帧率
            loop_start = loop_now;
            
            std::cout << "\n==================== 性能统计 (30帧平均) ====================" << std::endl;
            std::cout << "[取流]   USB摄像头取流: " << std::fixed << std::setprecision(2) 
                      << total_capture_ms / 30.0 << " ms" << std::endl;
            std::cout << "[预处理] Letterbox+NV12转换+内存拷贝: " << total_preprocess_ms / 30.0 << " ms" << std::endl;
            std::cout << "[推理]   BPU推理: " << total_infer_ms / 30.0 << " ms" << std::endl;
            std::cout << "[后处理] DFL解码+NMS+内存拷贝: " << total_postprocess_ms / 30.0 << " ms" << std::endl;
            std::cout << "[绘制]   目标框绘制: " << total_draw_ms / 30.0 << " ms" << std::endl;
            std::cout << "-------------------------------------------------------------" << std::endl;
            std::cout << "[流水线] 预处理+推理+后处理: " << (total_preprocess_ms + total_infer_ms + total_postprocess_ms) / 30.0 
                      << " ms (" << 30000.0 / (total_preprocess_ms + total_infer_ms + total_postprocess_ms) << " FPS理论)" << std::endl;
            std::cout << "[端到端] 取流到处理完成延迟: " << total_e2e_ms / 30.0 << " ms" << std::endl;
            std::cout << "[实际]   真实帧率: " << actual_fps << " FPS, 检测到 " << results.size() << " 个目标" << std::endl;
            std::cout << "=============================================================\n" << std::endl;
            
            // 重置累计
            total_capture_ms = 0.0;
            total_preprocess_ms = 0.0;
            total_infer_ms = 0.0;
            total_postprocess_ms = 0.0;
            total_draw_ms = 0.0;
            total_e2e_ms = 0.0;
        }

        // 5.6 显示FPS信息 (显示实时流水线FPS和端到端延迟)
        double pipeline_fps = 1000.0 / (pipeline_ms > 0 ? pipeline_ms : 1);
        std::string fps_text = "Pipeline: " + std::to_string(static_cast<int>(pipeline_fps)) + " FPS";
        std::string latency_text = "E2E: " + std::to_string(static_cast<int>(e2e_ms)) + " ms";
        cv::putText(frame, fps_text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, latency_text, cv::Point(10, 70), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // 5.7 显示当前帧
        cv::imshow(window_name, frame);

        // 5.8 检测按键，按 'q' 退出
        int key = cv::waitKey(1);
        if ((key & 0xFF) == 'q' || (key & 0xFF) == 'Q') {
            std::cout << "\n[INFO] 用户按下 'q' 键，正在退出..." << std::endl;
            break;
        }
    }

    // ==================== Step 6: 释放资源 ====================
    std::cout << "[INFO] 正在释放资源..." << std::endl;
    cap.release();
    cv::destroyAllWindows();

    std::cout << "[INFO] 程序正常退出" << std::endl;
    return 0;
}

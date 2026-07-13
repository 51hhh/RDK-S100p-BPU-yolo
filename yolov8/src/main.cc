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
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "gflags/gflags.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "ultralytics_yolo.hpp"
#include "common_utils.hpp"
#include "rgbd_utils.hpp"
#include "trajectory_utils.hpp"
#include "d435_observation_publisher.hpp"

// ==================== 命令行参数定义 ====================
DEFINE_string(config, "../config/yolo26_d435i_848x480_60fps.yaml",
              "运行配置YAML；命令行显式参数优先于配置文件");
DEFINE_string(video_device, "", "USB摄像头设备路径 (例如 /dev/video0)，留空则自动检测");
DEFINE_string(model_path, "../model/best.hbm",
              "BPU量化模型文件路径 (*.hbm)");
DEFINE_string(label_file, "../model/classes.names",
              "类别标签文件路径，每行一个类别名");
DEFINE_double(score_thres, 0.25, "置信度阈值，过滤低于此分数的检测结果");
DEFINE_double(nms_thres, 0.45, "NMS的IoU阈值，用于去除重叠框");
DEFINE_int32(camera_width, 848, "摄像头采集宽度");
DEFINE_int32(camera_height, 480, "摄像头采集高度");
DEFINE_int32(camera_fps, 60, "摄像头帧率");
DEFINE_string(camera_fourcc, "YUYV", "摄像头像素格式，四字符FOURCC");
DEFINE_bool(camera_auto_exposure, false, "是否启用摄像头自动曝光");
DEFINE_int32(camera_exposure_us, 15000, "手动曝光时间，单位微秒");
DEFINE_int32(camera_gain, 128, "摄像头固定增益（D435i最大值为128）");
DEFINE_bool(camera_dynamic_framerate, false, "是否允许曝光控制动态降低帧率");
DEFINE_int32(camera_power_line_frequency, 1,
             "防闪烁频率：0关闭，1=50Hz，2=60Hz，3=自动");
DEFINE_bool(camera_auto_white_balance, false, "是否启用自动白平衡");
DEFINE_int32(camera_white_balance_temperature, 4600, "固定白平衡色温，单位K");
DEFINE_int32(camera_buffer_count, 1, "OpenCV/V4L2采集缓冲数量");
DEFINE_int32(camera_warmup_frames, 120, "正式统计前丢弃的摄像头预热帧数");
DEFINE_bool(strict_camera_config, true, "摄像头参数设置或读回不一致时退出");
DEFINE_bool(depth_enabled, true, "使用librealsense同步获取并对齐D435i彩色/深度帧");
DEFINE_string(realsense_serial, "035422071550", "librealsense设备序列号，留空选择第一台设备");
DEFINE_double(object_diameter_m, 0.215, "已知球体直径，单位米");
DEFINE_double(mono_bbox_scale, 0.95, "检测框直径相对真实投影直径的补偿系数");
DEFINE_double(depth_min_m, 0.20, "有效深度最小值，单位米");
DEFINE_double(depth_max_m, 12.0, "有效深度最大值，单位米");
DEFINE_double(depth_sample_radius_scale, 0.68, "球框内深度采样椭圆半径比例");
DEFINE_int32(depth_sample_rings, 3, "球表面同心采样环数量");
DEFINE_int32(depth_samples_per_ring, 12, "每个采样环的采样点数量");
DEFINE_int32(depth_patch_radius, 1, "每个采样点局部中值窗口半径");
DEFINE_int32(depth_min_samples, 6, "鲁棒球心深度所需最少有效采样点");
DEFINE_double(depth_mad_scale, 3.0, "深度候选MAD离群值门限倍数");
DEFINE_double(depth_max_spread_m, 0.12, "球心深度候选最大鲁棒标准差");
DEFINE_double(depth_mono_consistency_relative, 0.45, "RGB-D候选相对单目距离的最大偏差比例");
DEFINE_double(depth_mono_consistency_absolute_m, 0.75, "RGB-D候选相对单目距离的最小绝对门限");
DEFINE_bool(depth_spatial_filter, true, "启用RealSense空间深度滤波");
DEFINE_bool(depth_temporal_filter, false, "启用RealSense时域深度滤波（高速运动默认关闭）");
DEFINE_bool(depth_hole_filling_filter, false, "启用RealSense深度孔洞填充");
DEFINE_bool(depth_draw_samples, true, "在彩色图上绘制通过鲁棒筛选的深度采样点");
DEFINE_bool(depth_show_colormap, false, "显示对齐到彩色图的深度伪彩窗口");
DEFINE_bool(display, true, "是否显示实时检测窗口");
DEFINE_bool(headless, false, "无图形界面兼容参数；true时强制关闭显示");
DEFINE_bool(fullscreen, true, "是否全屏显示");
DEFINE_int32(display_width, 1280, "非全屏显示窗口最大宽度");
DEFINE_int32(display_height, 720, "非全屏显示窗口最大高度");
DEFINE_string(save_video, "", "保存带检测框的视频路径，留空则不保存");
DEFINE_string(snapshot_dir, "", "定时保存可视化帧的目录，留空则不保存");
DEFINE_int32(snapshot_interval, 0, "每隔多少帧保存一张可视化帧，0表示不保存");
DEFINE_bool(tracking_enabled, true, "启用球轨迹滤波");
DEFINE_int32(tracking_min_init_frames, 3, "第一层滤波初始化帧数");
DEFINE_int32(tracking_max_lost_frames, 8, "轨迹最大丢失帧数");
DEFINE_double(tracking_process_accel, 30.0, "第一层过程加速度噪声");
DEFINE_double(tracking_gate_chi2, 11.34, "三维观测卡方门限");
DEFINE_double(tracking_min_detection_confidence, 0.10, "轨迹更新最小检测置信度");
DEFINE_int32(tracking_min_depth_samples, 8, "轨迹更新最少深度采样点");
DEFINE_double(tracking_max_depth_spread, 0.10, "轨迹更新最大深度离散度");
DEFINE_double(tracking_max_rgbd_mono_delta, 0.80, "RGB-D与单目距离最大差值");
DEFINE_bool(trajectory_enabled, true, "启用本地诊断轨迹预测");
DEFINE_double(trajectory_student_t_nu, 12.0, "Student-t自由度");
DEFINE_double(trajectory_q_position, 0.0001, "轨迹位置过程噪声");
DEFINE_double(trajectory_q_velocity, 1.5, "轨迹速度过程噪声");
DEFINE_double(trajectory_gravity, 9.81, "轨迹预测重力加速度");
DEFINE_double(trajectory_ball_mass, 0.270, "球质量，单位kg");
DEFINE_double(trajectory_ball_radius, 0.1075, "球半径，单位m");
DEFINE_double(trajectory_drag_coefficient, 0.10, "球体阻力系数");
DEFINE_double(trajectory_air_density, 1.225, "空气密度");
DEFINE_int32(trajectory_min_track_frames, 6, "允许预测落点的最小轨迹帧数");
DEFINE_double(trajectory_min_speed, 0.5, "弹道预测最小速度");
DEFINE_double(trajectory_max_predict_time, 3.0, "最大轨迹预测时间");
DEFINE_double(trajectory_rk4_dt, 0.008, "RK4积分步长");
DEFINE_double(ground_normal_x, 0.0, "地面法向量x");
DEFINE_double(ground_normal_y, 1.0, "地面法向量y");
DEFINE_double(ground_normal_z, 0.0, "地面法向量z");
DEFINE_double(ground_offset_m, -0.50, "地面平面n*p+d=0的d");
DEFINE_bool(landing_gate_enabled, true, "启用本地落点质量门控");
DEFINE_double(landing_min_confidence, 0.70, "控制落点最小检测置信度");
DEFINE_double(landing_min_student_weight, 0.15, "Student-t最小权重");
DEFINE_double(landing_min_time, 0.25, "最短有效落地时间");
DEFINE_double(landing_max_time, 2.20, "最长有效落地时间");
DEFINE_double(landing_min_speed, 0.80, "最小有效球速");
DEFINE_int32(landing_stable_frames, 3, "落点稳定门控连续帧数");
DEFINE_double(landing_max_stable_jump, 0.35, "相邻落点最大跳变");
DEFINE_double(landing_max_sigma, 0.30, "最大落点位置标准差");
DEFINE_double(landing_max_abs_x, 3.6, "落点最大横向绝对值");
DEFINE_double(landing_min_depth, 0.0, "落点最小深度");
DEFINE_double(landing_max_depth, 12.0, "落点最大深度");
DEFINE_bool(landing_allow_fallback, false, "是否允许单目/预测观测发布控制落点");
DEFINE_bool(d435_observation_publish, true, "发布D435i球观测给独立控制节点");
DEFINE_string(d435_observation_topic, "/d435/ball/observation", "D435i球观测话题");
DEFINE_int32(volleyball_class_id, 0, "排球类别ID");
DEFINE_bool(depth_allow_hardware_time_fallback, false,
            "允许D435硬件时钟软件映射；实车默认要求global/system time");
DEFINE_int32(depth_timestamp_fallback_warmup_frames, 30,
             "D435硬件时钟映射稳定前的最少帧数");
DEFINE_double(depth_max_timestamp_uncertainty_s, 0.002,
              "D435硬件时钟映射最大不确定度");

namespace fs = std::filesystem;

bool flag_is_default(const char* name)
{
    gflags::CommandLineFlagInfo info;
    return gflags::GetCommandLineFlagInfo(name, &info) && info.is_default;
}

template <typename T>
void apply_yaml_value(const YAML::Node& node,
                      const char* key,
                      const char* flag_name,
                      T& value)
{
    if (node && node[key] && flag_is_default(flag_name)) {
        value = node[key].as<T>();
    }
}

bool load_runtime_config(const std::string& path)
{
    if (path.empty()) return true;

    try {
        YAML::Node root = YAML::LoadFile(path);
        YAML::Node camera = root["camera"];
        YAML::Node detector = root["detector"];
        YAML::Node depth = root["depth"];
        YAML::Node display = root["display"];
        YAML::Node recording = root["recording"];
        YAML::Node tracking = root["tracking"];
        YAML::Node trajectory = root["trajectory"];
        YAML::Node ground = root["ground"];
        YAML::Node landing = root["landing_publish"];
        YAML::Node d435_observation = root["d435_observation"];

        apply_yaml_value(camera, "device", "video_device", FLAGS_video_device);
        apply_yaml_value(camera, "width", "camera_width", FLAGS_camera_width);
        apply_yaml_value(camera, "height", "camera_height", FLAGS_camera_height);
        apply_yaml_value(camera, "fps", "camera_fps", FLAGS_camera_fps);
        apply_yaml_value(camera, "fourcc", "camera_fourcc", FLAGS_camera_fourcc);
        apply_yaml_value(camera, "auto_exposure", "camera_auto_exposure",
                         FLAGS_camera_auto_exposure);
        apply_yaml_value(camera, "exposure_us", "camera_exposure_us",
                         FLAGS_camera_exposure_us);
        apply_yaml_value(camera, "gain", "camera_gain", FLAGS_camera_gain);
        apply_yaml_value(camera, "dynamic_framerate", "camera_dynamic_framerate",
                         FLAGS_camera_dynamic_framerate);
        apply_yaml_value(camera, "power_line_frequency", "camera_power_line_frequency",
                         FLAGS_camera_power_line_frequency);
        apply_yaml_value(camera, "auto_white_balance", "camera_auto_white_balance",
                         FLAGS_camera_auto_white_balance);
        apply_yaml_value(camera, "white_balance_temperature",
                         "camera_white_balance_temperature",
                         FLAGS_camera_white_balance_temperature);
        apply_yaml_value(camera, "buffer_count", "camera_buffer_count",
                         FLAGS_camera_buffer_count);
        apply_yaml_value(camera, "warmup_frames", "camera_warmup_frames",
                         FLAGS_camera_warmup_frames);
        apply_yaml_value(camera, "strict", "strict_camera_config",
                         FLAGS_strict_camera_config);

        apply_yaml_value(detector, "model_path", "model_path", FLAGS_model_path);
        apply_yaml_value(detector, "label_file", "label_file", FLAGS_label_file);
        apply_yaml_value(detector, "score_threshold", "score_thres", FLAGS_score_thres);
        apply_yaml_value(detector, "nms_threshold", "nms_thres", FLAGS_nms_thres);

        apply_yaml_value(depth, "enabled", "depth_enabled", FLAGS_depth_enabled);
        apply_yaml_value(depth, "serial", "realsense_serial", FLAGS_realsense_serial);
        apply_yaml_value(depth, "object_diameter_m", "object_diameter_m", FLAGS_object_diameter_m);
        apply_yaml_value(depth, "bbox_scale", "mono_bbox_scale", FLAGS_mono_bbox_scale);
        apply_yaml_value(depth, "min_m", "depth_min_m", FLAGS_depth_min_m);
        apply_yaml_value(depth, "max_m", "depth_max_m", FLAGS_depth_max_m);
        apply_yaml_value(depth, "sample_radius_scale", "depth_sample_radius_scale",
                         FLAGS_depth_sample_radius_scale);
        apply_yaml_value(depth, "sample_rings", "depth_sample_rings", FLAGS_depth_sample_rings);
        apply_yaml_value(depth, "samples_per_ring", "depth_samples_per_ring",
                         FLAGS_depth_samples_per_ring);
        apply_yaml_value(depth, "patch_radius", "depth_patch_radius", FLAGS_depth_patch_radius);
        apply_yaml_value(depth, "min_samples", "depth_min_samples", FLAGS_depth_min_samples);
        apply_yaml_value(depth, "mad_scale", "depth_mad_scale", FLAGS_depth_mad_scale);
        apply_yaml_value(depth, "max_spread_m", "depth_max_spread_m",
                         FLAGS_depth_max_spread_m);
        apply_yaml_value(depth, "mono_consistency_relative",
                         "depth_mono_consistency_relative",
                         FLAGS_depth_mono_consistency_relative);
        apply_yaml_value(depth, "mono_consistency_absolute_m",
                         "depth_mono_consistency_absolute_m",
                         FLAGS_depth_mono_consistency_absolute_m);
        apply_yaml_value(depth, "spatial_filter", "depth_spatial_filter",
                         FLAGS_depth_spatial_filter);
        apply_yaml_value(depth, "temporal_filter", "depth_temporal_filter",
                         FLAGS_depth_temporal_filter);
        apply_yaml_value(depth, "hole_filling_filter", "depth_hole_filling_filter",
                         FLAGS_depth_hole_filling_filter);
        apply_yaml_value(depth, "draw_samples", "depth_draw_samples",
                         FLAGS_depth_draw_samples);
        apply_yaml_value(depth, "show_colormap", "depth_show_colormap",
                         FLAGS_depth_show_colormap);
        apply_yaml_value(depth, "allow_hardware_time_fallback",
                         "depth_allow_hardware_time_fallback",
                         FLAGS_depth_allow_hardware_time_fallback);
        apply_yaml_value(depth, "timestamp_fallback_warmup_frames",
                         "depth_timestamp_fallback_warmup_frames",
                         FLAGS_depth_timestamp_fallback_warmup_frames);
        apply_yaml_value(depth, "max_timestamp_uncertainty_s",
                         "depth_max_timestamp_uncertainty_s",
                         FLAGS_depth_max_timestamp_uncertainty_s);

        apply_yaml_value(display, "enabled", "display", FLAGS_display);
        apply_yaml_value(display, "fullscreen", "fullscreen", FLAGS_fullscreen);
        apply_yaml_value(display, "width", "display_width", FLAGS_display_width);
        apply_yaml_value(display, "height", "display_height", FLAGS_display_height);

        apply_yaml_value(recording, "video_path", "save_video", FLAGS_save_video);
        apply_yaml_value(recording, "snapshot_dir", "snapshot_dir", FLAGS_snapshot_dir);
        apply_yaml_value(recording, "snapshot_interval", "snapshot_interval",
                         FLAGS_snapshot_interval);
        apply_yaml_value(tracking, "enabled", "tracking_enabled", FLAGS_tracking_enabled);
        apply_yaml_value(tracking, "min_init_frames", "tracking_min_init_frames", FLAGS_tracking_min_init_frames);
        apply_yaml_value(tracking, "max_lost_frames", "tracking_max_lost_frames", FLAGS_tracking_max_lost_frames);
        apply_yaml_value(tracking, "process_accel_mps2", "tracking_process_accel", FLAGS_tracking_process_accel);
        apply_yaml_value(tracking, "innovation_gate_chi2", "tracking_gate_chi2", FLAGS_tracking_gate_chi2);
        apply_yaml_value(tracking, "min_detection_confidence", "tracking_min_detection_confidence", FLAGS_tracking_min_detection_confidence);
        apply_yaml_value(tracking, "min_depth_samples", "tracking_min_depth_samples", FLAGS_tracking_min_depth_samples);
        apply_yaml_value(tracking, "max_depth_spread_m", "tracking_max_depth_spread", FLAGS_tracking_max_depth_spread);
        apply_yaml_value(tracking, "max_rgbd_mono_delta_m", "tracking_max_rgbd_mono_delta", FLAGS_tracking_max_rgbd_mono_delta);
        apply_yaml_value(trajectory, "enabled", "trajectory_enabled", FLAGS_trajectory_enabled);
        apply_yaml_value(trajectory, "student_t_nu", "trajectory_student_t_nu", FLAGS_trajectory_student_t_nu);
        apply_yaml_value(trajectory, "q_position", "trajectory_q_position", FLAGS_trajectory_q_position);
        apply_yaml_value(trajectory, "q_velocity", "trajectory_q_velocity", FLAGS_trajectory_q_velocity);
        apply_yaml_value(trajectory, "gravity_mps2", "trajectory_gravity", FLAGS_trajectory_gravity);
        apply_yaml_value(trajectory, "ball_mass_kg", "trajectory_ball_mass", FLAGS_trajectory_ball_mass);
        apply_yaml_value(trajectory, "ball_radius_m", "trajectory_ball_radius", FLAGS_trajectory_ball_radius);
        apply_yaml_value(trajectory, "drag_coefficient", "trajectory_drag_coefficient", FLAGS_trajectory_drag_coefficient);
        apply_yaml_value(trajectory, "air_density", "trajectory_air_density", FLAGS_trajectory_air_density);
        apply_yaml_value(trajectory, "min_track_frames", "trajectory_min_track_frames", FLAGS_trajectory_min_track_frames);
        apply_yaml_value(trajectory, "min_speed_mps", "trajectory_min_speed", FLAGS_trajectory_min_speed);
        apply_yaml_value(trajectory, "max_predict_time_s", "trajectory_max_predict_time", FLAGS_trajectory_max_predict_time);
        apply_yaml_value(trajectory, "rk4_dt_s", "trajectory_rk4_dt", FLAGS_trajectory_rk4_dt);
        if (ground && ground["normal"] && ground["normal"].IsSequence() &&
            ground["normal"].size() == 3) {
            if (flag_is_default("ground_normal_x")) FLAGS_ground_normal_x = ground["normal"][0].as<double>();
            if (flag_is_default("ground_normal_y")) FLAGS_ground_normal_y = ground["normal"][1].as<double>();
            if (flag_is_default("ground_normal_z")) FLAGS_ground_normal_z = ground["normal"][2].as<double>();
        }
        apply_yaml_value(ground, "offset_m", "ground_offset_m", FLAGS_ground_offset_m);
        apply_yaml_value(landing, "enabled", "landing_gate_enabled", FLAGS_landing_gate_enabled);
        apply_yaml_value(landing, "min_confidence", "landing_min_confidence", FLAGS_landing_min_confidence);
        apply_yaml_value(landing, "min_student_weight", "landing_min_student_weight", FLAGS_landing_min_student_weight);
        apply_yaml_value(landing, "min_time_to_land_s", "landing_min_time", FLAGS_landing_min_time);
        apply_yaml_value(landing, "max_time_to_land_s", "landing_max_time", FLAGS_landing_max_time);
        apply_yaml_value(landing, "min_speed_mps", "landing_min_speed", FLAGS_landing_min_speed);
        apply_yaml_value(landing, "stable_frames", "landing_stable_frames", FLAGS_landing_stable_frames);
        apply_yaml_value(landing, "max_stable_jump_m", "landing_max_stable_jump", FLAGS_landing_max_stable_jump);
        apply_yaml_value(landing, "max_landing_sigma_m", "landing_max_sigma", FLAGS_landing_max_sigma);
        apply_yaml_value(landing, "max_abs_x_m", "landing_max_abs_x", FLAGS_landing_max_abs_x);
        apply_yaml_value(landing, "min_depth_m", "landing_min_depth", FLAGS_landing_min_depth);
        apply_yaml_value(landing, "max_depth_m", "landing_max_depth", FLAGS_landing_max_depth);
        apply_yaml_value(landing, "allow_fallback_observation", "landing_allow_fallback", FLAGS_landing_allow_fallback);
        apply_yaml_value(d435_observation, "enabled", "d435_observation_publish", FLAGS_d435_observation_publish);
        apply_yaml_value(d435_observation, "topic", "d435_observation_topic", FLAGS_d435_observation_topic);
        apply_yaml_value(d435_observation, "volleyball_class_id", "volleyball_class_id", FLAGS_volleyball_class_id);

        std::cout << "[INFO] 已加载配置文件: " << path << std::endl;
        return true;
    } catch (const YAML::Exception& error) {
        std::cerr << "[ERROR] 配置文件加载失败: " << path
                  << ", " << error.what() << std::endl;
        return false;
    }
}

bool validate_runtime_config()
{
    bool ok = true;
    if (FLAGS_camera_width <= 0 || FLAGS_camera_height <= 0 || FLAGS_camera_fps <= 0) {
        std::cerr << "[ERROR] 摄像头分辨率和帧率必须为正数" << std::endl;
        ok = false;
    }
    if (FLAGS_camera_fourcc.size() != 4) {
        std::cerr << "[ERROR] camera.fourcc必须正好包含4个字符" << std::endl;
        ok = false;
    }
    if (!FLAGS_camera_auto_exposure) {
        const double frame_period_us = 1000000.0 / std::max(1, FLAGS_camera_fps);
        if (FLAGS_camera_exposure_us <= 0 ||
            FLAGS_camera_exposure_us >= frame_period_us) {
            std::cerr << "[ERROR] 手动曝光 " << FLAGS_camera_exposure_us
                      << " us 必须小于单帧周期 " << frame_period_us
                      << " us" << std::endl;
            ok = false;
        }
    }
    if (FLAGS_camera_gain < 0) {
        std::cerr << "[ERROR] camera.gain不能为负数" << std::endl;
        ok = false;
    }
    if (FLAGS_object_diameter_m <= 0.0 || FLAGS_depth_min_m <= 0.0 ||
        FLAGS_depth_max_m <= FLAGS_depth_min_m) {
        std::cerr << "[ERROR] depth物体直径和有效深度范围配置无效" << std::endl;
        ok = false;
    }
    if (FLAGS_depth_sample_radius_scale <= 0.0 ||
        FLAGS_depth_sample_radius_scale >= 1.0 ||
        FLAGS_depth_sample_rings < 1 || FLAGS_depth_samples_per_ring < 4 ||
        FLAGS_depth_patch_radius < 0 || FLAGS_depth_min_samples < 1 ||
        FLAGS_depth_mad_scale <= 0.0 || FLAGS_depth_max_spread_m <= 0.0 ||
        FLAGS_depth_mono_consistency_relative <= 0.0 ||
        FLAGS_depth_mono_consistency_absolute_m <= 0.0) {
        std::cerr << "[ERROR] depth多点采样参数配置无效" << std::endl;
        ok = false;
    }
    if (FLAGS_depth_timestamp_fallback_warmup_frames < 1 ||
        FLAGS_depth_max_timestamp_uncertainty_s < 0.0) {
        std::cerr << "[ERROR] D435时间映射参数配置无效" << std::endl;
        ok = false;
    }
    if (FLAGS_camera_power_line_frequency < 0 ||
        FLAGS_camera_power_line_frequency > 3) {
        std::cerr << "[ERROR] camera.power_line_frequency必须在0到3之间" << std::endl;
        ok = false;
    }
    if (!FLAGS_camera_auto_white_balance &&
        (FLAGS_camera_white_balance_temperature < 2800 ||
         FLAGS_camera_white_balance_temperature > 6500)) {
        std::cerr << "[ERROR] 固定白平衡色温必须在2800K到6500K之间" << std::endl;
        ok = false;
    }
    return ok;
}

bool set_v4l2_control(int fd, uint32_t id, int32_t value, const char* name)
{
    v4l2_control control{};
    control.id = id;
    control.value = value;
    if (ioctl(fd, VIDIOC_S_CTRL, &control) < 0) {
        std::cerr << "[ERROR] 设置V4L2控制失败: " << name << "=" << value
                  << ", " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool get_v4l2_control(int fd, uint32_t id, int32_t& value, const char* name)
{
    v4l2_control control{};
    control.id = id;
    if (ioctl(fd, VIDIOC_G_CTRL, &control) < 0) {
        std::cerr << "[ERROR] 读取V4L2控制失败: " << name
                  << ", " << std::strerror(errno) << std::endl;
        return false;
    }
    value = control.value;
    return true;
}

bool configure_v4l2_controls(const std::string& device_path)
{
    int fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[ERROR] 无法打开V4L2控制设备: " << device_path
                  << ", " << std::strerror(errno) << std::endl;
        return false;
    }

    bool ok = true;
    const int32_t exposure_mode = FLAGS_camera_auto_exposure
        ? V4L2_EXPOSURE_APERTURE_PRIORITY
        : V4L2_EXPOSURE_MANUAL;
    ok &= set_v4l2_control(fd, V4L2_CID_EXPOSURE_AUTO,
                           exposure_mode, "auto_exposure");

    if (!FLAGS_camera_auto_exposure) {
        const int32_t exposure_absolute =
            std::max(1, static_cast<int32_t>(std::lround(FLAGS_camera_exposure_us / 100.0)));
        ok &= set_v4l2_control(fd, V4L2_CID_EXPOSURE_ABSOLUTE,
                               exposure_absolute, "exposure_time_absolute");
    }

    ok &= set_v4l2_control(fd, V4L2_CID_GAIN, FLAGS_camera_gain, "gain");
    ok &= set_v4l2_control(fd, V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                           FLAGS_camera_dynamic_framerate ? 1 : 0,
                           "exposure_dynamic_framerate");
    ok &= set_v4l2_control(fd, V4L2_CID_POWER_LINE_FREQUENCY,
                           FLAGS_camera_power_line_frequency,
                           "power_line_frequency");
    ok &= set_v4l2_control(fd, V4L2_CID_AUTO_WHITE_BALANCE,
                           FLAGS_camera_auto_white_balance ? 1 : 0,
                           "auto_white_balance");
    if (!FLAGS_camera_auto_white_balance) {
        ok &= set_v4l2_control(fd, V4L2_CID_WHITE_BALANCE_TEMPERATURE,
                               FLAGS_camera_white_balance_temperature,
                               "white_balance_temperature");
    }

    int32_t actual_auto = -1;
    int32_t actual_exposure = -1;
    int32_t actual_gain = -1;
    int32_t actual_dynamic_fps = -1;
    int32_t actual_power_line_frequency = -1;
    int32_t actual_auto_white_balance = -1;
    int32_t actual_white_balance_temperature = -1;
    ok &= get_v4l2_control(fd, V4L2_CID_EXPOSURE_AUTO,
                           actual_auto, "auto_exposure");
    ok &= get_v4l2_control(fd, V4L2_CID_EXPOSURE_ABSOLUTE,
                           actual_exposure, "exposure_time_absolute");
    ok &= get_v4l2_control(fd, V4L2_CID_GAIN, actual_gain, "gain");
    ok &= get_v4l2_control(fd, V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                           actual_dynamic_fps, "exposure_dynamic_framerate");
    ok &= get_v4l2_control(fd, V4L2_CID_POWER_LINE_FREQUENCY,
                           actual_power_line_frequency, "power_line_frequency");
    ok &= get_v4l2_control(fd, V4L2_CID_AUTO_WHITE_BALANCE,
                           actual_auto_white_balance, "auto_white_balance");
    ok &= get_v4l2_control(fd, V4L2_CID_WHITE_BALANCE_TEMPERATURE,
                           actual_white_balance_temperature,
                           "white_balance_temperature");
    close(fd);

    std::cout << "[INFO] V4L2曝光配置读回:" << std::endl;
    std::cout << "       自动曝光: " << actual_auto
              << (actual_auto == V4L2_EXPOSURE_MANUAL ? " (Manual)" : "") << std::endl;
    std::cout << "       曝光时间: " << actual_exposure * 100 << " us" << std::endl;
    std::cout << "       增益: " << actual_gain << std::endl;
    std::cout << "       动态降帧: " << actual_dynamic_fps << std::endl;
    std::cout << "       防闪烁频率: " << actual_power_line_frequency << std::endl;
    std::cout << "       自动白平衡: " << actual_auto_white_balance << std::endl;
    std::cout << "       白平衡色温: " << actual_white_balance_temperature << " K" << std::endl;

    if (!FLAGS_camera_auto_exposure) {
        const int32_t target_exposure =
            std::max(1, static_cast<int32_t>(std::lround(FLAGS_camera_exposure_us / 100.0)));
        ok &= actual_auto == V4L2_EXPOSURE_MANUAL;
        ok &= std::abs(actual_exposure - target_exposure) <= 1;
    }
    ok &= actual_gain == FLAGS_camera_gain;
    ok &= actual_dynamic_fps == (FLAGS_camera_dynamic_framerate ? 1 : 0);
    ok &= actual_power_line_frequency == FLAGS_camera_power_line_frequency;
    ok &= actual_auto_white_balance == (FLAGS_camera_auto_white_balance ? 1 : 0);
    if (!FLAGS_camera_auto_white_balance) {
        ok &= actual_white_balance_temperature == FLAGS_camera_white_balance_temperature;
    }

    if (!ok) {
        std::cerr << "[ERROR] 摄像头曝光/增益设置未通过读回校验" << std::endl;
    }
    return ok;
}

int fourcc_from_string(const std::string& value)
{
    if (value.size() != 4) return 0;
    return cv::VideoWriter::fourcc(value[0], value[1], value[2], value[3]);
}

std::string fourcc_to_string(int value)
{
    std::string result(4, ' ');
    result[0] = static_cast<char>(value & 0xff);
    result[1] = static_cast<char>((value >> 8) & 0xff);
    result[2] = static_cast<char>((value >> 16) & 0xff);
    result[3] = static_cast<char>((value >> 24) & 0xff);
    return result;
}

/**
 * @brief 在检测画面左上角绘制实时状态面板。
 */
void draw_status_overlay(cv::Mat& image,
                         const std::vector<Detection>& detections,
                         const std::vector<std::string>& class_names,
                         double actual_fps,
                         double pipeline_fps,
                         float max_candidate_score,
                         double e2e_ms,
                         double preprocess_ms,
                         double infer_ms,
                         double postprocess_ms)
{
    if (image.empty()) return;

    std::vector<std::string> lines = {
        "Detections: " + std::to_string(detections.size()),
        "Camera: " + cv::format("%.1f FPS", actual_fps),
        "Pipeline: " + cv::format("%.1f FPS", pipeline_fps),
        "Max candidate: " + cv::format("%.3f", max_candidate_score),
        "E2E: " + cv::format("%.1f ms", e2e_ms),
        "Pre/Infer/Post: " + cv::format("%.1f / %.1f / %.1f ms",
                                        preprocess_ms, infer_ms, postprocess_ms)
    };

    // Show top-3 detections for live visual debugging.
    const size_t top_n = std::min<size_t>(3, detections.size());
    for (size_t i = 0; i < top_n; ++i) {
        const auto& det = detections[i];
        std::string name = (static_cast<size_t>(det.class_id) < class_names.size())
            ? class_names[static_cast<size_t>(det.class_id)]
            : ("class_" + std::to_string(det.class_id));
        lines.push_back(cv::format("#%zu %s %.2f", i + 1, name.c_str(), det.score));
    }

    const int padding = 10;
    const int line_height = 24;
    const double font_scale = 0.55;
    const int thickness = 1;
    int baseline = 0;
    int panel_width = 0;
    for (const auto& line : lines) {
        cv::Size text_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX,
                                             font_scale, thickness, &baseline);
        panel_width = std::max(panel_width, text_size.width);
    }
    panel_width += padding * 2;
    int panel_height = static_cast<int>(lines.size()) * line_height + padding;
    panel_width = std::min(panel_width, image.cols);
    panel_height = std::min(panel_height, image.rows);

    cv::Mat overlay = image.clone();
    cv::rectangle(overlay,
                  cv::Rect(0, 0, panel_width, panel_height),
                  cv::Scalar(0, 0, 0),
                  cv::FILLED);
    cv::addWeighted(overlay, 0.45, image, 0.55, 0, image);

    int y = padding + 14;
    for (const auto& line : lines) {
        cv::putText(image, line, cv::Point(padding, y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        y += line_height;
    }
}

/**
 * @brief 根据最大显示尺寸生成窗口显示帧。
 */
cv::Mat make_display_frame(const cv::Mat& frame)
{
    if (FLAGS_fullscreen || FLAGS_display_width <= 0 || FLAGS_display_height <= 0) {
        return frame;
    }

    double scale = std::min(FLAGS_display_width / static_cast<double>(frame.cols),
                            FLAGS_display_height / static_cast<double>(frame.rows));
    if (scale <= 0.0 || std::abs(scale - 1.0) < 1e-3) {
        return frame;
    }

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    return resized;
}

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
    std::cout << "  --config          运行配置YAML" << std::endl;
    std::cout << "  --model_path      模型文件路径 (.hbm)" << std::endl;
    std::cout << "  --label_file      类别标签文件路径" << std::endl;
    std::cout << "  --video_device    摄像头设备 (留空自动检测)" << std::endl;
    std::cout << "  --score_thres     置信度阈值 (默认: 0.25)" << std::endl;
    std::cout << "  --nms_thres       NMS阈值 (默认: 0.45)" << std::endl;
    std::cout << "  --camera_width    采集宽度 (默认: 848)" << std::endl;
    std::cout << "  --camera_height   采集高度 (默认: 480)" << std::endl;
    std::cout << "  --camera_fps      采集帧率 (默认: 60)" << std::endl;
    std::cout << "  --depth_enabled   使用RealSense对齐RGB-D深度 (默认: true)" << std::endl;
    std::cout << "  --object_diameter_m 球体实际直径 (默认: 0.215m)" << std::endl;
    std::cout << "  --depth_show_colormap 显示对齐深度伪彩窗口" << std::endl;
    std::cout << "  --display         是否显示实时窗口 (默认: true)" << std::endl;
    std::cout << "  --fullscreen      全屏显示 (默认: true)" << std::endl;
    std::cout << "  --save_video      保存可视化视频路径 (默认: 不保存)" << std::endl;
    std::cout << "  --snapshot_dir    保存可视化截图目录 (默认: 不保存)" << std::endl;
    std::cout << "  --snapshot_interval 每隔多少帧截图一次 (默认: 0)" << std::endl;
    std::cout << "\n操作说明:" << std::endl;
    std::cout << "  实时窗口中按 'q' 键退出程序" << std::endl;
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

    if (!load_runtime_config(FLAGS_config)) {
        return 1;
    }
    if (!validate_runtime_config()) {
        return 1;
    }
    
    print_usage();
    std::cout << "启动参数: " << gflags::GetArgv() << std::endl;

    std::string video_device = FLAGS_video_device;
    cv::VideoCapture cap;
    const bool use_rgbd = FLAGS_depth_enabled;
    int actual_camera_width = FLAGS_camera_width;
    int actual_camera_height = FLAGS_camera_height;
    double actual_camera_fps = FLAGS_camera_fps;

#ifdef HAVE_REALSENSE2
    std::unique_ptr<RealSenseRgbdCapture> rgbd_capture;
#endif

    // ==================== Step 1: 打开摄像头 ====================
    if (use_rgbd) {
#ifdef HAVE_REALSENSE2
        RealSenseCaptureConfig capture_config;
        capture_config.serial = FLAGS_realsense_serial;
        capture_config.width = FLAGS_camera_width;
        capture_config.height = FLAGS_camera_height;
        capture_config.fps = FLAGS_camera_fps;
        capture_config.auto_exposure = FLAGS_camera_auto_exposure;
        capture_config.exposure_us = static_cast<float>(FLAGS_camera_exposure_us);
        capture_config.gain = static_cast<float>(FLAGS_camera_gain);
        capture_config.auto_white_balance = FLAGS_camera_auto_white_balance;
        capture_config.white_balance_temperature =
            static_cast<float>(FLAGS_camera_white_balance_temperature);
        capture_config.power_line_frequency =
            static_cast<float>(FLAGS_camera_power_line_frequency);
        capture_config.auto_exposure_priority = FLAGS_camera_dynamic_framerate;
        capture_config.strict_controls = FLAGS_strict_camera_config;
        capture_config.spatial_filter = FLAGS_depth_spatial_filter;
        capture_config.temporal_filter = FLAGS_depth_temporal_filter;
        capture_config.hole_filling_filter = FLAGS_depth_hole_filling_filter;
        capture_config.allow_hardware_time_fallback =
            FLAGS_depth_allow_hardware_time_fallback;
        capture_config.timestamp_fallback_warmup_frames =
            FLAGS_depth_timestamp_fallback_warmup_frames;
        capture_config.max_timestamp_uncertainty_s =
            FLAGS_depth_max_timestamp_uncertainty_s;

        rgbd_capture = std::make_unique<RealSenseRgbdCapture>();
        std::string error;
        if (!rgbd_capture->start(capture_config, &error)) {
            std::cerr << "[ERROR] RealSense RGB-D启动失败: " << error << std::endl;
            return 1;
        }
        std::cout << "[INFO] RealSense RGB-D已启动: serial="
                  << (FLAGS_realsense_serial.empty() ? "auto" : FLAGS_realsense_serial)
                  << ", " << actual_camera_width << "x" << actual_camera_height
                  << "@" << actual_camera_fps << std::endl;
#else
        std::cerr << "[ERROR] depth.enabled=true，但构建时未找到librealsense2" << std::endl;
        return 1;
#endif
    } else {
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

        cap.open(video_device, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            std::cerr << "[ERROR] 无法打开摄像头设备: " << video_device << std::endl;
            return -1;
        }
        std::cout << "[INFO] 摄像头打开成功: " << video_device << std::endl;

        const int requested_fourcc = fourcc_from_string(FLAGS_camera_fourcc);
        if (requested_fourcc == 0) {
            std::cerr << "[ERROR] camera_fourcc必须是4个字符: "
                      << FLAGS_camera_fourcc << std::endl;
            return 1;
        }
        cap.set(cv::CAP_PROP_FOURCC, requested_fourcc);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, FLAGS_camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, FLAGS_camera_height);
        cap.set(cv::CAP_PROP_FPS, FLAGS_camera_fps);
        cap.set(cv::CAP_PROP_BUFFERSIZE, FLAGS_camera_buffer_count);

        if (!configure_v4l2_controls(video_device) && FLAGS_strict_camera_config) {
            return 1;
        }

        actual_camera_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        actual_camera_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        actual_camera_fps = cap.get(cv::CAP_PROP_FPS);
        const int actual_fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        if (actual_camera_width <= 0) actual_camera_width = FLAGS_camera_width;
        if (actual_camera_height <= 0) actual_camera_height = FLAGS_camera_height;

        std::cout << "[INFO] 摄像头配置:" << std::endl;
        std::cout << "       分辨率: " << actual_camera_width << "x"
                  << actual_camera_height << std::endl;
        std::cout << "       帧率: " << actual_camera_fps << " FPS" << std::endl;
        std::cout << "       像素格式: " << fourcc_to_string(actual_fourcc) << std::endl;

        const bool camera_mode_ok =
            actual_camera_width == FLAGS_camera_width &&
            actual_camera_height == FLAGS_camera_height &&
            std::abs(actual_camera_fps - FLAGS_camera_fps) <= 0.5 &&
            actual_fourcc == requested_fourcc;
        if (!camera_mode_ok) {
            std::cerr << "[ERROR] 摄像头模式读回与配置不一致，目标="
                      << FLAGS_camera_width << "x" << FLAGS_camera_height << "@"
                      << FLAGS_camera_fps << " " << FLAGS_camera_fourcc << std::endl;
            if (FLAGS_strict_camera_config) return 1;
        }
    }

    // ==================== Step 2: 加载模型 ====================
    std::cout << "[INFO] 正在加载模型: " << FLAGS_model_path << std::endl;
    UltralyticsYOLO yolo(FLAGS_model_path);

    // ==================== Step 3: 加载类别标签 ====================
    std::cout << "[INFO] 正在加载类别标签: " << FLAGS_label_file << std::endl;
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    std::cout << "[INFO] 已加载 " << class_names.size() << " 个类别" << std::endl;
    if (class_names.empty()) {
        std::cerr << "[WARN] 标签文件为空或加载失败，检测框将使用 class_<id> 显示" << std::endl;
    }

    SphereDepthConfig sphere_config;
    sphere_config.object_diameter_m = static_cast<float>(FLAGS_object_diameter_m);
    sphere_config.bbox_scale = static_cast<float>(FLAGS_mono_bbox_scale);
    sphere_config.min_depth_m = static_cast<float>(FLAGS_depth_min_m);
    sphere_config.max_depth_m = static_cast<float>(FLAGS_depth_max_m);
    sphere_config.sample_radius_scale = static_cast<float>(FLAGS_depth_sample_radius_scale);
    sphere_config.sample_rings = FLAGS_depth_sample_rings;
    sphere_config.samples_per_ring = FLAGS_depth_samples_per_ring;
    sphere_config.patch_radius = FLAGS_depth_patch_radius;
    sphere_config.min_valid_samples = FLAGS_depth_min_samples;
    sphere_config.mad_scale = static_cast<float>(FLAGS_depth_mad_scale);
    sphere_config.max_center_spread_m = static_cast<float>(FLAGS_depth_max_spread_m);
    sphere_config.mono_consistency_relative =
        static_cast<float>(FLAGS_depth_mono_consistency_relative);
    sphere_config.mono_consistency_absolute_m =
        static_cast<float>(FLAGS_depth_mono_consistency_absolute_m);
    SphereDistanceEstimator distance_estimator(sphere_config);
    TrackingConfig tracking_config;
    tracking_config.enabled = FLAGS_tracking_enabled;
    tracking_config.min_init_frames = FLAGS_tracking_min_init_frames;
    tracking_config.max_lost_frames = FLAGS_tracking_max_lost_frames;
    tracking_config.process_accel_mps2 = FLAGS_tracking_process_accel;
    tracking_config.innovation_gate_chi2 = FLAGS_tracking_gate_chi2;
    tracking_config.min_detection_confidence = FLAGS_tracking_min_detection_confidence;
    tracking_config.min_depth_samples = FLAGS_tracking_min_depth_samples;
    tracking_config.max_depth_spread_m = FLAGS_tracking_max_depth_spread;
    tracking_config.max_rgbd_mono_delta_m = FLAGS_tracking_max_rgbd_mono_delta;
    TrajectoryConfig trajectory_config;
    trajectory_config.enabled = FLAGS_trajectory_enabled;
    trajectory_config.student_t_nu = FLAGS_trajectory_student_t_nu;
    trajectory_config.q_position = FLAGS_trajectory_q_position;
    trajectory_config.q_velocity = FLAGS_trajectory_q_velocity;
    trajectory_config.gravity_mps2 = FLAGS_trajectory_gravity;
    trajectory_config.ball_mass_kg = FLAGS_trajectory_ball_mass;
    trajectory_config.ball_radius_m = FLAGS_trajectory_ball_radius;
    trajectory_config.drag_coefficient = FLAGS_trajectory_drag_coefficient;
    trajectory_config.air_density = FLAGS_trajectory_air_density;
    trajectory_config.min_track_frames = FLAGS_trajectory_min_track_frames;
    trajectory_config.min_speed_mps = FLAGS_trajectory_min_speed;
    trajectory_config.max_predict_time_s = FLAGS_trajectory_max_predict_time;
    trajectory_config.rk4_dt_s = FLAGS_trajectory_rk4_dt;
    LandingGateConfig landing_config;
    landing_config.enabled = FLAGS_landing_gate_enabled;
    landing_config.min_confidence = FLAGS_landing_min_confidence;
    landing_config.min_student_weight = FLAGS_landing_min_student_weight;
    landing_config.min_time_to_land_s = FLAGS_landing_min_time;
    landing_config.max_time_to_land_s = FLAGS_landing_max_time;
    landing_config.min_speed_mps = FLAGS_landing_min_speed;
    landing_config.stable_frames = FLAGS_landing_stable_frames;
    landing_config.max_stable_jump_m = FLAGS_landing_max_stable_jump;
    landing_config.max_landing_sigma_m = FLAGS_landing_max_sigma;
    landing_config.max_abs_x_m = FLAGS_landing_max_abs_x;
    landing_config.min_depth_m = FLAGS_landing_min_depth;
    landing_config.max_depth_m = FLAGS_landing_max_depth;
    landing_config.allow_fallback_observation = FLAGS_landing_allow_fallback;
    BallTrajectoryTracker trajectory_tracker(tracking_config, trajectory_config,
                                              landing_config,
                                              {static_cast<float>(FLAGS_ground_normal_x),
                                               static_cast<float>(FLAGS_ground_normal_y),
                                               static_cast<float>(FLAGS_ground_normal_z)},
                                              FLAGS_ground_offset_m);
    D435ObservationPublisher d435_publisher(FLAGS_d435_observation_publish,
                                             FLAGS_d435_observation_topic,
                                             FLAGS_volleyball_class_id);

    if (FLAGS_camera_warmup_frames > 0) {
        std::cout << "[INFO] 摄像头预热，丢弃 " << FLAGS_camera_warmup_frames
                  << " 帧..." << std::endl;
        for (int i = 0; i < FLAGS_camera_warmup_frames; ++i) {
            bool warmup_ok = false;
            if (use_rgbd) {
#ifdef HAVE_REALSENSE2
                RgbdFrame warmup_frame;
                std::string error;
                warmup_ok = rgbd_capture->read(warmup_frame, &error) &&
                            !warmup_frame.color_bgr.empty();
                if (!warmup_ok && !error.empty()) {
                    std::cerr << "[ERROR] RealSense预热取流失败: " << error << std::endl;
                }
#endif
            } else {
                cv::Mat warmup_frame;
                warmup_ok = cap.read(warmup_frame) && !warmup_frame.empty();
            }
            if (!warmup_ok) {
                std::cerr << "[ERROR] 摄像头预热取流失败，frame=" << i << std::endl;
                if (FLAGS_strict_camera_config) return 1;
                break;
            }
        }
        std::cout << "[INFO] 摄像头预热完成" << std::endl;
    }

    // ==================== Step 4: 创建显示窗口 ====================
    const char* display_env = std::getenv("DISPLAY");
    const char* wayland_env = std::getenv("WAYLAND_DISPLAY");
    bool has_display = (display_env && *display_env) || (wayland_env && *wayland_env);
    bool enable_display = FLAGS_display && !FLAGS_headless;
    if (enable_display && !has_display) {
        // RDK desktop sessions commonly expose Xorg on :0 even when SSH has no DISPLAY.
        if (fs::exists("/tmp/.X11-unix/X0")) {
            setenv("DISPLAY", ":0", 0);
            has_display = true;
            std::cout << "[INFO] 未检测到DISPLAY，自动使用 :0 进行可视化" << std::endl;
        } else {
            std::cout << "[WARN] 未检测到DISPLAY/WAYLAND_DISPLAY，自动关闭显示" << std::endl;
            enable_display = false;
        }
    }

    const std::string window_name = "YOLO Detection - RDK BPU";
    const std::string depth_window_name = "Aligned Depth - D435i";
    if (enable_display) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        if (use_rgbd && FLAGS_depth_show_colormap) {
            cv::namedWindow(depth_window_name, cv::WINDOW_NORMAL);
        }
        if (FLAGS_fullscreen) {
            cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        }
        std::cout << "\n[INFO] 实时可视化窗口已开启，按 'q' 键退出程序\n" << std::endl;
    } else {
        std::cout << "\n[INFO] 实时可视化窗口已关闭，可使用 --save_video 或 --snapshot_dir 保存结果\n" << std::endl;
    }

    if (!FLAGS_snapshot_dir.empty()) {
        fs::create_directories(FLAGS_snapshot_dir);
        if (use_rgbd) {
            fs::create_directories(fs::path(FLAGS_snapshot_dir) / "depth_mm");
        }
        std::cout << "[INFO] 可视化截图目录: " << FLAGS_snapshot_dir << std::endl;
    }

    cv::VideoWriter visual_writer;
    if (!FLAGS_save_video.empty()) {
        fs::path video_path(FLAGS_save_video);
        if (video_path.has_parent_path()) {
            fs::create_directories(video_path.parent_path());
        }
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double output_fps = actual_camera_fps > 0 ? actual_camera_fps : 30;
        visual_writer.open(FLAGS_save_video, fourcc, output_fps,
                           cv::Size(actual_camera_width, actual_camera_height));
        if (!visual_writer.isOpened()) {
            std::cerr << "[WARN] 无法打开可视化视频输出: " << FLAGS_save_video << std::endl;
        } else {
            std::cout << "[INFO] 可视化视频输出: " << FLAGS_save_video << std::endl;
        }
    }

    // ==================== Step 5: 主推理循环 ====================
    // 性能统计变量
    double total_capture_ms = 0.0;      // USB取流累计时间
    double total_preprocess_ms = 0.0;   // 预处理累计时间
    double total_infer_ms = 0.0;        // BPU推理累计时间
    double total_postprocess_ms = 0.0;  // 后处理累计时间
    double total_draw_ms = 0.0;         // 绘制累计时间
    double total_e2e_ms = 0.0;          // 端到端延迟累计时间
    int frame_count = 0;
    double display_actual_fps = 0.0;
    cv::Mat frame;
    CameraModel frame_camera;
    DepthImageView frame_depth;
#ifdef HAVE_REALSENSE2
    RgbdFrame rgbd_frame;
#endif
    
    auto loop_start = std::chrono::high_resolution_clock::now();

    while (true) {
        // 端到端计时起点 (从取流开始)
        auto e2e_start = std::chrono::high_resolution_clock::now();
        
        // 5.1 获取一帧图像；RGB-D模式下深度已对齐到彩色坐标系。
        auto capture_start = std::chrono::high_resolution_clock::now();
        bool capture_ok = false;
        if (use_rgbd) {
#ifdef HAVE_REALSENSE2
            std::string error;
            capture_ok = rgbd_capture->read(rgbd_frame, &error);
            if (capture_ok) {
                frame = rgbd_frame.color_bgr;
                frame_camera = rgbd_frame.camera;
                frame_depth = rgbd_frame.depth_view;
            } else if (!error.empty()) {
                std::cerr << "[WARN] RealSense取流失败: " << error << std::endl;
            }
#endif
        } else {
            capture_ok = cap.read(frame);
            frame_camera = {};
            frame_depth = {};
        }
        if (!capture_ok || frame.empty()) {
            std::cerr << "[WARN] 获取图像帧失败，尝试重新获取..." << std::endl;
            continue;
        }
        // ROS messages require the wall/ROS epoch. This is sampled immediately
        // after frame acquisition; do not publish steady_clock timestamps.
        double capture_timestamp_s = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        RgbdTiming observation_timing;
        observation_timing.color_capture_timestamp_s = capture_timestamp_s;
        observation_timing.depth_capture_timestamp_s = capture_timestamp_s;
#ifdef HAVE_REALSENSE2
        if (use_rgbd && rgbd_frame.capture_timestamp_s > 0.0) {
            capture_timestamp_s = rgbd_frame.capture_timestamp_s;
            observation_timing = rgbd_frame.timing;
        }
#endif
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

        // 5.5 对每个球框进行多点RGB-D球心估计和去畸变单目估距。
        auto draw_start = std::chrono::high_resolution_clock::now();
        std::vector<DistanceMeasurement> distance_measurements;
        distance_measurements.reserve(results.size());
        if (frame_camera.valid()) {
            for (const auto& detection : results) {
                distance_measurements.push_back(distance_estimator.estimate(
                    detection, frame_depth.valid() ? &frame_depth : nullptr, frame_camera));
            }
        }

        draw_boxes(frame, results, class_names, rdk_colors);
        drawDistanceMeasurements(frame, results, distance_measurements,
                                 FLAGS_depth_draw_samples);
        BallTrackResult track_result;
        if (FLAGS_tracking_enabled && FLAGS_trajectory_enabled) {
            int best = -1;
            for (size_t i = 0; i < results.size(); ++i)
                if (best < 0 || results[i].score > results[best].score) best = static_cast<int>(i);
            const double timestamp_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            track_result = trajectory_tracker.update(
                best >= 0 ? &results[best] : nullptr,
                best >= 0 && best < static_cast<int>(distance_measurements.size())
                    ? &distance_measurements[best] : nullptr,
                frame_camera, timestamp_s);
            drawTrajectoryStatus(frame, track_result);
        }
        // Vision node always publishes sensor observations, independently of
        // any local diagnostic tracker. Control selection belongs exclusively
        // to catch_controller.
        d435_publisher.publish(results, distance_measurements, observation_timing,
                               static_cast<uint32_t>(frame_count));
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
            display_actual_fps = actual_fps;
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
            std::cout << "[实际]   真实帧率: " << actual_fps << " FPS, 检测到 "
                      << results.size() << " 个目标, 最大候选置信度: "
                      << yolo.get_last_max_score() << std::endl;
            if (!distance_measurements.empty()) {
                const auto& distance = distance_measurements.front();
                std::cout << "[测距]   RGB-D球心/表面: ";
                if (distance.depth_valid) {
                    std::cout << distance.center_xyz_m.z << " / "
                              << distance.surface_z_m << " m, samples="
                              << distance.valid_samples << ", spread="
                              << distance.center_spread_m << " m";
                } else {
                    std::cout << "invalid";
                }
                std::cout << "; 单目球心: ";
                if (distance.mono_valid) {
                    std::cout << distance.mono_center_z_m << " m";
                } else {
                    std::cout << "invalid";
                }
                std::cout << std::endl;
            }
            std::cout << "=============================================================\n" << std::endl;
            
            // 重置累计
            total_capture_ms = 0.0;
            total_preprocess_ms = 0.0;
            total_infer_ms = 0.0;
            total_postprocess_ms = 0.0;
            total_draw_ms = 0.0;
            total_e2e_ms = 0.0;
        }

        // 5.6 绘制可视化状态面板
        double pipeline_fps = 1000.0 / (pipeline_ms > 0 ? pipeline_ms : 1);
        draw_status_overlay(frame, results, class_names, display_actual_fps,
                            pipeline_fps, yolo.get_last_max_score(), e2e_ms,
                            preprocess_ms, infer_ms, postprocess_ms);

        // 5.7 保存可视化结果
        if (visual_writer.isOpened()) {
            visual_writer.write(frame);
        }
        if (!FLAGS_snapshot_dir.empty() &&
            FLAGS_snapshot_interval > 0 &&
            frame_count % FLAGS_snapshot_interval == 0) {
            fs::path snapshot_path = fs::path(FLAGS_snapshot_dir) /
                std::string(cv::format("frame_%06d.jpg", frame_count));
            cv::imwrite(snapshot_path.string(), frame);
            if (frame_depth.valid()) {
                cv::Mat raw_depth(frame_depth.height, frame_depth.width, CV_16UC1,
                                  const_cast<uint16_t*>(frame_depth.data),
                                  frame_depth.stride_bytes);
                cv::Mat depth_mm;
                raw_depth.convertTo(depth_mm, CV_16UC1,
                                    frame_depth.depth_scale_m * 1000.0f);
                fs::path depth_path = fs::path(FLAGS_snapshot_dir) / "depth_mm" /
                    std::string(cv::format("depth_%06d.png", frame_count));
                cv::imwrite(depth_path.string(), depth_mm);
            }
        }

        // 5.8 显示当前帧
        if (enable_display) {
            cv::Mat display_frame = make_display_frame(frame);
            cv::imshow(window_name, display_frame);
            if (use_rgbd && FLAGS_depth_show_colormap && frame_depth.valid()) {
                cv::Mat depth_colormap = colorizeDepth(
                    frame_depth, static_cast<float>(FLAGS_depth_min_m),
                    static_cast<float>(FLAGS_depth_max_m));
                if (!depth_colormap.empty()) {
                    cv::imshow(depth_window_name, make_display_frame(depth_colormap));
                }
            }

            // 检测按键，按 'q' 退出
            int key = cv::waitKey(1);
            if ((key & 0xFF) == 'q' || (key & 0xFF) == 'Q') {
                std::cout << "\n[INFO] 用户按下 'q' 键，正在退出..." << std::endl;
                break;
            }
        }
    }

    // ==================== Step 6: 释放资源 ====================
    std::cout << "[INFO] 正在释放资源..." << std::endl;
    if (visual_writer.isOpened()) {
        visual_writer.release();
    }
    if (cap.isOpened()) cap.release();
#ifdef HAVE_REALSENSE2
    if (rgbd_capture) rgbd_capture->stop();
#endif
    if (enable_display) {
        cv::destroyAllWindows();
    }

    std::cout << "[INFO] 程序正常退出" << std::endl;
    return 0;
}

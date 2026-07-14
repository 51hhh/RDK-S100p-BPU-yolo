#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common_utils.hpp"

struct CameraModel {
    enum class DistortionModel {
        None,
        BrownConrady,
        InverseBrownConrady,
        Unsupported
    };

    int width{0};
    int height{0};
    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};
    // OpenCV Brown-Conrady order: k1, k2, p1, p2, k3.
    cv::Vec<float, 5> distortion{0, 0, 0, 0, 0};
    DistortionModel distortion_model{DistortionModel::None};

    bool valid() const { return width > 0 && height > 0 && fx > 0.0f && fy > 0.0f; }
};

struct DepthImageView {
    const uint16_t* data{nullptr};
    int width{0};
    int height{0};
    int stride_bytes{0};
    float depth_scale_m{0.001f};

    float at(int x, int y) const;
    bool valid() const;
};

struct SphereDepthConfig {
    float object_diameter_m{0.215f};
    float bbox_scale{0.95f};
    float min_depth_m{0.20f};
    float max_depth_m{12.0f};
    float sample_radius_scale{0.68f};
    int sample_rings{3};
    int samples_per_ring{12};
    int patch_radius{1};
    int min_valid_samples{6};
    float mad_scale{3.0f};
    float max_center_spread_m{0.12f};
    float mono_consistency_relative{0.45f};
    float mono_consistency_absolute_m{0.75f};
};

struct DistanceMeasurement {
    bool depth_valid{false};
    bool mono_valid{false};
    cv::Point3f center_xyz_m{0.0f, 0.0f, 0.0f};
    float center_range_m{0.0f};
    float surface_range_m{0.0f};
    float surface_z_m{0.0f};
    float mono_center_z_m{0.0f};
    float mono_surface_z_m{0.0f};
    float center_spread_m{0.0f};
    int valid_samples{0};
    std::vector<cv::Point> accepted_pixels;
};

enum class RgbdTimestampDomain : uint8_t {
    Unknown = 0,
    HardwareClock = 1,
    SystemTime = 2,
    GlobalTime = 3,
};

struct RgbdTiming {
    double color_capture_timestamp_s{0.0};
    double depth_capture_timestamp_s{0.0};
    int64_t rgb_depth_delta_ns{0};
    uint64_t timestamp_uncertainty_ns{0};
    RgbdTimestampDomain timestamp_domain{RgbdTimestampDomain::Unknown};
    bool timestamp_mapping_valid{false};
};

class SphereDistanceEstimator {
public:
    explicit SphereDistanceEstimator(SphereDepthConfig config = {});

    DistanceMeasurement estimate(const Detection& detection,
                                 const DepthImageView* aligned_depth,
                                 const CameraModel& camera) const;

    static cv::Point2f undistortPixel(const cv::Point2f& pixel,
                                      const CameraModel& camera);

private:
    SphereDepthConfig config_;
};

void drawDistanceMeasurements(cv::Mat& image,
                              const std::vector<Detection>& detections,
                              const std::vector<DistanceMeasurement>& measurements,
                              bool draw_sample_points);

cv::Mat colorizeDepth(const DepthImageView& depth,
                      float min_depth_m,
                      float max_depth_m);

struct D435ImuSample {
    cv::Vec3d angular_velocity{0.0, 0.0, 0.0};
    cv::Vec3d linear_acceleration{0.0, 0.0, 0.0};
    double gyro_timestamp_s{0.0};
    double accel_timestamp_s{0.0};
    bool gyro_valid{false};
    bool accel_valid{false};
};

#ifdef HAVE_REALSENSE2

#include <librealsense2/rs.hpp>

struct RealSenseCaptureConfig {
    std::string serial;
    int width{848};
    int height{480};
    int fps{60};
    bool auto_exposure{false};
    float exposure_us{15000.0f};
    float gain{128.0f};
    bool auto_white_balance{false};
    float white_balance_temperature{4600.0f};
    float power_line_frequency{1.0f};
    bool auto_exposure_priority{false};
    bool strict_controls{true};
    bool spatial_filter{true};
    bool temporal_filter{false};
    bool hole_filling_filter{false};
    bool allow_hardware_time_fallback{false};
    int timestamp_fallback_warmup_frames{30};
    double max_timestamp_uncertainty_s{0.002};
    bool imu_enabled{true};
    int gyro_fps{200};
    int accel_fps{63};
    std::function<void(const D435ImuSample&)> imu_callback;
};

struct RgbdFrame {
    cv::Mat color_bgr;
    std::optional<rs2::depth_frame> aligned_depth;
    CameraModel camera;
    DepthImageView depth_view;
    double capture_timestamp_s{0.0};
    RgbdTiming timing;
};

class RealSenseRgbdCapture {
public:
    bool start(const RealSenseCaptureConfig& config, std::string* error = nullptr);
    bool read(RgbdFrame& output, std::string* error = nullptr);
    void stop();
    bool isStarted() const { return started_; }

private:
    bool configureColorSensor(const RealSenseCaptureConfig& config,
                              std::string* error);
    bool startMotionSensor(const RealSenseCaptureConfig& config,
                           std::string* error);
    void onMotionFrame(const rs2::frame& frame);
    double motionTimestampS(const rs2::frame& frame);

    rs2::pipeline pipeline_;
    rs2::pipeline_profile profile_;
    std::optional<rs2::sensor> motion_sensor_;
    rs2::align align_to_color_{RS2_STREAM_COLOR};
    rs2::spatial_filter spatial_filter_;
    rs2::temporal_filter temporal_filter_;
    rs2::hole_filling_filter hole_filling_filter_;
    float depth_scale_m_{0.001f};
    bool use_spatial_filter_{true};
    bool use_temporal_filter_{false};
    bool use_hole_filling_filter_{false};
    bool allow_hardware_time_fallback_{false};
    int timestamp_fallback_warmup_frames_{30};
    double max_timestamp_uncertainty_s_{0.002};
    bool started_{false};
    std::optional<double> device_to_system_offset_s_;
    int device_timestamp_samples_{0};
    std::mutex motion_mutex_;
    D435ImuSample latest_imu_;
    std::optional<double> motion_device_to_system_offset_s_;
    std::function<void(const D435ImuSample&)> imu_callback_;
};

#endif  // HAVE_REALSENSE2

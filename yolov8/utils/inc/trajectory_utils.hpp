#pragma once

#include <string>
#include <opencv2/core.hpp>

#include "common_utils.hpp"
#include "rgbd_utils.hpp"

enum class ObservationSource { None, Rgbd, KalmanPrediction, Monocular };

struct TrackingConfig {
    bool enabled{true};
    int min_init_frames{3};
    int max_lost_frames{8};
    float process_accel_mps2{30.0f};
    float innovation_gate_chi2{11.34f};
    float min_detection_confidence{0.10f};
    int min_depth_samples{8};
    float max_depth_spread_m{0.10f};
    float max_rgbd_mono_delta_m{0.80f};
};

struct TrajectoryConfig {
    bool enabled{true};
    float student_t_nu{12.0f};
    float q_position{0.0001f};
    float q_velocity{1.5f};
    float gravity_mps2{9.81f};
    float ball_mass_kg{0.270f};
    float ball_radius_m{0.1075f};
    float drag_coefficient{0.10f};
    float air_density{1.225f};
    int min_track_frames{6};
    float min_speed_mps{0.5f};
    float max_predict_time_s{3.0f};
    float rk4_dt_s{0.008f};
};

struct LandingGateConfig {
    bool enabled{true};
    float min_confidence{0.70f};
    float min_student_weight{0.15f};
    float min_time_to_land_s{0.25f};
    float max_time_to_land_s{2.20f};
    float min_speed_mps{0.80f};
    int stable_frames{3};
    float max_stable_jump_m{0.35f};
    float max_landing_sigma_m{0.30f};
    float max_abs_x_m{3.6f};
    float min_depth_m{0.0f};
    float max_depth_m{12.0f};
    bool allow_fallback_observation{false};
};

struct BallTrackResult {
    bool track_valid{false};
    bool landing_valid{false};
    bool control_publish{false};
    cv::Point3f position_m{};
    cv::Point3f velocity_mps{};
    cv::Point3f landing_m{};
    float time_to_land_s{0.0f};
    float confidence{0.0f};
    float student_weight{0.0f};
    float landing_sigma_m{0.0f};
    ObservationSource source{ObservationSource::None};
    std::string gate_reason{"no track"};
};

class BallTrajectoryTracker {
public:
    BallTrajectoryTracker(TrackingConfig tracking, TrajectoryConfig trajectory,
                          LandingGateConfig gate, cv::Vec3f ground_normal,
                          float ground_offset_m);
    BallTrackResult update(const Detection* detection,
                           const DistanceMeasurement* distance,
                           const CameraModel& camera, double timestamp_s);
    void reset();

private:
    TrackingConfig tracking_;
    TrajectoryConfig trajectory_;
    LandingGateConfig gate_;
    cv::Vec3f ground_normal_;
    float ground_offset_m_;
    cv::Matx<float, 9, 1> x9_{};
    cv::Matx<float, 9, 9> p9_{cv::Matx<float,9,9>::eye()};
    cv::Matx<float, 6, 1> x6_{};
    cv::Matx<float, 6, 6> p6_{cv::Matx<float,6,6>::eye()};
    bool initialized_{false};
    double last_time_s_{0.0};
    int track_frames_{0};
    int lost_frames_{0};
    int stable_frames_{0};
    cv::Point3f last_landing_{};
};

void drawTrajectoryStatus(cv::Mat& image, const BallTrackResult& result);
const char* observationSourceName(ObservationSource source);

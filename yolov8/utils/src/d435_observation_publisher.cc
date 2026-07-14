#include "d435_observation_publisher.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#ifdef HAVE_VOLLEYBALL_ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <volleyball_interfaces/msg/d435_ball_observation.hpp>
#endif

namespace {

bool validDetection(const Detection& detection)
{
    if (!std::isfinite(detection.score)) return false;
    for (float value : detection.bbox) {
        if (!std::isfinite(value)) return false;
    }
    return detection.bbox[2] > detection.bbox[0] &&
           detection.bbox[3] > detection.bbox[1];
}

}  // namespace

struct D435ObservationPublisher::Impl {
    bool enabled{false};
    bool imu_enabled{false};
    int class_id{0};
    uint32_t source_epoch{0};
    std::string imu_frame_id;
    double gyro_variance{0.0};
    double accel_variance{0.0};
    double last_gyro_timestamp_s{0.0};
#ifdef HAVE_VOLLEYBALL_ROS2
    rclcpp::Node::SharedPtr node;
    rclcpp::Publisher<volleyball_interfaces::msg::D435BallObservation>::SharedPtr publisher;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher;
#endif
};

D435ObservationPublisher::D435ObservationPublisher(bool enabled,
                                                   std::string topic,
                                                   int class_id,
                                                   bool imu_enabled,
                                                   std::string imu_topic,
                                                   std::string imu_frame_id,
                                                   double gyro_stddev,
                                                   double accel_stddev)
    : impl_(new Impl)
{
    impl_->enabled = enabled;
    impl_->imu_enabled = imu_enabled;
    impl_->class_id = class_id;
    impl_->imu_frame_id = std::move(imu_frame_id);
    impl_->gyro_variance = std::max(0.0, gyro_stddev * gyro_stddev);
    impl_->accel_variance = std::max(0.0, accel_stddev * accel_stddev);
    impl_->source_epoch = std::random_device{}();
    if (impl_->source_epoch == 0) impl_->source_epoch = 1;
#ifdef HAVE_VOLLEYBALL_ROS2
    if (enabled || imu_enabled) {
        if (!rclcpp::ok()) {
            int argc = 0;
            char** argv = nullptr;
            rclcpp::init(argc, argv);
        }
        impl_->node = std::make_shared<rclcpp::Node>("d435_ball_vision");
    }
    if (enabled) {
        const auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                             .best_effort()
                             .durability_volatile()
                             .deadline(rclcpp::Duration::from_seconds(0.04));
        impl_->publisher =
            impl_->node->create_publisher<volleyball_interfaces::msg::D435BallObservation>(
                topic, qos);
        std::cout << "[INFO] D435观测发布: " << topic << std::endl;
    }
    if (imu_enabled) {
        impl_->imu_publisher = impl_->node->create_publisher<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::SensorDataQoS());
        std::cout << "[INFO] D435i IMU发布: " << imu_topic
                  << " frame=" << impl_->imu_frame_id << std::endl;
    }
#else
    if (enabled || imu_enabled) {
        std::cout << "[WARN] 未找到ROS2消息依赖，D435观测/IMU发布不可用"
                  << std::endl;
    }
#endif
}

void D435ObservationPublisher::publishImu(const D435ImuSample& sample)
{
#ifdef HAVE_VOLLEYBALL_ROS2
    if (!impl_->imu_enabled || !impl_->imu_publisher ||
        !sample.gyro_valid || !sample.accel_valid ||
        !std::isfinite(sample.gyro_timestamp_s) ||
        sample.gyro_timestamp_s <= impl_->last_gyro_timestamp_s ||
        std::abs(sample.gyro_timestamp_s - sample.accel_timestamp_s) > 0.10) {
        return;
    }
    const auto finite_vec = [](const cv::Vec3d& vector) {
        return std::isfinite(vector[0]) && std::isfinite(vector[1]) &&
               std::isfinite(vector[2]);
    };
    if (!finite_vec(sample.angular_velocity) ||
        !finite_vec(sample.linear_acceleration)) {
        return;
    }
    sensor_msgs::msg::Imu message;
    const int64_t timestamp_ns = static_cast<int64_t>(
        sample.gyro_timestamp_s * 1e9);
    message.header.stamp.sec = static_cast<int32_t>(timestamp_ns / 1000000000LL);
    message.header.stamp.nanosec =
        static_cast<uint32_t>(timestamp_ns % 1000000000LL);
    message.header.frame_id = impl_->imu_frame_id;
    message.orientation_covariance[0] = -1.0;
    message.angular_velocity.x = sample.angular_velocity[0];
    message.angular_velocity.y = sample.angular_velocity[1];
    message.angular_velocity.z = sample.angular_velocity[2];
    message.linear_acceleration.x = sample.linear_acceleration[0];
    message.linear_acceleration.y = sample.linear_acceleration[1];
    message.linear_acceleration.z = sample.linear_acceleration[2];
    for (int index : {0, 4, 8}) {
        message.angular_velocity_covariance[index] = impl_->gyro_variance;
        message.linear_acceleration_covariance[index] = impl_->accel_variance;
    }
    impl_->imu_publisher->publish(message);
    impl_->last_gyro_timestamp_s = sample.gyro_timestamp_s;
#else
    (void)sample;
#endif
}

D435ObservationPublisher::~D435ObservationPublisher() = default;

bool D435ObservationPublisher::available() const
{
#ifdef HAVE_VOLLEYBALL_ROS2
    return impl_->enabled && bool(impl_->publisher);
#else
    return false;
#endif
}

void D435ObservationPublisher::publish(
    const std::vector<Detection>& detections,
    const std::vector<DistanceMeasurement>& measurements,
    const RgbdTiming& timing,
    uint32_t frame_id)
{
#ifdef HAVE_VOLLEYBALL_ROS2
    if (!available() || !std::isfinite(timing.color_capture_timestamp_s) ||
        timing.color_capture_timestamp_s <= 0.0) {
        return;
    }

    int best = -1;
    for (size_t index = 0; index < detections.size(); ++index) {
        if (detections[index].class_id != impl_->class_id ||
            !validDetection(detections[index])) {
            continue;
        }
        if (best < 0 || detections[index].score > detections[best].score) {
            best = static_cast<int>(index);
        }
    }

    volleyball_interfaces::msg::D435BallObservation message;
    const int64_t timestamp_ns = static_cast<int64_t>(
        timing.color_capture_timestamp_s * 1e9);
    message.header.stamp.sec = static_cast<int32_t>(timestamp_ns / 1000000000LL);
    message.header.stamp.nanosec =
        static_cast<uint32_t>(timestamp_ns % 1000000000LL);
    message.header.frame_id = "camera_color_optical_frame";
    message.source_epoch = impl_->source_epoch;
    message.frame_id = frame_id;
    message.rgb_depth_timestamp_delta_ns = timing.rgb_depth_delta_ns;
    message.timestamp_uncertainty_ns = timing.timestamp_uncertainty_ns;
    message.timestamp_domain = static_cast<uint8_t>(timing.timestamp_domain);
    message.timestamp_mapping_valid = timing.timestamp_mapping_valid;
    message.class_id = impl_->class_id;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    message.position.x = nan;
    message.position.y = nan;
    message.position.z = nan;
    message.depth_spread_m = static_cast<float>(nan);
    message.mono_center_z_m = static_cast<float>(nan);
    std::fill(message.position_covariance.begin(),
              message.position_covariance.end(), 0.0);
    message.position_covariance[0] = 0.25;
    message.position_covariance[4] = 0.25;
    message.position_covariance[8] = 0.25;

    if (best >= 0) {
        const auto& detection = detections[best];
        message.detection_valid = true;
        message.detection_confidence = detection.score;
        message.class_id = detection.class_id;
        for (int index = 0; index < 4; ++index) {
            message.bbox_xyxy[index] = detection.bbox[index];
        }

        if (best < static_cast<int>(measurements.size())) {
            const auto& measurement = measurements[best];
            message.rgbd_valid = measurement.depth_valid;
            message.mono_valid = measurement.mono_valid;
            message.valid_depth_samples = static_cast<uint16_t>(
                std::clamp(measurement.valid_samples, 0, 65535));
            if (std::isfinite(measurement.center_spread_m)) {
                message.depth_spread_m = measurement.center_spread_m;
            }
            if (measurement.mono_valid && std::isfinite(measurement.mono_center_z_m)) {
                message.mono_center_z_m = measurement.mono_center_z_m;
            }
            if (measurement.depth_valid &&
                std::isfinite(measurement.center_xyz_m.x) &&
                std::isfinite(measurement.center_xyz_m.y) &&
                std::isfinite(measurement.center_xyz_m.z)) {
                message.position.x = measurement.center_xyz_m.x;
                message.position.y = measurement.center_xyz_m.y;
                message.position.z = measurement.center_xyz_m.z;
                const double variance = std::max(
                    0.0025,
                    double(measurement.center_spread_m * measurement.center_spread_m));
                message.position_covariance[0] = variance;
                message.position_covariance[4] = variance;
                message.position_covariance[8] = variance;
            } else {
                message.rgbd_valid = false;
            }
        }
    }

    impl_->publisher->publish(message);
#else
    (void)detections;
    (void)measurements;
    (void)timing;
    (void)frame_id;
#endif
}

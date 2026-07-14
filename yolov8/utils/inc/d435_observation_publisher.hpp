#pragma once
#include <memory>
#include <string>
#include <vector>
#include "common_utils.hpp"
#include "rgbd_utils.hpp"

class D435ObservationPublisher {
public:
    D435ObservationPublisher(bool enabled, std::string topic, int volleyball_class_id,
                             bool imu_enabled, std::string imu_topic,
                             std::string imu_frame_id,
                             double gyro_stddev, double accel_stddev);
    ~D435ObservationPublisher();
    bool available() const;
    void publish(const std::vector<Detection>& detections,
                 const std::vector<DistanceMeasurement>& measurements,
                 const RgbdTiming& timing, uint32_t frame_id);
    void publishImu(const D435ImuSample& sample);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

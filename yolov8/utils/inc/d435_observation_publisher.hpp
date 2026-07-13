#pragma once
#include <memory>
#include <string>
#include <vector>
#include "common_utils.hpp"
#include "rgbd_utils.hpp"

class D435ObservationPublisher {
public:
    D435ObservationPublisher(bool enabled, std::string topic, int volleyball_class_id);
    ~D435ObservationPublisher();
    bool available() const;
    void publish(const std::vector<Detection>& detections,
                 const std::vector<DistanceMeasurement>& measurements,
                 double capture_timestamp_s, uint32_t frame_id);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

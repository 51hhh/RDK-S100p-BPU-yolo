#include "rgbd_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

float median(std::vector<float> values)
{
    if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        result = 0.5f * (result + *lower);
    }
    return result;
}

float robustSpread(const std::vector<float>& values, float center)
{
    std::vector<float> deviations;
    deviations.reserve(values.size());
    for (float value : values) deviations.push_back(std::abs(value - center));
    return 1.4826f * median(std::move(deviations));
}

cv::Point3f deproject(const cv::Point2f& undistorted_pixel,
                      float depth_z_m,
                      const CameraModel& camera)
{
    return {
        (undistorted_pixel.x - camera.cx) * depth_z_m / camera.fx,
        (undistorted_pixel.y - camera.cy) * depth_z_m / camera.fy,
        depth_z_m
    };
}

float norm3(const cv::Point3f& point)
{
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

cv::Point3f normalized(const cv::Point3f& point)
{
    const float length = norm3(point);
    if (length <= 1e-9f) return {0.0f, 0.0f, 1.0f};
    return {point.x / length, point.y / length, point.z / length};
}

float dot3(const cv::Point3f& a, const cv::Point3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float angleBetween(const cv::Point3f& a, const cv::Point3f& b)
{
    return std::acos(std::clamp(dot3(normalized(a), normalized(b)), -1.0f, 1.0f));
}

float localMedianDepth(const DepthImageView& depth,
                       int center_x,
                       int center_y,
                       int radius,
                       float min_depth_m,
                       float max_depth_m)
{
    std::vector<float> values;
    const int r = std::max(0, radius);
    values.reserve((2 * r + 1) * (2 * r + 1));
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const float z = depth.at(center_x + dx, center_y + dy);
            if (std::isfinite(z) && z >= min_depth_m && z <= max_depth_m) {
                values.push_back(z);
            }
        }
    }
    return median(std::move(values));
}

std::string formatMeters(float value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value << "m";
    return stream.str();
}

}  // namespace

float DepthImageView::at(int x, int y) const
{
    if (!valid() || x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
    const auto* row = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const uint8_t*>(data) + y * stride_bytes);
    return static_cast<float>(row[x]) * depth_scale_m;
}

bool DepthImageView::valid() const
{
    return data != nullptr && width > 0 && height > 0 &&
           stride_bytes >= width * static_cast<int>(sizeof(uint16_t)) &&
           depth_scale_m > 0.0f;
}

SphereDistanceEstimator::SphereDistanceEstimator(SphereDepthConfig config)
    : config_(std::move(config))
{
}

cv::Point2f SphereDistanceEstimator::undistortPixel(const cv::Point2f& pixel,
                                                     const CameraModel& camera)
{
    if (!camera.valid()) return pixel;
    if (camera.distortion_model == CameraModel::DistortionModel::None) return pixel;
    if (camera.distortion_model == CameraModel::DistortionModel::InverseBrownConrady) {
        const float x = (pixel.x - camera.cx) / camera.fx;
        const float y = (pixel.y - camera.cy) / camera.fy;
        const float r2 = x * x + y * y;
        const float radial = 1.0f + camera.distortion[0] * r2 +
            camera.distortion[1] * r2 * r2 + camera.distortion[4] * r2 * r2 * r2;
        const float x_u = x * radial + 2.0f * camera.distortion[2] * x * y +
            camera.distortion[3] * (r2 + 2.0f * x * x);
        const float y_u = y * radial + 2.0f * camera.distortion[3] * x * y +
            camera.distortion[2] * (r2 + 2.0f * y * y);
        return {camera.fx * x_u + camera.cx, camera.fy * y_u + camera.cy};
    }
    if (camera.distortion_model == CameraModel::DistortionModel::Unsupported) return pixel;
    cv::Matx33f intrinsic(camera.fx, 0.0f, camera.cx,
                          0.0f, camera.fy, camera.cy,
                          0.0f, 0.0f, 1.0f);
    std::vector<cv::Point2f> source{pixel};
    std::vector<cv::Point2f> destination;
    cv::undistortPoints(source, destination, intrinsic, camera.distortion,
                        cv::noArray(), intrinsic);
    return destination.empty() ? pixel : destination.front();
}

DistanceMeasurement SphereDistanceEstimator::estimate(
    const Detection& detection,
    const DepthImageView* aligned_depth,
    const CameraModel& camera) const
{
    DistanceMeasurement result;
    if (!camera.valid() || config_.object_diameter_m <= 0.0f) return result;

    const float x1 = std::clamp(detection.bbox[0], 0.0f, static_cast<float>(camera.width - 1));
    const float y1 = std::clamp(detection.bbox[1], 0.0f, static_cast<float>(camera.height - 1));
    const float x2 = std::clamp(detection.bbox[2], 0.0f, static_cast<float>(camera.width - 1));
    const float y2 = std::clamp(detection.bbox[3], 0.0f, static_cast<float>(camera.height - 1));
    if (x2 - x1 < 2.0f || y2 - y1 < 2.0f) return result;

    const cv::Point2f center(0.5f * (x1 + x2), 0.5f * (y1 + y2));
    const cv::Point2f center_u = undistortPixel(center, camera);
    const cv::Point2f left_u = undistortPixel({x1, center.y}, camera);
    const cv::Point2f right_u = undistortPixel({x2, center.y}, camera);
    const cv::Point2f top_u = undistortPixel({center.x, y1}, camera);
    const cv::Point2f bottom_u = undistortPixel({center.x, y2}, camera);
    const cv::Point3f center_ray = normalized(deproject(center_u, 1.0f, camera));

    const float radius_m = 0.5f * config_.object_diameter_m;
    std::vector<float> mono_center_ranges;
    const auto add_angular_diameter = [&](const cv::Point2f& first,
                                          const cv::Point2f& second) {
        const cv::Point3f first_ray = deproject(first, 1.0f, camera);
        const cv::Point3f second_ray = deproject(second, 1.0f, camera);
        const float angular_diameter = angleBetween(first_ray, second_ray) *
                                       config_.bbox_scale;
        const float half_angle = 0.5f * angular_diameter;
        if (half_angle > 1e-5f && half_angle < 1.4f) {
            mono_center_ranges.push_back(radius_m / std::sin(half_angle));
        }
    };
    add_angular_diameter(left_u, right_u);
    add_angular_diameter(top_u, bottom_u);
    if (!mono_center_ranges.empty()) {
        result.mono_center_z_m = median(mono_center_ranges) * center_ray.z;
        result.mono_surface_z_m = std::max(
            0.0f, (median(mono_center_ranges) - radius_m) * center_ray.z);
        result.mono_valid = std::isfinite(result.mono_center_z_m) &&
                            result.mono_center_z_m >= config_.min_depth_m &&
                            result.mono_center_z_m <= config_.max_depth_m;
    }

    if (aligned_depth == nullptr || !aligned_depth->valid()) return result;

    const float radius_x = 0.5f * (x2 - x1) *
        std::clamp(config_.sample_radius_scale, 0.1f, 0.95f);
    const float radius_y = 0.5f * (y2 - y1) *
        std::clamp(config_.sample_radius_scale, 0.1f, 0.95f);

    std::vector<float> center_ranges;
    std::vector<cv::Point> candidate_pixels;

    auto add_sample = [&](float px, float py) {
        const int ix = std::clamp(static_cast<int>(std::lround(px)), 0, aligned_depth->width - 1);
        const int iy = std::clamp(static_cast<int>(std::lround(py)), 0, aligned_depth->height - 1);
        const float depth_z = localMedianDepth(*aligned_depth, ix, iy,
                                               config_.patch_radius,
                                               config_.min_depth_m,
                                               config_.max_depth_m);
        if (!std::isfinite(depth_z)) return;

        const cv::Point2f pixel_u = undistortPixel({static_cast<float>(ix),
                                                     static_cast<float>(iy)}, camera);
        const cv::Point3f point = deproject(pixel_u, depth_z, camera);
        const float point_range = norm3(point);
        if (point_range <= 0.0f) return;
        const cv::Point3f point_ray = normalized(point);
        const float cosine = std::clamp(dot3(point_ray, center_ray), -1.0f, 1.0f);
        const float transverse_sq = point_range * point_range *
                                    std::max(0.0f, 1.0f - cosine * cosine);
        const float discriminant = radius_m * radius_m - transverse_sq;
        if (discriminant < 0.0f) return;

        // The camera observes the near sphere surface, so the center is the
        // farther of the two intersections along the bbox-center ray.
        const float center_range = point_range * cosine + std::sqrt(discriminant);
        if (center_range < config_.min_depth_m || center_range > config_.max_depth_m) return;
        if (result.mono_valid) {
            const float mono_center_range = result.mono_center_z_m /
                                            std::max(0.1f, center_ray.z);
            const float consistency_gate = std::max(
                config_.mono_consistency_absolute_m,
                config_.mono_consistency_relative * mono_center_range);
            if (std::abs(center_range - mono_center_range) > consistency_gate) return;
        }
        center_ranges.push_back(center_range);
        candidate_pixels.emplace_back(ix, iy);
    };

    add_sample(center.x, center.y);
    const int rings = std::max(1, config_.sample_rings);
    const int samples_per_ring = std::max(4, config_.samples_per_ring);
    constexpr float kTwoPi = 6.2831853071795864769f;
    for (int ring = 1; ring <= rings; ++ring) {
        const float ring_scale = static_cast<float>(ring) / static_cast<float>(rings);
        const float phase = (ring % 2 == 0) ? (0.5f * kTwoPi / samples_per_ring) : 0.0f;
        for (int index = 0; index < samples_per_ring; ++index) {
            const float angle = phase + kTwoPi * index / samples_per_ring;
            add_sample(center.x + radius_x * ring_scale * std::cos(angle),
                       center.y + radius_y * ring_scale * std::sin(angle));
        }
    }

    if (static_cast<int>(center_ranges.size()) < config_.min_valid_samples) return result;

    const float initial_center = median(center_ranges);
    const float initial_spread = robustSpread(center_ranges, initial_center);
    const float gate = std::max(0.015f, config_.mad_scale * std::max(initial_spread, 0.005f));

    std::vector<float> filtered_ranges;
    std::vector<cv::Point> filtered_pixels;
    for (size_t i = 0; i < center_ranges.size(); ++i) {
        if (std::abs(center_ranges[i] - initial_center) <= gate) {
            filtered_ranges.push_back(center_ranges[i]);
            filtered_pixels.push_back(candidate_pixels[i]);
        }
    }
    if (static_cast<int>(filtered_ranges.size()) < config_.min_valid_samples) return result;

    const float center_range = median(filtered_ranges);
    const float spread = robustSpread(filtered_ranges, center_range);
    if (!std::isfinite(spread) || spread > config_.max_center_spread_m) return result;

    result.depth_valid = true;
    result.center_range_m = center_range;
    result.surface_range_m = std::max(0.0f, center_range - radius_m);
    result.center_xyz_m = center_ray * center_range;
    result.surface_z_m = result.surface_range_m * center_ray.z;
    result.center_spread_m = spread;
    result.valid_samples = static_cast<int>(filtered_ranges.size());
    result.accepted_pixels = std::move(filtered_pixels);
    return result;
}

void drawDistanceMeasurements(cv::Mat& image,
                              const std::vector<Detection>& detections,
                              const std::vector<DistanceMeasurement>& measurements,
                              bool draw_sample_points)
{
    const size_t count = std::min(detections.size(), measurements.size());
    for (size_t index = 0; index < count; ++index) {
        const auto& detection = detections[index];
        const auto& measurement = measurements[index];
        if (draw_sample_points && measurement.depth_valid) {
            for (const auto& pixel : measurement.accepted_pixels) {
                cv::circle(image, pixel, 2, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
            }
        }

        std::vector<std::string> lines;
        if (measurement.depth_valid) {
            lines.push_back("RGB-D Zc=" + formatMeters(measurement.center_xyz_m.z) +
                            " Zs=" + formatMeters(measurement.surface_z_m));
            lines.push_back("XYZ=" + formatMeters(measurement.center_xyz_m.x) + "," +
                            formatMeters(measurement.center_xyz_m.y) + "," +
                            formatMeters(measurement.center_xyz_m.z) +
                            " n=" + std::to_string(measurement.valid_samples));
        }
        if (measurement.mono_valid) {
            lines.push_back("Mono center=" + formatMeters(measurement.mono_center_z_m) +
                            " surface=" + formatMeters(measurement.mono_surface_z_m));
        }
        if (lines.empty()) continue;

        constexpr double font_scale = 0.45;
        constexpr int thickness = 1;
        constexpr int line_height = 18;
        constexpr int padding = 5;
        int panel_width = 0;
        for (const auto& line : lines) {
            int baseline = 0;
            panel_width = std::max(panel_width,
                cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX,
                                font_scale, thickness, &baseline).width);
        }
        panel_width = std::min(image.cols, panel_width + padding * 2);
        const int panel_height = std::min(
            image.rows, static_cast<int>(lines.size()) * line_height + padding * 2);
        const int x = std::clamp(static_cast<int>(detection.bbox[0]),
                                 0, std::max(0, image.cols - panel_width));
        const int bbox_bottom = std::clamp(static_cast<int>(detection.bbox[3]), 0, image.rows);
        const int bbox_top = std::clamp(static_cast<int>(detection.bbox[1]), 0, image.rows);
        const int panel_y = (bbox_bottom + panel_height + 2 <= image.rows)
            ? bbox_bottom + 2
            : std::max(0, bbox_top - panel_height - 2);

        cv::Mat overlay = image.clone();
        cv::rectangle(overlay, cv::Rect(x, panel_y, panel_width, panel_height),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(overlay, 0.55, image, 0.45, 0.0, image);

        int y = panel_y + padding + 12;
        for (const auto& line : lines) {
            cv::putText(image, line, cv::Point(x + padding, y), cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
            y += line_height;
        }
    }
}

cv::Mat colorizeDepth(const DepthImageView& depth,
                      float min_depth_m,
                      float max_depth_m)
{
    if (!depth.valid() || max_depth_m <= min_depth_m) return {};
    cv::Mat gray(depth.height, depth.width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < depth.height; ++y) {
        auto* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < depth.width; ++x) {
            const float value = depth.at(x, y);
            if (value >= min_depth_m && value <= max_depth_m) {
                const float normalized_value = 1.0f -
                    (value - min_depth_m) / (max_depth_m - min_depth_m);
                row[x] = static_cast<uint8_t>(
                    std::clamp(normalized_value * 255.0f, 0.0f, 255.0f));
            }
        }
    }
    cv::Mat colored;
    cv::applyColorMap(gray, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), gray == 0);
    return colored;
}

#ifdef HAVE_REALSENSE2

namespace {

bool setSensorOption(rs2::sensor& sensor,
                     rs2_option option,
                     float value,
                     bool strict,
                     std::string* error)
{
    if (!sensor.supports(option)) {
        if (!strict) return true;
        if (error) *error = std::string("RealSense option unsupported: ") + rs2_option_to_string(option);
        return false;
    }
    sensor.set_option(option, value);
    const float actual = sensor.get_option(option);
    const auto range = sensor.get_option_range(option);
    const float tolerance = std::max(range.step * 0.51f, 1e-3f);
    if (std::abs(actual - value) > tolerance && strict) {
        if (error) {
            *error = std::string("RealSense option readback mismatch: ") +
                     rs2_option_to_string(option) + " target=" + std::to_string(value) +
                     " actual=" + std::to_string(actual);
        }
        return false;
    }
    return true;
}

CameraModel cameraModelFromIntrinsics(const rs2_intrinsics& intrinsics)
{
    CameraModel camera;
    camera.width = intrinsics.width;
    camera.height = intrinsics.height;
    camera.fx = intrinsics.fx;
    camera.fy = intrinsics.fy;
    camera.cx = intrinsics.ppx;
    camera.cy = intrinsics.ppy;
    camera.distortion = {intrinsics.coeffs[0], intrinsics.coeffs[1],
                         intrinsics.coeffs[2], intrinsics.coeffs[3],
                         intrinsics.coeffs[4]};
    switch (intrinsics.model) {
        case RS2_DISTORTION_NONE:
            camera.distortion_model = CameraModel::DistortionModel::None;
            break;
        case RS2_DISTORTION_BROWN_CONRADY:
        case RS2_DISTORTION_MODIFIED_BROWN_CONRADY:
            camera.distortion_model = CameraModel::DistortionModel::BrownConrady;
            break;
        case RS2_DISTORTION_INVERSE_BROWN_CONRADY:
            camera.distortion_model = CameraModel::DistortionModel::InverseBrownConrady;
            break;
        default:
            camera.distortion_model = CameraModel::DistortionModel::Unsupported;
            break;
    }
    return camera;
}

}  // namespace

bool RealSenseRgbdCapture::start(const RealSenseCaptureConfig& config,
                                std::string* error)
{
    try {
        rs2::config stream_config;
        if (!config.serial.empty()) stream_config.enable_device(config.serial);
        stream_config.enable_stream(RS2_STREAM_COLOR, config.width, config.height,
                                    RS2_FORMAT_BGR8, config.fps);
        stream_config.enable_stream(RS2_STREAM_DEPTH, config.width, config.height,
                                    RS2_FORMAT_Z16, config.fps);
        profile_ = pipeline_.start(stream_config);

        // Prefer librealsense global time so the frame timestamp shares the
        // system/ROS epoch used by odometry and the control node.
        for (auto sensor : profile_.get_device().query_sensors()) {
            if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
                try {
                    sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
                } catch (const rs2::error&) {
                    // Some firmware exposes the option as read-only. The
                    // hardware-clock fallback below still supplies a mapped timestamp.
                }
            }
        }

        auto depth_sensor = profile_.get_device().first<rs2::depth_sensor>();
        depth_scale_m_ = depth_sensor.get_depth_scale();
        use_spatial_filter_ = config.spatial_filter;
        use_temporal_filter_ = config.temporal_filter;
        use_hole_filling_filter_ = config.hole_filling_filter;
        allow_hardware_time_fallback_ = config.allow_hardware_time_fallback;
        timestamp_fallback_warmup_frames_ =
            std::max(1, config.timestamp_fallback_warmup_frames);
        max_timestamp_uncertainty_s_ =
            std::max(0.0, config.max_timestamp_uncertainty_s);
        if (!configureColorSensor(config, error)) {
            pipeline_.stop();
            return false;
        }
        if (config.imu_enabled && !startMotionSensor(config, error)) {
            pipeline_.stop();
            return false;
        }
        started_ = true;
        return true;
    } catch (const rs2::error& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

bool RealSenseRgbdCapture::configureColorSensor(const RealSenseCaptureConfig& config,
                                                std::string* error)
{
    for (auto sensor : profile_.get_device().query_sensors()) {
        if (!sensor.is<rs2::color_sensor>()) continue;
        if (!setSensorOption(sensor, RS2_OPTION_ENABLE_AUTO_EXPOSURE,
                             config.auto_exposure ? 1.0f : 0.0f,
                             config.strict_controls, error)) return false;
        if (!config.auto_exposure &&
            !setSensorOption(sensor, RS2_OPTION_EXPOSURE,
                             std::max(1.0f, config.exposure_us / 100.0f),
                             config.strict_controls, error)) return false;
        if (!setSensorOption(sensor, RS2_OPTION_GAIN, config.gain,
                             config.strict_controls, error)) return false;
        if (!setSensorOption(sensor, RS2_OPTION_AUTO_EXPOSURE_PRIORITY,
                             config.auto_exposure_priority ? 1.0f : 0.0f,
                             config.strict_controls, error)) return false;
        if (!setSensorOption(sensor, RS2_OPTION_POWER_LINE_FREQUENCY,
                             config.power_line_frequency,
                             config.strict_controls, error)) return false;
        if (!setSensorOption(sensor, RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE,
                             config.auto_white_balance ? 1.0f : 0.0f,
                             config.strict_controls, error)) return false;
        if (!config.auto_white_balance &&
            !setSensorOption(sensor, RS2_OPTION_WHITE_BALANCE,
                             config.white_balance_temperature,
                             config.strict_controls, error)) return false;
        return true;
    }
    if (error) *error = "RealSense color sensor not found";
    return false;
}

bool RealSenseRgbdCapture::startMotionSensor(
    const RealSenseCaptureConfig& config, std::string* error)
{
    try {
        imu_callback_ = config.imu_callback;
        for (auto sensor : profile_.get_device().query_sensors()) {
            std::optional<rs2::stream_profile> gyro_profile;
            std::optional<rs2::stream_profile> accel_profile;
            int gyro_delta = std::numeric_limits<int>::max();
            int accel_delta = std::numeric_limits<int>::max();
            for (const auto& stream : sensor.get_stream_profiles()) {
                if (stream.format() != RS2_FORMAT_MOTION_XYZ32F) continue;
                if (stream.stream_type() == RS2_STREAM_GYRO) {
                    const int delta = std::abs(stream.fps() - config.gyro_fps);
                    if (delta < gyro_delta) {
                        gyro_profile = stream;
                        gyro_delta = delta;
                    }
                } else if (stream.stream_type() == RS2_STREAM_ACCEL) {
                    const int delta = std::abs(stream.fps() - config.accel_fps);
                    if (delta < accel_delta) {
                        accel_profile = stream;
                        accel_delta = delta;
                    }
                }
            }
            if (!gyro_profile || !accel_profile) continue;
            if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
                try {
                    sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
                } catch (const rs2::error&) {
                    // Hardware-clock timestamps are mapped below when required.
                }
            }
            std::vector<rs2::stream_profile> profiles{
                *gyro_profile, *accel_profile
            };
            sensor.open(profiles);
            sensor.start([this](rs2::frame frame) { onMotionFrame(frame); });
            motion_sensor_ = sensor;
            return true;
        }
        if (error) *error = "D435i gyro/accelerometer motion sensor not found";
        return false;
    } catch (const rs2::error& exception) {
        if (error) *error = std::string("D435i IMU start failed: ") + exception.what();
        motion_sensor_.reset();
        return false;
    }
}

double RealSenseRgbdCapture::motionTimestampS(const rs2::frame& frame)
{
    const double device_timestamp_s = frame.get_timestamp() * 1e-3;
    const auto domain = frame.get_frame_timestamp_domain();
    if (domain == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME ||
        domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME) {
        return device_timestamp_s;
    }
    const double system_now_s = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const double observed_offset = system_now_s - device_timestamp_s;
    if (!motion_device_to_system_offset_s_ ||
        observed_offset < *motion_device_to_system_offset_s_) {
        motion_device_to_system_offset_s_ = observed_offset;
    } else {
        *motion_device_to_system_offset_s_ +=
            0.001 * (observed_offset - *motion_device_to_system_offset_s_);
    }
    return device_timestamp_s + *motion_device_to_system_offset_s_;
}

void RealSenseRgbdCapture::onMotionFrame(const rs2::frame& frame)
{
    const rs2::motion_frame motion = frame.as<rs2::motion_frame>();
    if (!motion) return;
    const rs2_vector data = motion.get_motion_data();
    D435ImuSample publish_sample;
    bool should_publish = false;
    {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        const double timestamp_s = motionTimestampS(frame);
        const auto stream = frame.get_profile().stream_type();
        if (stream == RS2_STREAM_GYRO) {
            latest_imu_.angular_velocity = {data.x, data.y, data.z};
            latest_imu_.gyro_timestamp_s = timestamp_s;
            latest_imu_.gyro_valid = true;
            if (latest_imu_.accel_valid) {
                publish_sample = latest_imu_;
                should_publish = true;
            }
        } else if (stream == RS2_STREAM_ACCEL) {
            latest_imu_.linear_acceleration = {data.x, data.y, data.z};
            latest_imu_.accel_timestamp_s = timestamp_s;
            latest_imu_.accel_valid = true;
        }
    }
    if (should_publish && imu_callback_) {
        imu_callback_(publish_sample);
    }
}

bool RealSenseRgbdCapture::read(RgbdFrame& output, std::string* error)
{
    if (!started_) {
        if (error) *error = "RealSense pipeline is not started";
        return false;
    }
    try {
        rs2::frameset frames = pipeline_.wait_for_frames();
        const rs2::video_frame captured_color = frames.get_color_frame();
        const rs2::depth_frame captured_depth = frames.get_depth_frame();
        if (!captured_color || !captured_depth) {
            if (error) *error = "RealSense color/depth frame is missing";
            return false;
        }
        const double color_device_timestamp_s = captured_color.get_timestamp() * 1e-3;
        const double depth_device_timestamp_s = captured_depth.get_timestamp() * 1e-3;
        const auto color_domain = captured_color.get_frame_timestamp_domain();
        const auto depth_domain = captured_depth.get_frame_timestamp_domain();
        const double system_now_s = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        output.timing = {};
        const bool color_is_system =
            color_domain == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME ||
            color_domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME;
        const bool depth_is_system =
            depth_domain == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME ||
            depth_domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME;
        if (color_is_system && depth_is_system) {
            output.timing.color_capture_timestamp_s = color_device_timestamp_s;
            output.timing.depth_capture_timestamp_s = depth_device_timestamp_s;
            output.timing.timestamp_domain =
                color_domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME &&
                        depth_domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME
                    ? RgbdTimestampDomain::GlobalTime
                    : RgbdTimestampDomain::SystemTime;
            output.timing.timestamp_uncertainty_ns = 100000;
            output.timing.timestamp_mapping_valid = true;
        } else if (!color_is_system && !depth_is_system &&
                   color_domain == depth_domain) {
            const double newest_device_timestamp_s =
                std::max(color_device_timestamp_s, depth_device_timestamp_s);
            const double observed_offset = system_now_s - newest_device_timestamp_s;
            if (!device_to_system_offset_s_) {
                device_to_system_offset_s_ = observed_offset;
                device_timestamp_samples_ = 1;
            } else {
                // A lower observed offset is a better low-queue-delay sample.
                // Follow increases slowly so clock drift is tracked without
                // treating USB/driver queueing as clock offset.
                if (observed_offset < *device_to_system_offset_s_) {
                    *device_to_system_offset_s_ = observed_offset;
                } else {
                    *device_to_system_offset_s_ +=
                        0.001 * (observed_offset - *device_to_system_offset_s_);
                }
                ++device_timestamp_samples_;
            }
            output.timing.color_capture_timestamp_s =
                color_device_timestamp_s + *device_to_system_offset_s_;
            output.timing.depth_capture_timestamp_s =
                depth_device_timestamp_s + *device_to_system_offset_s_;
            output.timing.timestamp_domain = RgbdTimestampDomain::HardwareClock;
            const double residual_s =
                std::max(0.0, observed_offset - *device_to_system_offset_s_);
            output.timing.timestamp_uncertainty_ns = static_cast<uint64_t>(
                std::ceil(std::max(0.0005, residual_s) * 1e9));
            output.timing.timestamp_mapping_valid =
                allow_hardware_time_fallback_ &&
                device_timestamp_samples_ >= timestamp_fallback_warmup_frames_ &&
                residual_s <= max_timestamp_uncertainty_s_;
        } else {
            // Mixed timestamp domains cannot prove RGB/depth simultaneity.
            // Keep the color header in the system epoch when possible, but
            // mark the mapping invalid so the controller cannot use RGB-D.
            const double hardware_timestamp_s = color_is_system
                ? depth_device_timestamp_s : color_device_timestamp_s;
            const double observed_offset = system_now_s - hardware_timestamp_s;
            if (!device_to_system_offset_s_ ||
                observed_offset < *device_to_system_offset_s_) {
                device_to_system_offset_s_ = observed_offset;
            }
            output.timing.color_capture_timestamp_s = color_is_system
                ? color_device_timestamp_s
                : color_device_timestamp_s + *device_to_system_offset_s_;
            output.timing.depth_capture_timestamp_s = depth_is_system
                ? depth_device_timestamp_s
                : depth_device_timestamp_s + *device_to_system_offset_s_;
            output.timing.timestamp_domain = RgbdTimestampDomain::Unknown;
            output.timing.timestamp_uncertainty_ns = 0xFFFFFFFFFFFFFFFFULL;
            output.timing.timestamp_mapping_valid = false;
        }
        output.timing.rgb_depth_delta_ns = static_cast<int64_t>(std::llround(
            (output.timing.depth_capture_timestamp_s -
             output.timing.color_capture_timestamp_s) * 1e9));
        output.capture_timestamp_s = output.timing.color_capture_timestamp_s;
        frames = align_to_color_.process(frames);
        rs2::video_frame color = frames.get_color_frame();
        rs2::depth_frame depth = frames.get_depth_frame();
        if (!color || !depth) {
            if (error) *error = "Aligned RealSense color/depth frame is missing";
            return false;
        }
        if (use_spatial_filter_) depth = spatial_filter_.process(depth).as<rs2::depth_frame>();
        if (use_temporal_filter_) depth = temporal_filter_.process(depth).as<rs2::depth_frame>();
        if (use_hole_filling_filter_) depth = hole_filling_filter_.process(depth).as<rs2::depth_frame>();

        const auto color_profile = color.get_profile().as<rs2::video_stream_profile>();
        output.camera = cameraModelFromIntrinsics(color_profile.get_intrinsics());
        output.color_bgr = cv::Mat(color.get_height(), color.get_width(), CV_8UC3,
                                   const_cast<void*>(color.get_data()),
                                   color.get_stride_in_bytes()).clone();
        output.aligned_depth = depth;
        output.depth_view.data = reinterpret_cast<const uint16_t*>(depth.get_data());
        output.depth_view.width = depth.get_width();
        output.depth_view.height = depth.get_height();
        output.depth_view.stride_bytes = depth.get_stride_in_bytes();
        output.depth_view.depth_scale_m = depth_scale_m_;
        return true;
    } catch (const rs2::error& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

void RealSenseRgbdCapture::stop()
{
    if (!started_) return;
    if (motion_sensor_) {
        try {
            motion_sensor_->stop();
            motion_sensor_->close();
        } catch (const rs2::error&) {
            // Continue stopping the video pipeline even if the IMU is gone.
        }
        motion_sensor_.reset();
    }
    pipeline_.stop();
    started_ = false;
    device_to_system_offset_s_.reset();
    device_timestamp_samples_ = 0;
    motion_device_to_system_offset_s_.reset();
    imu_callback_ = {};
    {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        latest_imu_ = {};
    }
}

#endif  // HAVE_REALSENSE2

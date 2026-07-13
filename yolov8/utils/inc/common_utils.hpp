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

#pragma once

#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <opencv2/freetype.hpp>
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hb_ucp_status.h"

using json = nlohmann::json;

/**
 * @def HBDNN_CHECK_SUCCESS
 * @brief Check a HBDNN API call for success and throw std::runtime_error on failure.
 * @param func_call  HBDNN API call expression.                        // (in)
 * @param context_str Text to append for easier debugging.             // (in)
 */
#define HBDNN_CHECK_SUCCESS(func_call, context_str)                                                     \
    do {                                                                                                \
        int32_t __err_code = (func_call);                                                               \
        if (__err_code != 0) {                                                                          \
            const char* __err_desc = hbUCPGetErrorDesc(__err_code);                                     \
            throw std::runtime_error(std::string("DNN Error (code: ") + std::to_string(__err_code) +    \
                                     ", desc: " + __err_desc + ") " + (context_str));                   \
        }                                                                                               \
    } while (0)

/**
 * @def HBUCP_CHECK_SUCCESS
 * @brief Check a HBUCP API call for success and throw std::runtime_error on failure.
 * @param func_call  HBUCP API call expression.                       // (in)
 * @param context_str Text to append for easier debugging.            // (in)
 */
#define HBUCP_CHECK_SUCCESS(func_call, context_str)                                                     \
    do {                                                                                                \
        int32_t __err_code = (func_call);                                                               \
        if (__err_code != 0) {                                                                          \
            const char* __err_desc = hbUCPGetErrorDesc(__err_code);                                     \
            throw std::runtime_error(std::string("UCP Error (code: ") + std::to_string(__err_code) +    \
                                     ", desc: " + __err_desc + ") " + (context_str));                   \
        }                                                                                               \
    } while (0)

/**
 * @brief Small, visually distinct color palette (BGR).
 * @note Colors repeat cyclically when class_id exceeds the palette length.
 */
inline std::vector<cv::Scalar> rdk_colors = {
    {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255},
    {49, 210, 207}, {10, 249, 72},  {23, 204, 146}, {134, 219, 61},
    {52, 147, 26},  {187, 212, 0},  {168, 153, 44}, {255, 194, 0},
    {147, 69, 52},  {255, 115, 100},{236, 24, 0},   {255, 56, 132},
    {133, 0, 82},   {255, 56, 203}, {200, 149, 255},{199, 55, 255}
};

/**
 * @brief Top-k classification item.
 */
typedef struct Classification {
  int         id;           // Class id
  float       score;        // Logit or probability
  const char* class_name;   // Optional pointer to label string (may be null)

  Classification() : class_name(nullptr), id(0), score(0.0f) {}
  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  ~Classification() = default;
} Classification;

/**
 * @brief Generic detection with an axis-aligned bounding box.
 */
struct Detection {
    float bbox[4];           // x1, y1, x2, y2 (pixels, inclusive)
    float score;             // Confidence score (e.g., obj*cls)
    int   class_id;          // Argmax class id
};

/**
 * @brief A single keypoint.
 */
struct Keypoint {
    float x;                 // X coordinate (pixels)
    float y;                 // Y coordinate (pixels)
    float score;             // Raw score/logit (apply sigmoid if needed)
};

/**
 * @brief Fast sigmoid approximation using std::exp.
 * @param[in] x Raw value (logit).
 * @return float Sigmoid(x) in (0,1).
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Numerically stable softmax for a short vector.
 * @param[in]  input  Pointer to input array (length = len).
 * @param[out] output Pointer to output array (length = len).
 * @param[in]  len    Vector length (default 16).
 */
inline void softmax(const float* input, float* output, int len = 16)
{
    if (len <= 0) return;
    float max_val = *std::max_element(input, input + len); // for stability
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    if (sum == 0.0f) sum = 1.0f; // avoid division by zero
    for (int i = 0; i < len; ++i) {
        output[i] /= sum;
    }
}

/**
 * @brief Load an image from disk in BGR color space.
 *
 * @param[in] image_file Absolute or relative path to the image file.
 * @return cv::Mat Loaded image in BGR (empty if loading fails).
 */
cv::Mat load_bgr_image(const std::string& image_file);

/**
 * @brief Load an ImageNet-1000 label map dumped as "{123: 'label'}" text.
 *
 * Expected file format (single object): {0: 'label0', 1: 'label1', ...}
 *
 * @param[in] label_path Path to label file.
 * @return std::map<int,std::string> Mapping from class id to label string. Returns empty map on failure.
 */
std::map<int, std::string> load_imagenet1000_label_map(const std::string &label_path);

/**
 * @brief Load labels line-by-line from a plain text file.
 *
 * Each non-empty line is considered one label. Trailing '\r' is stripped.
 *
 * @param[in] filename Path to the label text file.
 * @return std::vector<std::string> All labels in order of appearance.
 */
std::vector<std::string> load_linewise_labels(const std::string& filename);

/**
 * @brief Load a token->id vocabulary from JSON and produce id->token table.
 *
 * The JSON is expected to be an object: {"<token>": <id>, ... }.
 *
 * @param[in] vocab_file Path to JSON vocabulary file.
 * @return std::vector<std::string> Vector where index is token id and value is token string.
 *
 * @throws std::runtime_error If the file cannot be opened or parsed.
 */
std::vector<std::string> load_id2token(const std::string& vocab_file);

/**
 * @brief Print top-k classification results in a friendly format.
 *
 * @param[in] top_k_cls Vector of top-k (id, score) pairs.
 * @param[in] label_map Optional mapping from id to human-readable label.
 */
void print_topk_results(const std::vector<Classification>& top_k_cls,
                        const std::map<int, std::string>& label_map);

/**
 * @brief Draw axis-aligned bounding boxes with class label and score.
 *
 * @param[in,out] image  BGR image to draw on.
 * @param[in]     detections Vector of detections (xyxy, score, class_id).
 * @param[in]     class_names Vector of class names; indexed by class_id.
 * @param[in]     colors Color palette for classes; used cyclically.
 */
void draw_boxes(cv::Mat& image,
                const std::vector<Detection>& detections,
                const std::vector<std::string>& class_names,
                const std::vector<cv::Scalar>& colors);

/**
 * @brief Blend segmentation masks into the image inside each detection's bbox.
 *
 * @param[in,out] image       BGR image to draw on.
 * @param[in]     detections  Detections with xyxy boxes (used as ROI).
 * @param[in]     masks       Binary masks aligned to detections (1=foreground).
 * @param[in]     colors      Color palette for classes.
 * @param[in]     alpha       Alpha blending factor (0=orig, 1=color).
 */
void draw_masks(cv::Mat& image,
                const std::vector<Detection>& detections,
                const std::vector<cv::Mat>& masks,
                const std::vector<cv::Scalar>& colors,
                float alpha = 0.3f);

/**
 * @brief Draw contour lines (polygon outlines) for segmentation masks.
 *
 * @param[in,out] img     Image to draw on.
 * @param[in]     detections Detected boxes; used to shift local contours to image coords.
 * @param[in]     masks   Binary masks (per detection) in local bbox coordinates.
 * @param[in]     colors  Color palette used cyclically by class id.
 * @param[in]     thickness Line thickness in pixels.
 */
void draw_contours(cv::Mat& img,
                   const std::vector<Detection>& detections,
                   const std::vector<cv::Mat>& masks,
                   const std::vector<cv::Scalar>& colors,
                   int thickness = 2);

/**
 * @brief Draw keypoints for pose estimation results.
 *
 * A keypoint is drawn if its score (pre-sigmoid) passes the given threshold after sigmoid.
 * Two concentric circles are used for a simple highlight.
 *
 * @param[in,out] image           Image to draw on.
 * @param[in]     kpts            Keypoints per instance; shape: N x K.
 * @param[in]     kpt_conf_thresh Display threshold applied to sigmoid(score).
 * @param[in]     radius_outer    Radius of the outer filled circle.
 * @param[in]     radius_inner    Radius of the inner filled circle.
 */
void draw_keypoints(cv::Mat& image,
                    const std::vector<std::vector<Keypoint>>& kpts,
                    float kpt_conf_thresh = 0.5f,
                    int radius_outer = 5,
                    int radius_inner = 2);

/**
 * @brief Draw multiple text strings onto an image using FreeType (TrueType) fonts.
 *
 * Text origin for each string is taken from the first vertex of the corresponding box.
 * Coordinates are clamped to the image canvas to avoid out-of-bound drawing.
 *
 * @param[in] img        Background image (BGR).
 * @param[in] texts      Texts to draw (aligned with boxes).
 * @param[in] boxes      Polygon boxes; only boxes[i][0] is used as the anchor point.
 * @param[in] font_path  Path to a .ttf font file.
 * @param[in] font_size  Font size in FreeType units.
 * @param[in] color      Text color (BGR).
 * @param[in] thickness  Stroke thickness.
 * @return cv::Mat A copy of @p img with texts rendered (original is unchanged).
 */
cv::Mat draw_text(cv::Mat img,
                  const std::vector<std::string>& texts,
                  const std::vector<std::vector<cv::Point>>& boxes,
                  const std::string& font_path,
                  int font_size,
                  cv::Scalar color,
                  int thickness);

/**
 * @brief Draw closed polygon boxes on an image.
 *
 * @param[in] img        Input image (BGR).
 * @param[in] bboxes     List of polygons (each is a vector of points).
 * @param[in] color      Line color (BGR).
 * @param[in] thickness  Line thickness.
 * @return cv::Mat A copy of the input image with polygons drawn.
 */
cv::Mat draw_polygon_boxes(const cv::Mat& img,
                           const std::vector<std::vector<cv::Point>>& bboxes,
                           const cv::Scalar& color = cv::Scalar(128, 240, 128),
                           int thickness = 3);

/**
 * @brief Convert a label mask (H x W, int32 class ids) to a color image.
 *
 * @param[in] seg_mask  Per-pixel class ids (CV_32S).
 * @param[in] colors    Color palette; index is the class id.
 * @return cv::Mat BGR colorized image (CV_8UC3).
 */
cv::Mat colorize_mask(const cv::Mat& seg_mask, const std::vector<cv::Scalar>& colors);

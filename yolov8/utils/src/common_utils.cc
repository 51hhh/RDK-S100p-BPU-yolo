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

#include "common_utils.hpp"

/**
 * @brief Load an image from disk in BGR color space.
 *
 * @param[in] image_file Absolute or relative path to the image file.
 * @return cv::Mat Loaded image in BGR (empty if loading fails).
 */
cv::Mat load_bgr_image(const std::string& image_file) {
    // Read as 3-channel BGR image
    cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);

    // Warn if failed
    if (bgr_mat.empty()) {
        std::cerr << "ERROR: Failed to load image: " << image_file << std::endl;
    }
    return bgr_mat;
}

/**
 * @brief Load an ImageNet-1000 label map dumped as "{123: 'label'}" text.
 *
 * Expected file format (single object): {0: 'label0', 1: 'label1', ...}
 *
 * @param[in] label_path Path to label file.
 * @return std::map<int,std::string> Mapping from class id to label string. Returns empty map on failure.
 */
std::map<int, std::string> load_imagenet1000_label_map(const std::string &label_path) {
  std::map<int, std::string> label_map;

  std::ifstream infile(label_path);
  if (!infile.is_open()) {
    std::cerr << "Failed to open label file: " << label_path << std::endl;
    return label_map;  // empty
  }

  // Read entire file into a string
  std::string content((std::istreambuf_iterator<char>(infile)),
                      std::istreambuf_iterator<char>());
  infile.close();

  // Trim possible leading '{' and trailing '}'
  if (!content.empty() && content.front() == '{') content.erase(0, 1);
  if (!content.empty() && content.back() == '}') content.pop_back();

  // Regex to match: "<spaces><id><spaces>:<spaces>'<label>'"
  std::regex entry_regex(R"(\s*(\d+)\s*:\s*'([^']*)')");
  auto begin = std::sregex_iterator(content.begin(), content.end(), entry_regex);
  auto end   = std::sregex_iterator();

  // Parse all matches
  for (auto it = begin; it != end; ++it) {
    int id = std::stoi((*it)[1].str());     // group 1: numeric id
    std::string label = (*it)[2].str();     // group 2: label text
    label_map[id] = label;
  }

  return label_map;
}

/**
 * @brief Load labels line-by-line from a plain text file.
 *
 * Each non-empty line is considered one label. Trailing '\r' is stripped.
 *
 * @param[in] filename Path to the label text file.
 * @return std::vector<std::string> All labels in order of appearance.
 */
std::vector<std::string> load_linewise_labels(const std::string& filename)
{
    std::vector<std::string> labels;
    std::ifstream infile(filename);
    std::string line;

    if (!infile.is_open()) {
        std::cerr << "Failed to open label file: " << filename << std::endl;
        return labels; // empty
    }

    while (std::getline(infile, line)) {
        if (!line.empty()) {
            // Remove trailing '\r' if present (Windows line endings)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            labels.push_back(line);
        }
    }
    return labels;
}

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
std::vector<std::string> load_id2token(const std::string& vocab_file)
{
    std::ifstream ifs(vocab_file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open vocab file: " + vocab_file);
    }

    json j;
    ifs >> j;  // Parse JSON to object

    // Find maximum id to size the vector
    int max_id = -1;
    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = it.value().get<int>();
        if (id > max_id) max_id = id;
    }

    // Allocate id->token vector
    std::vector<std::string> id2token(max_id + 1);

    // Reverse the mapping: token->id → id->token
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& token = it.key();
        int id = it.value().get<int>();
        if (id >= 0 && id <= max_id) {
            id2token[id] = token;
        }
    }

    return id2token;
}

/**
 * @brief Print top-k classification results in a friendly format.
 *
 * @param[in] top_k_cls Vector of top-k (id, score) pairs.
 * @param[in] label_map Optional mapping from id to human-readable label.
 */
void print_topk_results(const std::vector<Classification>& top_k_cls,
                               const std::map<int, std::string>& label_map)
{
    for (size_t i = 0; i < top_k_cls.size(); ++i) {
        const auto& item = top_k_cls[i];
        auto it = label_map.find(item.id);
        const char* label = (it != label_map.end()) ? it->second.c_str() : "Unknown";
        std::cout << "TOP " << i
           << ": label=" << label
           << ", prob=" << item.score
           << '\n';
    }
}

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
                const std::vector<cv::Scalar>& colors)
{
    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.bbox[0]); // top-left x
        int y1 = static_cast<int>(det.bbox[1]); // top-left y
        int x2 = static_cast<int>(det.bbox[2]); // bottom-right x
        int y2 = static_cast<int>(det.bbox[3]); // bottom-right y
        int class_id = det.class_id;

        // Pick color by class id
        cv::Scalar color = colors[class_id % colors.size()];
        // Compose label "name score"
        std::string label = class_names[class_id] + " " + cv::format("%.2f", det.score);

        // Draw rectangle
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Draw label text above (or at) the top edge
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int y_text = std::max(y1 - 5, label_size.height);

        cv::putText(image, label, cv::Point(x1, y_text),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

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
                float alpha)
{
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);

        // Guard: valid bbox and corresponding mask
        if (x2 <= x1 || y2 <= y1) continue;
        if (i >= masks.size()) continue;

        const cv::Mat& mask = masks[i];
        if (mask.empty()) continue;

        // Clamp bbox to image region
        x1 = std::clamp(x1, 0, image.cols);
        x2 = std::clamp(x2, 0, image.cols);
        y1 = std::clamp(y1, 0, image.rows);
        y2 = std::clamp(y2, 0, image.rows);
        if (x2 <= x1 || y2 <= y1) continue;

        // ROI view over the image for in-place blending
        cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        // Ensure mask matches ROI size
        cv::Mat mask_resized;
        if (mask.size() != roi.size()) {
            cv::resize(mask, mask_resized, roi.size(), 0, 0, cv::INTER_NEAREST);
        } else {
            mask_resized = mask;
        }

        // Convert mask to 8-bit for countNonZero / indexing
        cv::Mat mask_bool;
        mask_resized.convertTo(mask_bool, CV_8U);
        if (cv::countNonZero(mask_bool) == 0) continue;

        // Single-color patch for blending
        cv::Mat color_patch(roi.size(), roi.type(),
                            colors[det.class_id % colors.size()]);

        // Manual alpha blending on masked pixels
        for (int y = 0; y < roi.rows; ++y) {
            for (int x = 0; x < roi.cols; ++x) {
                if (mask_bool.at<uint8_t>(y, x)) {
                    cv::Vec3b& pixel = roi.at<cv::Vec3b>(y, x);
                    const cv::Vec3b color_pixel = color_patch.at<cv::Vec3b>(y, x);
                    // Blend each channel
                    for (int c = 0; c < 3; ++c) {
                        pixel[c] = static_cast<uint8_t>(
                            (1 - alpha) * pixel[c] + alpha * color_pixel[c]);
                    }
                }
            }
        }
    }
}

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
                   int thickness)
{
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);

        // Ensure mask exists
        if (i >= masks.size()) continue;
        const cv::Mat& mask = masks[i];
        if (mask.empty()) continue;

        // Extract contours from the local mask (ROI space)
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        // Shift local contour points by top-left ROI corner to image coords
        for (auto& contour : contours) {
            for (auto& pt : contour) {
                pt.x += x1;
                pt.y += y1;
            }
        }

        // Draw the polygon outlines
        cv::Scalar color = colors[det.class_id % colors.size()];
        cv::polylines(img, contours, true, color, thickness);
    }
}

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
void draw_keypoints(
    cv::Mat& image,
    const std::vector<std::vector<Keypoint>>& kpts,
    float kpt_conf_thresh,
    int radius_outer,
    int radius_inner)
{
    // Convert threshold on sigmoid(score) to raw-logit space for a cheap compare:
    // sigmoid(s) > t  <=>  s > -log(1/t - 1)
    const float kpt_conf_inverse = -std::log(1.0f / kpt_conf_thresh - 1.0f);

    for (size_t i = 0; i < kpts.size(); ++i) {
        const auto& instance = kpts[i];
        for (size_t j = 0; j < instance.size(); ++j) {
            const Keypoint& kp = instance[j];
            // Skip low-confidence points
            if (kp.score < kpt_conf_inverse) continue;

            // Integer pixel coordinates
            const int x = static_cast<int>(kp.x);
            const int y = static_cast<int>(kp.y);

            // Draw two concentric circles for better visibility
            cv::circle(image, {x, y}, radius_outer, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            cv::circle(image, {x, y}, radius_inner, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

            // Overlay keypoint index (shadow + bright)
            const std::string id_str = std::to_string(static_cast<int>(j));
            cv::putText(image, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            cv::putText(image, id_str, {x, y}, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    }
}

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
                  int thickness)
{
    // Work on a clone to keep the original image untouched
    cv::Mat output = img.clone();

    // Create FreeType renderer
    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    if (ft2.empty()) {
        std::cerr << "[Error] FreeType module not available in this OpenCV build.\n";
        return output;
    }
    ft2->loadFontData(font_path, 0);  // load TTF face

    for (size_t i = 0; i < texts.size() && i < boxes.size(); ++i) {
        const std::string& text = texts[i];

        // Skip empty strings
        if (text.empty()) {
            continue;
        }

        // Use first vertex of the polygon as text origin
        cv::Point org = boxes[i][0];

        // Clamp origin into the canvas (leave room for ascent)
        if (org.x < 0)            org.x = 0;
        if (org.y < font_size)    org.y = font_size;
        if (org.x >= img.cols)    org.x = img.cols - 1;
        if (org.y >= img.rows)    org.y = img.rows - 1;

        // Render UTF-8 text with anti-aliasing
        ft2->putText(output, text, org, font_size, color, thickness, cv::LINE_AA, true);
    }
    return output;
}

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
                           const cv::Scalar& color,
                           int thickness)
{
    // Work on a clone to keep the original image intact
    cv::Mat img_copy = img.clone();

    for (const auto& bbox : bboxes) {
        if (bbox.empty()) continue;

        // Draw a closed polygon (last point connected to first)
        cv::polylines(img_copy, bbox, true, color, thickness);
    }

    return img_copy;
}

/**
 * @brief Convert a label mask (H x W, int32 class ids) to a color image.
 *
 * @param[in] seg_mask  Per-pixel class ids (CV_32S).
 * @param[in] colors    Color palette; index is the class id.
 * @return cv::Mat BGR colorized image (CV_8UC3).
 */
cv::Mat colorize_mask(const cv::Mat& seg_mask,
                      const std::vector<cv::Scalar>& colors)
{
    CV_Assert(seg_mask.type() == CV_32S);      // ensure integer ids
    CV_Assert(!colors.empty());

    const int H = seg_mask.rows;
    const int W = seg_mask.cols;

    cv::Mat parsing_img(H, W, CV_8UC3);        // output BGR image

    // Map each class id to a palette color (out-of-range → black)
    for (int y = 0; y < H; ++y) {
        const int*    mask_row = seg_mask.ptr<int>(y);
        cv::Vec3b*    out_row  = parsing_img.ptr<cv::Vec3b>(y);

        for (int x = 0; x < W; ++x) {
            const int cls = mask_row[x];

            if (cls >= 0 && cls < static_cast<int>(colors.size())) {
                const cv::Scalar& c = colors[cls];  // B,G,R,(A)
                out_row[x] = cv::Vec3b(
                    static_cast<uchar>(c[0]),
                    static_cast<uchar>(c[1]),
                    static_cast<uchar>(c[2])
                );
            } else {
                out_row[x] = cv::Vec3b(0, 0, 0);    // unknown → black
            }
        }
    }
    return parsing_img;
}

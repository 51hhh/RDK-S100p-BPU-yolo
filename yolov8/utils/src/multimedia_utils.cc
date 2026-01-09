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

#include "multimedia_utils.hpp"
#include "sp_display.h"

/**
 * @brief Pack 8-bit RGB into ARGB8888 (alpha set to 0xFF).
 * @param[in] r  Red channel   [0..255].
 * @param[in] g  Green channel [0..255].
 * @param[in] b  Blue channel  [0..255].
 * @return uint32_t Packed color as 0xAARRGGBB with AA=0xFF.
 */
uint32_t rgb_to_argb8888(uint8_t r, uint8_t g, uint8_t b)
{
    // 0xFF for full opacity, then R,G,B in big-endian byte order
    return (0xFFu << 24) | (static_cast<uint32_t>(r) << 16)
                         | (static_cast<uint32_t>(g) << 8)
                         | (static_cast<uint32_t>(b));
}

/**
 * @brief Draw detection boxes and class labels on the hardware display overlay.
 *
 * This renders each detection as a rectangle with a text label above it using the
 * sp_display_* overlay APIs. Colors are chosen per-class from the provided palette.
 *
 * @param[in] display_obj  Opaque handle returned by display init/start functions.
 * @param[in] detections   List of detections with xyxy boxes, score, and class_id.
 * @param[in] class_names  List of class names; indexed by detection.class_id.
 * @param[in] colors       BGR color palette; cycled by class_id.
 * @param[in] chn          Display channel/layer index to draw on.
 */
void draw_detections_on_disp(void* display_obj,
                             const std::vector<Detection>& detections,
                             const std::vector<std::string>& class_names,
                             const std::vector<cv::Scalar>& colors,
                             int chn)
{
    // Clear canvas
    sp_display_draw_rect(display_obj, 0, 0, 0, 0, chn, 1, 1, 3);
    sp_display_draw_string(display_obj, 0, 0,  const_cast<char*>(""), chn, 1, 1, 16);

    for (const auto& det : detections) {
        // Clamp/convert bbox coordinates to int (display API expects ints)
        const int x1 = static_cast<int>(det.bbox[0]);   // left
        const int y1 = static_cast<int>(det.bbox[1]);   // top
        const int x2 = static_cast<int>(det.bbox[2]);   // right
        const int y2 = static_cast<int>(det.bbox[3]);   // bottom
        const int class_id = det.class_id;

        // Compose label text, e.g. "person 0.94"
        std::ostringstream oss;
        oss << class_names[class_id] << ' ' << std::fixed << std::setprecision(2) << det.score;
        const std::string label = oss.str();

        // Convert BGR (cv::Scalar) to ARGB8888 expected by display (with full alpha)
        const cv::Scalar bgr = colors[class_id % colors.size()]; // color per class
        const uint32_t argb8888_color = rgb_to_argb8888(
            static_cast<uint8_t>(bgr[2]),  // R
            static_cast<uint8_t>(bgr[1]),  // G
            static_cast<uint8_t>(bgr[0])   // B
        );

        // Draw rectangle; flush=0 (accumulate on overlay), line_width=3
        sp_display_draw_rect(display_obj, x1, y1, x2, y2, chn, 0, argb8888_color, 3);

        // Draw text slightly above the top-left of the box
        const int y_text = std::max(y1 - 20, 0);
        sp_display_draw_string(display_obj, x1, y_text,
                               const_cast<char*>(label.c_str()),
                               chn, 0, argb8888_color, 16);
    }
}

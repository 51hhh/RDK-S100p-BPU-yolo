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
#include <nlohmann/json.hpp>
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "common_utils.hpp"

/**
 * @brief Pack 8-bit RGB into ARGB8888 (alpha set to 0xFF).
 * @param[in] r  Red channel   [0..255].
 * @param[in] g  Green channel [0..255].
 * @param[in] b  Blue channel  [0..255].
 * @return uint32_t Packed color as 0xAARRGGBB with AA=0xFF.
 */
uint32_t rgb_to_argb8888(uint8_t r, uint8_t g, uint8_t b);

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
                              int chn = 2);

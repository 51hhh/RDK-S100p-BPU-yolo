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

#include "common_utils.hpp"
#include "preprocess_utils.hpp"
#include "postprocess_utils.hpp"

/**
 * @class UltralyticsYOLO
 * @brief A unified wrapper class for YOLOv8/YOLOv11 object detection models
 *        using the D-Robotics DNN API on RDK BPU.
 *
 * This class provides:
 * - Model loading and initialization
 * - Input preprocessing (BGR -> NV12 with letterbox)
 * - Inference execution on BPU
 * - Postprocessing with DFL decoding, confidence filtering and NMS
 *
 * Supports both YOLOv8 and YOLOv11 models with NV12 input format.
 */
class UltralyticsYOLO
{
    private:
        int model_count_;                             // Number of models loaded in the packed handle
        hbDNNPackedHandle_t packed_dnn_handle_;       // Packed DNN handle for managing loaded models
        hbDNNHandle_t dnn_handle_;                    // Handle to the YOLO model
        int32_t input_count_;                         // Number of input tensors for the model
        int32_t output_count_;                        // Number of output tensors for the model
        std::vector<hbDNNTensor> input_tensors_;      // Input tensor storage
        std::vector<hbDNNTensor> output_tensors_;     // Output tensor storage
        int input_h_;                                 // Model expected input height in pixels
        int input_w_;                                 // Model expected input width in pixels

    public:
        /**
         * @brief Construct a new UltralyticsYOLO object and load the model from a file.
         * @param model_path [in] Path to the YOLO model file (.hbm format).
         */
        UltralyticsYOLO(std::string model_path);

        /**
         * @brief Destroy the UltralyticsYOLO object and release allocated resources.
         */
        ~UltralyticsYOLO();

        /**
         * @brief Preprocess an input BGR image for YOLO inference.
         * @param bgr_mat [in] Input image in BGR format (OpenCV Mat).
         *
         * Performs:
         * - Letterbox resize to model input dimensions
         * - Format conversion from BGR to NV12
         */
        void pre_process(cv::Mat& bgr_mat);

        /**
         * @brief Run inference on preprocessed input tensors.
         *
         * Executes the YOLO model on BPU using the prepared inputs and
         * retrieves the raw output tensors.
         */
        void infer();

        /**
         * @brief Postprocess raw model outputs into detection results.
         * @param score_thres [in] Confidence score threshold for filtering detections.
         * @param nms_thres [in] IoU threshold for Non-Maximum Suppression (NMS).
         * @param img_w [in] Original image width (for rescaling boxes).
         * @param img_h [in] Original image height (for rescaling boxes).
         * @return std::vector<Detection> [out] Final list of detections after NMS.
         */
        std::vector<Detection> post_process(float score_thres, float nms_thres, int img_w, int img_h);

        /**
         * @brief Get the model input width.
         * @return int Input width in pixels.
         */
        int get_input_width() const { return input_w_; }

        /**
         * @brief Get the model input height.
         * @return int Input height in pixels.
         */
        int get_input_height() const { return input_h_; }
};

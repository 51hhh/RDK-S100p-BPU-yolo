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

#include "ultralytics_yolo.hpp"

// DFL 回归的 bin 数量
constexpr int REG = 16;

/**
 * @brief Construct a new UltralyticsYOLO object and initialize DNN resources.
 *
 * Loads the model from disk, obtains model handle, queries I/O tensor counts
 * and properties, and allocates memory for all tensors.
 *
 * @param model_path [in] Path to the YOLO *.hbm model file.
 */
UltralyticsYOLO::UltralyticsYOLO(std::string model_path)
{
    auto modelFileName = model_path.c_str();

    const char **model_name_list = nullptr;

    // Initialize from model file
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &modelFileName, 1),
                        "hbDNNInitializeFromFiles failed");
    // Model names
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");
    // Model handle
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");
    // I/O counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle_),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle_),
                        "hbDNNGetOutputCount failed");

    // Prepare tensor descriptors
    input_tensors_.resize(input_count_);
    output_tensors_.resize(output_count_);

    // Input tensor properties
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    // Output tensor properties
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache expected input dimensions
    input_h_ = input_tensors_[0].properties.validShape.dimensionSize[1];
    input_w_ = input_tensors_[0].properties.validShape.dimensionSize[2];

    // Allocate memory for all tensors
    prepare_input_tensor(input_tensors_);
    prepare_output_tensor(output_tensors_);
    
    std::cout << "Model loaded successfully: " << model_path << std::endl;
    std::cout << "  Input: " << input_count_ << " tensors, " << input_w_ << "x" << input_h_ << std::endl;
    std::cout << "  Output: " << output_count_ << " tensors" << std::endl;
    
    // 打印输出张量详情用于调试
    for (int i = 0; i < output_count_; i++) {
        auto& prop = output_tensors_[i].properties;
        std::cout << "    output[" << i << "] shape: ("
                  << prop.validShape.dimensionSize[0] << ", "
                  << prop.validShape.dimensionSize[1] << ", "
                  << prop.validShape.dimensionSize[2] << ", "
                  << prop.validShape.dimensionSize[3] << "), "
                  << "quantiType: " << prop.quantiType << std::endl;
    }
}

/**
 * @brief Destructor: release tensor memory and model resources.
 */
UltralyticsYOLO::~UltralyticsYOLO()
{
    // Release input memory
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors_[i].sysMem));
    }
    // Release output memory
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors_[i].sysMem));
    }
    // Release packed model handle
    hbDNNRelease(packed_dnn_handle_);
}

/**
 * @brief Preprocess BGR image to match YOLO input format.
 *
 * Letterbox resize to (input_w_, input_h_) and convert image to
 * NV12 tensor layout as expected by the runtime.
 *
 * @param bgr_mat [in] Input image in OpenCV BGR format.
 */
void UltralyticsYOLO::pre_process(cv::Mat& bgr_mat)
{
    // Letterbox resize to model input size
    cv::Mat resized_mat;
    resized_mat.create(input_h_, input_w_, bgr_mat.type());
    letterbox_resize(bgr_mat, resized_mat);

    // BGR -> NV12 input tensor
    bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_);
}

/**
 * @brief Execute inference on current input and populate outputs.
 *
 * Creates task, submits to UCP scheduler, waits for completion,
 * invalidates output tensor CPU caches, and releases task handle.
 */
void UltralyticsYOLO::infer()
{
    hbUCPTaskHandle_t task_handle{nullptr};

    // Create inference task
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors_.data(), input_tensors_.data(), dnn_handle_),
                        "hbDNNInferV2 failed");

    // Submit to BPU scheduler
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                        "hbUCPSubmitTask failed");

    // Wait until completion (0 => blocking)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    // Ensure CPU sees latest outputs
    for (int i = 0; i < output_count_; i++) {
        hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // Release handle
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
}

/**
 * @brief Decode and filter predictions from all heads, run NMS, and rescale boxes.
 *
 * 基于 example.md 中 YOLOv8 的后处理方式实现
 * 
 * Steps:
 * 1) Convert probability threshold to logit threshold.
 * 2) For each head (8/16/32), filter classes and decode DFL boxes.
 * 3) Concatenate detections from all heads.
 * 4) Apply NMS on concatenated detections.
 * 5) Map boxes back to original image coordinates (inverse letterbox).
 *
 * @param score_thres [in]  Confidence threshold in probability domain.
 * @param nms_thres   [in]  IoU threshold for NMS.
 * @param img_w       [in]  Original image width in pixels.
 * @param img_h       [in]  Original image height in pixels.
 * @return std::vector<Detection> [out] Final detections after NMS and rescaling.
 */
std::vector<Detection> UltralyticsYOLO::post_process(float score_thres, float nms_thres, int img_w, int img_h)
{
    // 检查输出张量数量
    if (output_count_ != 6) {
        std::cerr << "[ERROR] Expected 6 output tensors, but got " << output_count_ << std::endl;
        return {};
    }
    
    // 计算置信度阈值的原始值（利用 Sigmoid 函数的单调性）
    float conf_thres_raw = -std::log(1.0f / score_thres - 1.0f);
    
    std::vector<Detection> all_detections;
    
    // 三个尺度的配置: stride 和 grid_size
    const int strides[3] = {8, 16, 32};
    const int grid_sizes[3] = {input_w_ / 8, input_w_ / 16, input_w_ / 32};  // 80, 40, 20 for 640 input
    
    // 处理 3 个尺度的输出
    for (int scale = 0; scale < 3; scale++) {
        int cls_idx = scale * 2;       // 0, 2, 4 - 分类输出
        int bbox_idx = scale * 2 + 1;  // 1, 3, 5 - 边框输出
        int stride = strides[scale];
        int grid_size = grid_sizes[scale];
        
        // 获取输出张量
        hbDNNTensor& cls_tensor = output_tensors_[cls_idx];
        hbDNNTensor& bbox_tensor = output_tensors_[bbox_idx];
        
        // 检查数据指针
        if (cls_tensor.sysMem.virAddr == nullptr || bbox_tensor.sysMem.virAddr == nullptr) {
            std::cerr << "[ERROR] Output tensor " << scale << " has null pointer" << std::endl;
            continue;
        }
        
        // 获取类别数 (从 cls_tensor 的最后一维)
        int num_classes = cls_tensor.properties.validShape.dimensionSize[3];
        
        // 获取输出数据指针 - cls 是 float
        auto* cls_data = reinterpret_cast<float*>(cls_tensor.sysMem.virAddr);
        
        // 判断 bbox 是否需要反量化 (quantiType: 0=NONE, 1=SCALE)
        bool bbox_need_dequant = (bbox_tensor.properties.quantiType == 1);
        float* bbox_scale = bbox_tensor.properties.scale.scaleData;
        
        int total_anchors = grid_size * grid_size;
        
        // 遍历所有 anchor 位置
        for (int anchor_idx = 0; anchor_idx < total_anchors; anchor_idx++) {
            float* cur_cls = cls_data + anchor_idx * num_classes;
            
            // 找到最大分数和对应类别
            int max_cls_id = 0;
            float max_cls_val = cur_cls[0];
            for (int c = 1; c < num_classes; c++) {
                if (cur_cls[c] > max_cls_val) {
                    max_cls_val = cur_cls[c];
                    max_cls_id = c;
                }
            }
            
            // 检查是否超过阈值（raw 值比较）
            if (max_cls_val < conf_thres_raw) {
                continue;
            }
            
            // 计算 Sigmoid 分数
            float score = 1.0f / (1.0f + std::exp(-max_cls_val));
            
            // DFL 计算 - 对每条边进行处理
            float ltrb[4];
            
            if (bbox_need_dequant) {
                // bbox 是 int32 量化的，需要反量化
                auto* bbox_data_int = reinterpret_cast<int32_t*>(bbox_tensor.sysMem.virAddr);
                int32_t* cur_bbox = bbox_data_int + anchor_idx * (REG * 4);
                
                for (int i = 0; i < 4; i++) {
                    float dfl_values[REG];
                    float softmax_values[REG];
                    
                    // 反量化 DFL 值
                    for (int j = 0; j < REG; j++) {
                        int scale_idx = i * REG + j;
                        dfl_values[j] = static_cast<float>(cur_bbox[scale_idx]) * bbox_scale[scale_idx];
                    }
                    
                    // Softmax
                    softmax(dfl_values, softmax_values, REG);
                    
                    // 计算期望值（DFL 到距离的转换）
                    ltrb[i] = 0.0f;
                    for (int j = 0; j < REG; j++) {
                        ltrb[i] += softmax_values[j] * j;
                    }
                }
            } else {
                // bbox 已经是 float，直接读取
                auto* bbox_data_float = reinterpret_cast<float*>(bbox_tensor.sysMem.virAddr);
                float* cur_bbox = bbox_data_float + anchor_idx * (REG * 4);
                
                for (int i = 0; i < 4; i++) {
                    float dfl_values[REG];
                    float softmax_values[REG];
                    
                    // 直接读取 float 值
                    for (int j = 0; j < REG; j++) {
                        int idx = i * REG + j;
                        dfl_values[j] = cur_bbox[idx];
                    }
                    
                    // Softmax
                    softmax(dfl_values, softmax_values, REG);
                    
                    // 计算期望值（DFL 到距离的转换）
                    ltrb[i] = 0.0f;
                    for (int j = 0; j < REG; j++) {
                        ltrb[i] += softmax_values[j] * j;
                    }
                }
            }
            
            // 计算 anchor 坐标
            int grid_y = anchor_idx / grid_size;
            int grid_x = anchor_idx % grid_size;
            float anchor_x = grid_x + 0.5f;
            float anchor_y = grid_y + 0.5f;
            
            // ltrb 转 xyxy 坐标
            float x1 = (anchor_x - ltrb[0]) * stride;
            float y1 = (anchor_y - ltrb[1]) * stride;
            float x2 = (anchor_x + ltrb[2]) * stride;
            float y2 = (anchor_y + ltrb[3]) * stride;
            
            // 检查边界框合法性
            if (x2 > x1 && y2 > y1) {
                Detection det;
                det.bbox[0] = x1;
                det.bbox[1] = y1;
                det.bbox[2] = x2;
                det.bbox[3] = y2;
                det.score = score;
                det.class_id = max_cls_id;
                all_detections.push_back(det);
            }
        }
    }
    
    // NMS 处理
    auto results = nms_bboxes(all_detections, nms_thres);
    
    // 将坐标映射回原图（撤销 letterbox）
    scale_letterbox_bboxes_back(results, img_w, img_h, input_w_, input_h_);
    
    return results;
}

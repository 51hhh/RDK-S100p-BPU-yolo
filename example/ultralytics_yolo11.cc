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

#include "ultralytics_yolo11.hpp"
#include <omp.h>

/**
 * @brief 每个检测头的步长 (从高分辨率到低分辨率)。
 */
std::vector<int> strides       = {8, 16, 32};

/**
 * @brief 每个检测头的特征图网格大小。
 * 典型的头是 80x80 / 40x40 / 20x20。
 */
std::vector<int> anchor_sizes  = {80, 40, 20};

/**
 * @brief DFL 的固定 bin 偏移量 (0..15)。
 */
std::vector<int> weights_static = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

/**
 * @brief 过滤类别 logits 并解码一个头的 DFL 框分布。
 *
 * 工作流程:
 * 1) 对于每个 (n,h,w)，从 cls_tensor 中取 C 个类别的 argmax。
 * 2) 将最大 logit 与 conf_thres_raw (logit 域) 进行比较；如果较低则跳过。
 * 3) 对于 4 个边中的每一个，从 bbox_tensor 读取 16-bin logits，进行 softmax，
 *    并使用 weights_static 计算期望值以获得距离。
 * 4) 将 (anchor_x,anchor_y,l,t,r,b) 转换为输入尺度下的 (x1,y1,x2,y2)，乘以 stride。
 *
 * 使用 OpenMP 在 HxW 上并行化。
 *
 * @param cls_tensor      [in]  分类 logits，形状 (N,H,W,C)，类型 float。
 * @param bbox_tensor     [in]  框分布，形状 (N,H,W,64)，类型 (通常) int32。
 * @param conf_thres_raw  [in]  Logit 空间中的置信度阈值 (pre-sigmoid)。
 * @param grid_size       [in]  此头的特征图大小 (例如 80/40/20)。
 * @param stride          [in]  此头的输入步长 (例如 8/16/32)。
 * @param weights_static  [in]  DFL bin 偏移量 (0..15)。
 * @param detections      [out] 解码后的检测结果追加到此处。
 */
void filter_and_decode_detections(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    const std::vector<int>& weights_static,
    std::vector<Detection>& detections)
{
    detections.clear();

    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int N = shape.dimensionSize[0];
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3];

    const int64_t* stride_cls  = cls_tensor.properties.stride;
    const int64_t* stride_bbox = bbox_tensor.properties.stride;

    const uint8_t* data_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* data_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);

    // 线程局部缓冲区以减少争用
    std::vector<std::vector<Detection>> thread_dets(omp_get_max_threads());

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();
            auto& dets_local = thread_dets[tid];

            for (int n = 0; n < N; ++n) {
                size_t base_cls_offset  = n * stride_cls[0]  + h * stride_cls[1]  + w * stride_cls[2];
                size_t base_bbox_offset = n * stride_bbox[0] + h * stride_bbox[1] + w * stride_bbox[2];

                // 对类别进行 Argmax (float logits)
                float max_val = -1e30f;
                int max_id = 0;
                for (int c = 0; c < C; ++c) {
                    const float* ptr_cls = reinterpret_cast<const float*>(
                        data_cls + base_cls_offset + c * stride_cls[3]);
                    float val = *ptr_cls;
                    if (val > max_val) {
                        max_val = val;
                        max_id = c;
                    }
                }
                if (max_val < conf_thres_raw) continue;  // logit 域中的阈值

                Detection det{};
                det.score = sigmoid(max_val);            // 转换为概率
                det.class_id = max_id;

                // 通过 DFL 解码 bbox (4 个边 × 16 bins)
                float anchor_x = 0.5f + w;
                float anchor_y = 0.5f + h;
                float ltrb[4] = {0, 0, 0, 0};

                for (int side = 0; side < 4; ++side) {
                    float bins[16];
                    for (int bin = 0; bin < 16; ++bin) {
                        int channel = side * 16 + bin;
                        const int32_t* ptr_bbox = reinterpret_cast<const int32_t*>(
                            data_bbox + base_bbox_offset + channel * stride_bbox[3]);
                        bins[bin] = dequant_value(*ptr_bbox, channel, bbox_tensor.properties);
                    }

                    // 对 bins 进行 Softmax (使用 max trick 保持稳定性)
                    float max_bin = bins[0];
                    for (int i = 1; i < 16; ++i) if (bins[i] > max_bin) max_bin = bins[i];
                    float sum = 0.0f;
                    float probs[16];
                    for (int i = 0; i < 16; ++i) {
                        probs[i] = std::exp(bins[i] - max_bin);
                        sum += probs[i];
                    }
                    for (int i = 0; i < 16; ++i) {
                        ltrb[side] += probs[i] * weights_static[i] / sum; // 期望值
                    }
                }

                // (anchor, ltrb) -> 输入尺度下的 (x1,y1,x2,y2)
                det.bbox[0] = (anchor_x - ltrb[0]) * stride;
                det.bbox[1] = (anchor_y - ltrb[1]) * stride;
                det.bbox[2] = (anchor_x + ltrb[2]) * stride;
                det.bbox[3] = (anchor_y + ltrb[3]) * stride;

                dets_local.push_back(det);
            }
        }
    }

    // 合并线程局部向量
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
    }
}

/**
 * @brief 构造一个新的 YOLO11 对象并初始化 DNN 资源。
 *
 * 从磁盘加载模型，获取模型句柄，查询输入/输出张量计数和属性，
 * 并为所有张量分配内存。
 *
 * @param model_path [in] YOLOv11 *.hbm 模型文件的路径。
 */
YOLO11::YOLO11(std::string model_path)
{
    auto modelFileName = model_path.c_str();

    const char **model_name_list = nullptr;

    // 从模型文件初始化
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &modelFileName, 1),
                        "hbDNNInitializeFromFiles failed");
    // 模型名称
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");
    // 模型句柄
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");
    // I/O 计数
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle_),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle_),
                        "hbDNNGetOutputCount failed");

    // 准备张量描述符
    input_tensors_.resize(input_count_);
    output_tensors_.resize(output_count_);

    // 输入张量属性
    for (int i = 0; i < input_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    // 输出张量属性
    for (int i = 0; i < output_count_; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // 缓存预期的输入尺寸
    input_h_ = input_tensors_[0].properties.validShape.dimensionSize[1];
    input_w_ = input_tensors_[0].properties.validShape.dimensionSize[2];

    // 为所有张量分配内存
    prepare_input_tensor(input_tensors_);
    prepare_output_tensor(output_tensors_);
}

/**
 * @brief 析构函数: 释放张量内存并释放模型资源。
 */
YOLO11::~YOLO11()
{
    // 释放输入内存
    for (int i = 0; i < input_count_; i++) {
        hbUCPFree(&(input_tensors_[i].sysMem));
    }
    // 释放输出内存
    for (int i = 0; i < output_count_; i++) {
        hbUCPFree(&(output_tensors_[i].sysMem));
    }
    // 释放打包的模型句柄
    hbDNNRelease(packed_dnn_handle_);
}

/**
 * @brief 预处理 BGR 图像以匹配 YOLOv11 输入格式。
 *
 * Letterbox 缩放到 (input_w_, input_h_) 并将图像转换为
 * 运行时预期的 NV12 张量布局。
 *
 * @param bgr_mat [in] OpenCV BGR 格式的输入图像。
 */
void YOLO11::pre_process(cv::Mat& bgr_mat)
{
    // Letterbox 缩放到模型输入尺寸
    cv::Mat resized_mat;
    resized_mat.create(input_h_, input_w_, bgr_mat.type());
    letterbox_resize(bgr_mat, resized_mat);

    // BGR -> NV12 输入张量
    bgr_to_nv12_tensor(resized_mat, input_tensors_, input_h_, input_w_);
}

/**
 * @brief 对当前输入执行推理并填充输出。
 *
 * 创建任务，提交给 UCP 调度程序，等待完成，
 * 使输出张量的 CPU 缓存失效，并释放任务句柄。
 */
void YOLO11::infer()
{
    hbUCPTaskHandle_t task_handle{nullptr};

    // 创建推理任务
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors_.data(), input_tensors_.data(), dnn_handle_),
                        "hbDNNInferV2 failed");

    // 提交给 BPU 调度程序
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                        "hbUCPSubmitTask failed");

    // 等待直到完成 (0 => 阻塞)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        "hbUCPWaitTaskDone failed");

    // 确保 CPU 看到最新的输出
    for (int i = 0; i < output_count_; i++) {
        hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // 释放句柄
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
}

/**
 * @brief 解码和过滤所有头的预测，运行 NMS，并重新缩放框。
 *
 * 步骤:
 * 1) 将概率阈值转换为 logit 阈值。
 * 2) 对于每个头 (8/16/32)，过滤类别并解码 DFL 框。
 * 3) 连接所有头的检测结果。
 * 4) 对连接的检测结果应用 NMS。
 * 5) 将框映射回原始图像坐标 (逆 Letterbox)。
 *
 * @param score_thres [in]  概率域中的置信度阈值。
 * @param nms_thres   [in]  NMS 的 IoU 阈值。
 * @param img_w       [in]  原始图像宽度，以像素为单位。
 * @param img_h       [in]  原始图像高度，以像素为单位。
 * @return std::vector<Detection> [out] NMS 和重新缩放后的最终检测结果。
 */
std::vector<Detection> YOLO11::post_process(float score_thres, float nms_thres, int img_w, int img_h)
{
    // 概率 -> logit 以便与原始 logits 进行比较
    float conf_thres_raw = -std::log(1.0f / score_thres - 1.0f);

    std::vector<Detection> all_detections;

    // 每个头贡献 [cls_head, bbox_head]
    for (size_t s = 0; s < strides.size(); ++s) {
        const hbDNNTensor& cls_tensor  = output_tensors_[2*s + 0];
        const hbDNNTensor& bbox_tensor = output_tensors_[2*s + 1];

        std::vector<Detection> dets;
        filter_and_decode_detections(cls_tensor, bbox_tensor, conf_thres_raw,
                                     anchor_sizes[s], strides[s], weights_static, dets);

        // 合并头结果
        all_detections.insert(all_detections.end(),
                              std::make_move_iterator(dets.begin()),
                              std::make_move_iterator(dets.end()));
    }

    // 对所有头进行 NMS
    auto results = nms_bboxes(all_detections, nms_thres);

    // 重新缩放到原始图像 (撤销 Letterbox)
    scale_letterbox_bboxes_back(results, img_w, img_h, input_w_, input_h_);

    return results;
}

/*
 * Copyright (c) 2025, D-Robotics.
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

/**
 * @file test_hbm_info.cc
 * @brief UCP HBM模型测试工具 - 用于测试hbm模型的输入输出属性及各种格式信息
 * 
 * 功能:
 * 1. 加载hbm模型
 * 2. 获取并打印模型基本信息（名称、数量）
 * 3. 获取并打印输入张量详细属性（形状、类型、量化信息、stride等）
 * 4. 获取并打印输出张量详细属性
 * 5. 测试内存分配
 * 6. 测试简单推理流程
 */

// C/C++ Standard Libraries
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <cmath>

// RDK S100 UCP API
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

// ============================================================================
// 辅助宏定义
// ============================================================================

/**
 * @brief RDK 错误检查宏
 */
#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
            return ret_code;                                                     \
        }                                                                        \
    } while (0);

/**
 * @brief 打印分隔线
 */
#define PRINT_SEPARATOR(title)                                                   \
    std::cout << "\n" << std::string(60, '=') << std::endl;                      \
    std::cout << "  " << title << std::endl;                                     \
    std::cout << std::string(60, '=') << std::endl;

/**
 * @brief 打印小分隔线
 */
#define PRINT_SUBSEPARATOR(title)                                                \
    std::cout << "\n" << std::string(40, '-') << std::endl;                      \
    std::cout << "  " << title << std::endl;                                     \
    std::cout << std::string(40, '-') << std::endl;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 获取张量类型名称
 * @param tensorType 张量类型ID
 * @return 类型名称字符串
 */
const char* get_tensor_type_name(int32_t tensorType) {
    switch(tensorType) {
        case 0: return "HB_DNN_TENSOR_TYPE_S8 (有符号8位整数)";
        case 1: return "HB_DNN_TENSOR_TYPE_U8 (无符号8位整数)";
        case 2: return "HB_DNN_TENSOR_TYPE_F32 (32位浮点)";
        case 3: return "HB_DNN_IMG_TYPE_NV12 (NV12图像格式)";
        case 4: return "HB_DNN_IMG_TYPE_NV12_SEPARATE (NV12分离Y/UV)";
        case 5: return "HB_DNN_TENSOR_TYPE_S16 (有符号16位整数)";
        case 6: return "HB_DNN_TENSOR_TYPE_U16 (无符号16位整数)";
        case 7: return "HB_DNN_TENSOR_TYPE_F16 (16位浮点)";
        case 8: return "HB_DNN_TENSOR_TYPE_S32 (有符号32位整数)";
        case 9: return "HB_DNN_TENSOR_TYPE_U32 (无符号32位整数)";
        case 10: return "HB_DNN_TENSOR_TYPE_F64 (64位浮点)";
        case 11: return "HB_DNN_TENSOR_TYPE_S64 (有符号64位整数)";
        case 12: return "HB_DNN_TENSOR_TYPE_U64 (无符号64位整数)";
        default: return "未知类型";
    }
}

/**
 * @brief 获取量化类型名称
 * @param quantiType 量化类型
 * @return 类型名称字符串
 */
const char* get_quanti_type_name(int32_t quantiType) {
    switch(quantiType) {
        case 0: return "NONE (无量化)";
        case 1: return "SCALE (缩放量化)";
        default: return "未知量化类型";
    }
}

/**
 * @brief 获取描述类型名称
 * @param descType 描述类型
 * @return 类型名称字符串
 */
const char* get_desc_type_name(int32_t descType) {
    switch(descType) {
        case 0: return "HB_DNN_DESC_TYPE_UNKNOWN";
        case 1: return "HB_DNN_DESC_TYPE_JSON";
        default: return "未知描述类型";
    }
}

/**
 * @brief 打印张量形状信息
 * @param shape 张量形状结构体
 */
void print_tensor_shape(const hbDNNTensorShape& shape) {
    std::cout << "    维度数量: " << shape.numDimensions << std::endl;
    std::cout << "    形状: (";
    for (int i = 0; i < shape.numDimensions; i++) {
        std::cout << shape.dimensionSize[i];
        if (i < shape.numDimensions - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}

/**
 * @brief 打印量化缩放信息
 * @param scale 量化缩放结构体
 */
void print_quanti_scale(const hbDNNQuantiScale& scale) {
    std::cout << "    缩放数据长度: " << scale.scaleLen << std::endl;
    if (scale.scaleLen > 0 && scale.scaleData != nullptr) {
        std::cout << "    缩放数据 (前10个): ";
        for (int i = 0; i < std::min(10, scale.scaleLen); i++) {
            std::cout << std::fixed << std::setprecision(6) << scale.scaleData[i];
            if (i < std::min(10, scale.scaleLen) - 1) std::cout << ", ";
        }
        if (scale.scaleLen > 10) std::cout << " ...";
        std::cout << std::endl;
    }
    std::cout << "    零点偏移长度: " << scale.zeroPointLen << std::endl;
    if (scale.zeroPointLen > 0 && scale.zeroPointData != nullptr) {
        std::cout << "    零点偏移数据 (前10个): ";
        for (int i = 0; i < std::min(10, scale.zeroPointLen); i++) {
            std::cout << scale.zeroPointData[i];
            if (i < std::min(10, scale.zeroPointLen) - 1) std::cout << ", ";
        }
        if (scale.zeroPointLen > 10) std::cout << " ...";
        std::cout << std::endl;
    }
}

/**
 * @brief 打印张量属性详细信息
 * @param properties 张量属性结构体
 * @param index 张量索引
 * @param isInput 是否是输入张量
 */
void print_tensor_properties(const hbDNNTensorProperties& properties, int index, bool isInput) {
    const char* type = isInput ? "输入" : "输出";
    
    std::cout << "\n  [" << type << " Tensor " << index << "]" << std::endl;
    std::cout << "  --------------------------------" << std::endl;
    
    // 形状信息
    std::cout << "  → 有效形状 (validShape):" << std::endl;
    print_tensor_shape(properties.validShape);
    
    // 类型信息
    std::cout << "  → 张量类型: " << properties.tensorType 
              << " - " << get_tensor_type_name(properties.tensorType) << std::endl;
    
    // 量化信息
    std::cout << "  → 量化类型: " << properties.quantiType 
              << " - " << get_quanti_type_name(properties.quantiType) << std::endl;
    std::cout << "  → 量化轴: " << properties.quantizeAxis << std::endl;
    
    // 量化缩放信息
    if (properties.quantiType != 0) {
        std::cout << "  → 量化缩放信息:" << std::endl;
        print_quanti_scale(properties.scale);
    }
    
    // 内存信息
    std::cout << "  → 对齐后字节大小: " << properties.alignedByteSize << " bytes" << std::endl;
    
    // stride信息
    std::cout << "  → Stride信息: (";
    for (int i = 0; i < properties.validShape.numDimensions; i++) {
        std::cout << properties.stride[i];
        if (i < properties.validShape.numDimensions - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
    
    // 计算张量元素总数
    int64_t total_elements = 1;
    for (int i = 0; i < properties.validShape.numDimensions; i++) {
        total_elements *= properties.validShape.dimensionSize[i];
    }
    std::cout << "  → 元素总数: " << total_elements << std::endl;
}

/**
 * @brief 分析NV12输入格式的详细信息
 * @param properties 输入张量属性
 * @param input_count 输入数量
 */
void analyze_nv12_input(const std::vector<hbDNNTensorProperties>& input_props, int input_count) {
    PRINT_SUBSEPARATOR("NV12输入格式分析");
    
    if (input_count < 1) {
        std::cout << "  ⚠ 输入数量不足，无法进行NV12分析" << std::endl;
        return;
    }
    
    // 检查是否是NV12格式
    if (input_props[0].tensorType == 3 || input_props[0].tensorType == 4) {
        std::cout << "  ✓ 检测到NV12输入格式" << std::endl;
        
        if (input_count == 1) {
            std::cout << "  → 模式: 紧凑NV12 (Y和UV在同一个tensor中)" << std::endl;
            int H = input_props[0].validShape.dimensionSize[1];
            int W = input_props[0].validShape.dimensionSize[2];
            std::cout << "  → 输入图像尺寸: " << W << "x" << H << std::endl;
            std::cout << "  → Y平面大小: " << H << "x" << W << " = " << H*W << " bytes" << std::endl;
            std::cout << "  → UV平面大小: " << (H/2) << "x" << W << " = " << (H/2)*W << " bytes" << std::endl;
            std::cout << "  → 总NV12大小: " << H*W*3/2 << " bytes" << std::endl;
        } else if (input_count >= 2) {
            std::cout << "  → 模式: 分离NV12 (Y和UV在不同tensor中)" << std::endl;
            
            // Y平面信息
            std::cout << "  → Y平面 (Tensor 0):" << std::endl;
            int Y_H = input_props[0].validShape.dimensionSize[1];
            int Y_W = input_props[0].validShape.dimensionSize[2];
            std::cout << "      尺寸: " << Y_H << "x" << Y_W << std::endl;
            std::cout << "      大小: " << Y_H * Y_W << " bytes" << std::endl;
            
            // UV平面信息
            if (input_count >= 2) {
                std::cout << "  → UV平面 (Tensor 1):" << std::endl;
                int UV_H = input_props[1].validShape.dimensionSize[1];
                int UV_W = input_props[1].validShape.dimensionSize[2];
                int UV_C = input_props[1].validShape.dimensionSize[3];
                std::cout << "      尺寸: " << UV_H << "x" << UV_W << "x" << UV_C << std::endl;
                std::cout << "      大小: " << UV_H * UV_W * UV_C << " bytes" << std::endl;
            }
        }
    } else {
        std::cout << "  ⚠ 当前模型不是NV12输入格式" << std::endl;
        std::cout << "  → 实际输入类型: " << get_tensor_type_name(input_props[0].tensorType) << std::endl;
    }
}

/**
 * @brief 分析YOLO输出格式
 * @param output_props 输出张量属性列表
 * @param output_count 输出数量
 */
void analyze_yolo_output(const std::vector<hbDNNTensorProperties>& output_props, int output_count) {
    PRINT_SUBSEPARATOR("YOLO输出格式分析");
    
    if (output_count == 6) {
        std::cout << "  ✓ 检测到6输出YOLO格式 (可能是YOLOv8/v11 检测模型)" << std::endl;
        std::cout << "  → 典型布局: [cls_80x80, bbox_80x80, cls_40x40, bbox_40x40, cls_20x20, bbox_20x20]" << std::endl;
        
        for (int i = 0; i < output_count; i++) {
            const auto& shape = output_props[i].validShape;
            std::cout << "  → 输出[" << i << "]: (";
            for (int j = 0; j < shape.numDimensions; j++) {
                std::cout << shape.dimensionSize[j];
                if (j < shape.numDimensions - 1) std::cout << ", ";
            }
            std::cout << ")";
            
            // 推断输出类型
            if (shape.numDimensions == 4) {
                int C = shape.dimensionSize[3];
                int H = shape.dimensionSize[1];
                if (C == 64) {
                    std::cout << " → bbox回归 (DFL, 4x16)";
                } else if (C == 80 || C >= 1 && C < 64) {
                    std::cout << " → 类别分数 (" << C << " classes)";
                }
                std::cout << " @ " << H << "x" << H << " 特征图";
            }
            
            std::cout << " - " << get_tensor_type_name(output_props[i].tensorType);
            if (output_props[i].quantiType == 1) {
                std::cout << " (量化)";
            }
            std::cout << std::endl;
        }
    } else if (output_count == 3) {
        std::cout << "  ✓ 检测到3输出格式 (可能是YOLOv5/v7 或融合输出)" << std::endl;
    } else if (output_count == 1) {
        std::cout << "  ✓ 检测到单输出格式 (可能是端到端模型)" << std::endl;
    } else {
        std::cout << "  → 检测到 " << output_count << " 个输出" << std::endl;
    }
}

/**
 * @brief 测试内存分配
 * @param input_props 输入张量属性
 * @param output_props 输出张量属性
 */
void test_memory_allocation(
    const std::vector<hbDNNTensorProperties>& input_props,
    const std::vector<hbDNNTensorProperties>& output_props)
{
    PRINT_SUBSEPARATOR("内存分配测试");
    
    std::cout << "  → 测试输入内存分配..." << std::endl;
    for (size_t i = 0; i < input_props.size(); i++) {
        hbUCPSysMem mem;
        int64_t size = input_props[i].alignedByteSize;
        
        auto start = std::chrono::high_resolution_clock::now();
        int ret = hbUCPMallocCached(&mem, size, 0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if (ret == 0) {
            std::cout << "    输入[" << i << "]: 分配成功, 大小=" << size 
                      << " bytes, 物理地址=0x" << std::hex << mem.phyAddr << std::dec
                      << ", 耗时=" << duration << " us" << std::endl;
            hbUCPFree(&mem);
        } else {
            std::cout << "    输入[" << i << "]: 分配失败, 错误码=" << ret << std::endl;
        }
    }
    
    std::cout << "  → 测试输出内存分配..." << std::endl;
    for (size_t i = 0; i < output_props.size(); i++) {
        hbUCPSysMem mem;
        int64_t size = output_props[i].alignedByteSize;
        
        auto start = std::chrono::high_resolution_clock::now();
        int ret = hbUCPMallocCached(&mem, size, 0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if (ret == 0) {
            std::cout << "    输出[" << i << "]: 分配成功, 大小=" << size 
                      << " bytes, 物理地址=0x" << std::hex << mem.phyAddr << std::dec
                      << ", 耗时=" << duration << " us" << std::endl;
            hbUCPFree(&mem);
        } else {
            std::cout << "    输出[" << i << "]: 分配失败, 错误码=" << ret << std::endl;
        }
    }
}

/**
 * @brief 测试完整推理流程
 * @param dnn_handle 模型句柄
 * @param input_props 输入张量属性
 * @param output_props 输出张量属性
 */
int test_inference_flow(
    hbDNNHandle_t dnn_handle,
    const std::vector<hbDNNTensorProperties>& input_props,
    const std::vector<hbDNNTensorProperties>& output_props)
{
    PRINT_SUBSEPARATOR("推理流程测试");
    
    int input_count = input_props.size();
    int output_count = output_props.size();
    
    // 1. 分配输入张量
    std::cout << "  Step 1: 分配输入张量内存..." << std::endl;
    std::vector<hbDNNTensor> input_tensors(input_count);
    for (int i = 0; i < input_count; i++) {
        input_tensors[i].properties = input_props[i];
        int64_t size = input_props[i].alignedByteSize;
        int ret = hbUCPMallocCached(&input_tensors[i].sysMem, size, 0);
        if (ret != 0) {
            std::cout << "    ✗ 输入[" << i << "]内存分配失败" << std::endl;
            // 清理已分配的内存
            for (int j = 0; j < i; j++) {
                hbUCPFree(&input_tensors[j].sysMem);
            }
            return -1;
        }
        // 填充测试数据 (全0)
        memset(input_tensors[i].sysMem.virAddr, 0, size);
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        std::cout << "    ✓ 输入[" << i << "]分配完成, 大小=" << size << " bytes" << std::endl;
    }
    
    // 2. 分配输出张量
    std::cout << "  Step 2: 分配输出张量内存..." << std::endl;
    std::vector<hbDNNTensor> output_tensors(output_count);
    for (int i = 0; i < output_count; i++) {
        output_tensors[i].properties = output_props[i];
        int64_t size = output_props[i].alignedByteSize;
        int ret = hbUCPMallocCached(&output_tensors[i].sysMem, size, 0);
        if (ret != 0) {
            std::cout << "    ✗ 输出[" << i << "]内存分配失败" << std::endl;
            // 清理已分配的内存
            for (int j = 0; j < input_count; j++) {
                hbUCPFree(&input_tensors[j].sysMem);
            }
            for (int j = 0; j < i; j++) {
                hbUCPFree(&output_tensors[j].sysMem);
            }
            return -1;
        }
        std::cout << "    ✓ 输出[" << i << "]分配完成, 大小=" << size << " bytes" << std::endl;
    }
    
    // 3. 执行推理
    std::cout << "  Step 3: 执行推理..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    hbUCPTaskHandle_t task_handle = nullptr;
    int infer_ret = hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
    if (infer_ret != 0) {
        std::cout << "    ✗ hbDNNInferV2失败, 错误码=" << infer_ret << std::endl;
        goto cleanup;
    }
    
    if (task_handle == nullptr) {
        std::cout << "    ✗ task_handle为空" << std::endl;
        goto cleanup;
    }
    
    {
        // 设置调度参数
        hbUCPSchedParam ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
        
        // 提交任务
        int submit_ret = hbUCPSubmitTask(task_handle, &ctrl_param);
        if (submit_ret != 0) {
            std::cout << "    ✗ hbUCPSubmitTask失败, 错误码=" << submit_ret << std::endl;
            hbUCPReleaseTask(task_handle);
            goto cleanup;
        }
        
        // 等待任务完成
        int wait_ret = hbUCPWaitTaskDone(task_handle, 10000);
        if (wait_ret != 0) {
            std::cout << "    ✗ hbUCPWaitTaskDone失败, 错误码=" << wait_ret << std::endl;
            hbUCPReleaseTask(task_handle);
            goto cleanup;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "    ✓ 推理完成, 耗时=" << duration / 1000.0 << " ms" << std::endl;
        
        // 释放任务句柄
        hbUCPReleaseTask(task_handle);
    }
    
    // 4. 刷新并读取输出
    std::cout << "  Step 4: 读取输出数据..." << std::endl;
    for (int i = 0; i < output_count; i++) {
        hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        
        // 检查输出数据（打印前几个值）
        const auto& props = output_tensors[i].properties;
        if (props.tensorType == 2) { // float32
            float* data = (float*)output_tensors[i].sysMem.virAddr;
            std::cout << "    输出[" << i << "] (float32) 前5个值: ";
            for (int j = 0; j < std::min(5, (int)(props.alignedByteSize / sizeof(float))); j++) {
                std::cout << std::fixed << std::setprecision(4) << data[j] << " ";
            }
            std::cout << std::endl;
        } else if (props.tensorType == 8) { // int32
            int32_t* data = (int32_t*)output_tensors[i].sysMem.virAddr;
            std::cout << "    输出[" << i << "] (int32) 前5个值: ";
            for (int j = 0; j < std::min(5, (int)(props.alignedByteSize / sizeof(int32_t))); j++) {
                std::cout << data[j] << " ";
            }
            std::cout << std::endl;
        } else if (props.tensorType == 0 || props.tensorType == 1) { // int8/uint8
            uint8_t* data = (uint8_t*)output_tensors[i].sysMem.virAddr;
            std::cout << "    输出[" << i << "] (int8/uint8) 前5个值: ";
            for (int j = 0; j < std::min(5, (int)props.alignedByteSize); j++) {
                std::cout << (int)data[j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "  ✓ 推理流程测试完成" << std::endl;

cleanup:
    // 5. 清理内存
    std::cout << "  Step 5: 清理内存..." << std::endl;
    for (int i = 0; i < input_count; i++) {
        hbUCPFree(&input_tensors[i].sysMem);
    }
    for (int i = 0; i < output_count; i++) {
        hbUCPFree(&output_tensors[i].sysMem);
    }
    std::cout << "    ✓ 内存清理完成" << std::endl;
    
    return 0;
}

// ============================================================================
// 主函数
// ============================================================================

/**
 * @brief 打印使用帮助
 */
void print_usage(const char* program_name) {
    std::cout << "使用方法: " << program_name << " <model_path.hbm> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --test-infer    执行推理测试 (默认: 否)" << std::endl;
    std::cout << "  --test-memory   执行内存分配测试 (默认: 否)" << std::endl;
    std::cout << "  --all           执行所有测试" << std::endl;
    std::cout << "  --help, -h      显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name << " model/best.hbm" << std::endl;
    std::cout << "  " << program_name << " model/best.hbm --all" << std::endl;
    std::cout << "  " << program_name << " model/best.hbm --test-infer" << std::endl;
}

/**
 * @brief 主函数入口
 */
int main(int argc, char** argv) {
    
    // 解析命令行参数
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string model_path;
    bool test_infer = false;
    bool test_memory = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--test-infer") {
            test_infer = true;
        } else if (arg == "--test-memory") {
            test_memory = true;
        } else if (arg == "--all") {
            test_infer = true;
            test_memory = true;
        } else if (model_path.empty() && arg[0] != '-') {
            model_path = arg;
        }
    }
    
    if (model_path.empty()) {
        std::cout << "错误: 未指定模型路径" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    PRINT_SEPARATOR("UCP HBM模型测试工具");
    std::cout << "  模型路径: " << model_path << std::endl;
    std::cout << "  测试推理: " << (test_infer ? "是" : "否") << std::endl;
    std::cout << "  测试内存: " << (test_memory ? "是" : "否") << std::endl;
    
    // ========================================================================
    // Step 1: 加载模型
    // ========================================================================
    PRINT_SEPARATOR("Step 1: 加载模型");
    
    auto load_start = std::chrono::high_resolution_clock::now();
    
    hbDNNPackedHandle_t packed_dnn_handle;
    const char* model_file_name = model_path.c_str();
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    
    std::cout << "  ✓ 模型加载成功, 耗时: " << load_duration << " ms" << std::endl;
    
    // ========================================================================
    // Step 2: 获取模型名称列表
    // ========================================================================
    PRINT_SEPARATOR("Step 2: 获取模型名称列表");
    
    const char** model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");
    
    std::cout << "  模型数量: " << model_count << std::endl;
    for (int i = 0; i < model_count; i++) {
        std::cout << "  模型[" << i << "]: " << model_name_list[i] << std::endl;
    }
    
    // ========================================================================
    // Step 3: 获取模型句柄
    // ========================================================================
    PRINT_SEPARATOR("Step 3: 获取模型句柄");
    
    hbDNNHandle_t dnn_handle;
    const char* model_name = model_name_list[0];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");
    
    std::cout << "  ✓ 已获取模型句柄: " << model_name << std::endl;
    
    // ========================================================================
    // Step 4: 获取输入输出数量
    // ========================================================================
    PRINT_SEPARATOR("Step 4: 获取输入输出数量");
    
    int32_t input_count = 0, output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");
    
    std::cout << "  输入张量数量: " << input_count << std::endl;
    std::cout << "  输出张量数量: " << output_count << std::endl;
    
    // ========================================================================
    // Step 5: 获取输入张量详细属性
    // ========================================================================
    PRINT_SEPARATOR("Step 5: 获取输入张量详细属性");
    
    std::vector<hbDNNTensorProperties> input_props(input_count);
    for (int i = 0; i < input_count; i++) {
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_props[i], dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        
        // 获取输入名称
        const char* input_name;
        hbDNNGetInputName(&input_name, dnn_handle, i);
        std::cout << "  输入名称[" << i << "]: " << input_name << std::endl;
        
        print_tensor_properties(input_props[i], i, true);
    }
    
    // ========================================================================
    // Step 6: 获取输出张量详细属性
    // ========================================================================
    PRINT_SEPARATOR("Step 6: 获取输出张量详细属性");
    
    std::vector<hbDNNTensorProperties> output_props(output_count);
    for (int i = 0; i < output_count; i++) {
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_props[i], dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        
        // 获取输出名称
        const char* output_name;
        hbDNNGetOutputName(&output_name, dnn_handle, i);
        std::cout << "  输出名称[" << i << "]: " << output_name << std::endl;
        
        print_tensor_properties(output_props[i], i, false);
    }
    
    // ========================================================================
    // Step 7: 获取模型描述信息
    // ========================================================================
    PRINT_SEPARATOR("Step 7: 获取模型描述信息");
    
    const char* model_desc = nullptr;
    uint32_t desc_size = 0;
    int32_t desc_type = 0;
    
    int desc_ret = hbDNNGetModelDesc(&model_desc, &desc_size, &desc_type, dnn_handle);
    if (desc_ret == 0 && model_desc != nullptr && desc_size > 0) {
        std::cout << "  描述类型: " << get_desc_type_name(desc_type) << std::endl;
        std::cout << "  描述大小: " << desc_size << " bytes" << std::endl;
        if (desc_size > 0) {
            std::string desc_str(model_desc, std::min((uint32_t)500, desc_size));
            std::cout << "  描述内容 (前500字符):" << std::endl;
            std::cout << "  " << desc_str << std::endl;
        }
    } else {
        std::cout << "  ⚠ 无法获取模型描述信息 (可能未嵌入)" << std::endl;
    }
    
    // ========================================================================
    // Step 8: 格式分析
    // ========================================================================
    PRINT_SEPARATOR("Step 8: 格式分析");
    
    // 分析NV12输入
    analyze_nv12_input(input_props, input_count);
    
    // 分析YOLO输出
    analyze_yolo_output(output_props, output_count);
    
    // ========================================================================
    // 可选测试
    // ========================================================================
    
    if (test_memory) {
        PRINT_SEPARATOR("可选测试: 内存分配");
        test_memory_allocation(input_props, output_props);
    }
    
    if (test_infer) {
        PRINT_SEPARATOR("可选测试: 推理流程");
        test_inference_flow(dnn_handle, input_props, output_props);
    }
    
    // ========================================================================
    // Step 9: 释放资源
    // ========================================================================
    PRINT_SEPARATOR("Step 9: 释放资源");
    
    RDK_CHECK_SUCCESS(
        hbDNNRelease(packed_dnn_handle),
        "hbDNNRelease failed");
    
    std::cout << "  ✓ 资源释放完成" << std::endl;
    
    PRINT_SEPARATOR("测试完成");
    std::cout << "  所有测试通过！" << std::endl;
    
    return 0;
}

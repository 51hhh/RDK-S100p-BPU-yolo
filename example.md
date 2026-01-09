RDKS100采用了和X5不同的异构编程接口也就是统一计算平台（Unify Compute Platform，以下简称 UCP）以实现对计算平台资源的调用，官方详细的介绍见UCP总览 - OpenExplorer，工具链手册中已经有了详细的API介绍，但我们这篇文章中为了避免大家频繁的来回切换网页，我们对于每个新出现的API函数还是会附上对应的介绍，话不多说，我们一步一步的开始实现推理代码，完整代码已传至ModelZoo

        首先我们先来完成代码的宏定义以便我们修改模型路径、类别名称等基本配置

// D-Robotics S100 *.hbm 模型路径
// Path of D-Robotics S100 *.hbm model.
#define MODEL_PATH "rdk_model_zoo_s/samples/Vision/ultralytics_YOLO_Detect/source/reference_hbm_models/yolov5nu_detect_nashe_640x640_nv12.hbm"
// 推理使用的测试图片路径
// Path of the test image used for inference.
#define TEST_IMG_PATH "rdk_model_zoo_s/resource/datasets/COCO2017/assets/bus.jpg"
// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0 
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE
// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "cpp_result.jpg"
// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 80
// NMS的阈值, 默认0.7
// Non-Maximum Suppression (NMS) threshold, default is 0.7
#define NMS_THRESHOLD 0.7
// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.25
// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16
// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0
// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0
// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0
// API运行控制
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
// COCO Names 类别名
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
// S100定制颜色
std::vector<cv::Scalar> rdk_colors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
};
        接着我们导入我们需要的所有头文件并手动完成softmax函数

// C/C++ Standard Libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
// Third Party Libraries
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
// RDK S100 UCP API
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
void softmax(float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}
        然后我们便可以一步一步的开始实现我们的板端推理代码啦，一切的一切首先肯定是需要加载我们的hbm模型，我们打开hb_dnn.h，和X5的SDK一样，UCP提供了两种加载模型的方式，分别是从文件加载以及从内存加载模型，这两种方式相比较来说FromFiles这一个函数由于文件I/O操作，相对较慢，代码较简单但是由于模型文件是独立存储存储的因此更加适合开发调试，而FromDDR这一个函数由于直接从内存读取，速度更快，适合嵌入式系统或需要快速加载的场景，但缺点便是代码较为复杂，比较贴近TensorRT加载模型的方式，两个API的具体介绍如下：

image-20250712235534531

/**
 * @brief Creates and initializes Horizon DNN Networks from file list
 * 
 * @param[out] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @param[in] modelFileNames Path of the model files.
 * @param[in] modelFileCount Number of the model files.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInitializeFromFiles(hbDNNPackedHandle_t *dnnPackedHandle,
                                 char const **modelFileNames,
                                 int32_t modelFileCount);

/**
 * @brief Creates and initializes Horizon DNN Networks from memory
 * 
 * @param[out] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @param[in] modelData Pointer to the model file
 * @param[in] modelDataLengths Length of the model data.
 * @param[in] modelDataCount Length of the model data.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInitializeFromDDR(hbDNNPackedHandle_t *dnnPackedHandle,
                               const void **modelData,
                               int32_t *modelDataLengths,
                               int32_t modelDataCount);
        我们可以看到这两个API都是传入模型然后以hbPackedDNNHandle_t结构体类型传出模型句柄，因此我们要使用这个函数的话我们首先需要用hbPackedDNNHandle_t创建一个变量packed_dnn_handle_，由于这部分和X5无差别，因此我在这里仅介绍更为常用的hbDNNInitializeFromFiles，由于我们前面利用宏定义来导入的模型路径，因此我们这里仅需要用一个字符指针变量来获取我们的模型路径地址，接着使用我们的错误检查宏来调用模型加载的API即可,具体代码如下：

hbDNNPackedHandle_t packed_dnn_handle;
const char *model_file_name = MODEL_PATH;
RDK_CHECK_SUCCESS(
    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
    "hbDNNInitializeFromFiles failed");
std::cout << "\033[31m Load D-Robotics S100 Quantize model time = " << std::fixed << std::setprecision(2) 
          << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
          << " ms\033[0m" << std::endl;
        接着我们需要从文件中获取模型信息包括模型的名称列表呀、模型句柄呀、输入信息呀以及输出信息等基本信息，这部分Get系的API在头文件中一共有12个函数，分别是：

image-20250712235524696

hbDNNGetModelNameList用于从hbDNNInitializeFromFiles函数加载模型文件获取的hbPackedDNNHandle_t模型句柄中提取所指向模型的名称列表和个数
/**
 * @brief Get model names from given packed handle
 * 
 * @param[out] modelNameList List of model names.
 * @param[out] modelNameCount Number of model names.
 * @param[in] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelNameList(char const ***modelNameList,
                              int32_t *modelNameCount,
                              hbDNNPackedHandle_t dnnPackedHandle);
hbDNNGetModelHandle()用于从 packedDNNHandle 所指向模型列表中获取一个模型的句柄并让调用方可以跨函数、跨线程使用返回的 dnnHandle
/**
 * @brief Get DNN Network handle from packed Handle with given model name
 * 
 * @param[out] dnnHandle DNN handle, pointing to one model.
 * @param[in] dnnPackedHandle DNN handle, pointing to multiple models.
 * @param[in] modelName Model name.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelHandle(hbDNNHandle_t *dnnHandle,
                            hbDNNPackedHandle_t dnnPackedHandle,
                            char const *modelName);
hbDNNGetInputCount()用于获取 dnnHandle 所指向模型输入张量的个数
/**
 * @brief Get input count
 * 
 * @param[out] inputCount Number of input tensors of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputCount(int32_t *inputCount, hbDNNHandle_t dnnHandle);
hbDNNGetInputName()用于获取 dnnHandle 所指向模型输入张量的名称
/**
 * @brief Get model input name
 * 
 * @param[out] name Name of the input tensor of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] inputIndex Index of the input tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputName(char const **name, hbDNNHandle_t dnnHandle,
                          int32_t inputIndex);
hbDNNGetInputTensorProperties()用于获取 dnnHandle 所指向模型特定输入张量的属性，其中hbDNNTensorProperties内容如下
/**
 * @brief Get input tensor properties
 * 
 * @param[out] properties Info of the input tensor.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] inputIndex Index of the input tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                      hbDNNHandle_t dnnHandle,
                                      int32_t inputIndex);

typedef struct hbDNNTensorProperties {
  hbDNNTensorShape validShape;
  int32_t tensorType;
  hbDNNQuantiScale scale;
  hbDNNQuantiType quantiType;
  int32_t quantizeAxis;
  int64_t alignedByteSize;
  int64_t stride[HB_DNN_TENSOR_MAX_DIMENSIONS];
} hbDNNTensorProperties;
hbDNNGetOutputCount()用于获取 dnnHandle 所指向模型输出张量的个数
/**
 * @brief Get output count
 * 
 * @param[out] outputCount Number of the output tensors of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputCount(int32_t *outputCount, hbDNNHandle_t dnnHandle);
hbDNNGetOutputName()用于获取 dnnHandle 所指向模型输出张量的名称
/**
 * @brief Get model output name
 * 
 * @param[out] name Name of the output tensor of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputName(char const **name, hbDNNHandle_t dnnHandle,
                           int32_t outputIndex);
hbDNNGetOutputTensorProperties()用于获取 dnnHandle 所指向模型特定输出张量的属性
/**
 * @brief Get output tensor properties
 * 
 * @param[out] properties Info of the output tensor.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbDNNHandle_t dnnHandle,
                                       int32_t outputIndex);
hbDNNGetInputDesc用于获取 dnnHandle 指向模型特定输入所关联的描述信息
/**
 * @brief Get model input description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] inputIndex Index of the input tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputDesc(char const **desc, uint32_t *size, int32_t *type,
                          hbDNNHandle_t dnnHandle, int32_t inputIndex);
hbDNNGetOutputDesc获取 dnnHandle 指向模型特定输出所关联的描述信息
/**
 * @brief Get model output description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputDesc(char const **desc, uint32_t *size, int32_t *type,
                           hbDNNHandle_t dnnHandle, int32_t outputIndex);
hbDNNGetModelDesc用于获取 dnnHandle 指向模型所关联的描述信息
/**
 * @brief Get model description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelDesc(char const **desc, uint32_t *size, int32_t *type,
                          hbDNNHandle_t dnnHandle);
hbDNNGetHBMDesc用于获取 dnnPackedHandle 和 index 指向hbm所关联的描述信息
/**
 * @brief Get hbm description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @param[in] index Index of multiple hbm models that are loaded through hbDNNInitializeFromFiles or hbDNNInitializeFromDDR, the index should be in the range of [0, modelFileCount) or [0, modelDataCount).
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetHBMDesc(char const **desc, uint32_t *size, int32_t *type,
                        hbDNNPackedHandle_t dnnPackedHandle, int32_t index);
        了解了模型信息相关的函数之后便可以完成对模型的加载同时在代码中对模型进行一些基本的检查以避免出错，我们首先先使用hbDNNGetModelNameList函数从我们上一步加载模型得到的packed_dnn_handle中获取我们加载的HBM模型里面的打包模型数量，因此我们根据API的要求创建model_name_list和model_count两个变量用来获取模型列表以及数量，接着我们便可以调用API并判断模型数量是否正确

const char **model_name_list;
int model_count = 0;
RDK_CHECK_SUCCESS(
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
    "hbDNNGetModelNameList failed");

if (model_count > 1) {
    std::cout << "This model file have more than 1 model, only use model 0." << std::endl;
}
const char *model_name = model_name_list[0];
std::cout << "[model name]: " << model_name << std::endl;
        完成了对模型本身的检查无误，我们便可以获取模型的一个让调用方可以跨函数、跨线程使用返回的 dnnHandle句柄，我们首先根据API的要求利用hbDNNHandle_t创建一个hbDNNHandle_t类的dnn_handle_模型句柄，接着便可以直接调用API获取啦

hbDNNHandle_t dnn_handle;
RDK_CHECK_SUCCESS(
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
    "hbDNNGetModelHandle failed");
        创建了模型句柄之后我们便可以从模型的句柄中获取我们模型的所有信息啦！比如模型输入输出个数、模型输入输出尺寸等，用到的函数也就是我们上面介绍的一些API，这部分涉及到API分别是hbDNNGetInputCount和hbDNNGetOutputCount用于获取模型网络输入输出的个数以及hbDNNGetInputTensorProperties和hbDNNGetOutputTensorProperties用来获取模型输入的张量，因此我们也可以在这里对我们模型进行新一轮的检查，我们目前已知我们YOLO系列模型都是单输入的，而且我们目标检测任务的模型应该只有六个输出，那么如果我们通过输入输出检测API得到的信息和我们已知的这些信息对不上，那就说明我们的模型出现了问题，同时我们发现hbDNNGetInputTensorPropertiesAPI输出是一个hbDNNTensorProperties类型的结构体，我们查看结构体定义可以发现这个结构体是一个嵌套结构体，里面通过嵌套hbDNNTensorShape结构体、hbDNNQuantiScale结构体、以及hbDNNQuantiType结构体能够准确的描述输入的张量信息，其结构体定义及每项成员的解释如下：

typedef struct hbDNNTensorProperties {
  hbDNNTensorShape validShape;//张量的有效形状，表示张量的真实尺寸
  int32_t tensorType;//张量的类型
  hbDNNQuantiScale scale;//量化缩放量
  hbDNNQuantiType quantiType;//量化类型
  int32_t quantizeAxis;//量化轴索引，仅按per-axis量化时生效
  int64_t alignedByteSize;//张量对齐内容的内存大小
  int64_t stride[HB_DNN_TENSOR_MAX_DIMENSIONS];//张量中validShape各维度步长，字节为单位
} hbDNNTensorProperties;

typedef struct hbDNNTensorShape {
  int32_t dimensionSize[HB_DNN_TENSOR_MAX_DIMENSIONS];//张量每个维度的大小
  int32_t numDimensions;//张量的维度
} hbDNNTensorShape;//例如一个张量 numDimensions=4，其数据排布为 1x4x224x224 则 dimensionSize 数组中按顺序存储数据 dimensionSize[0]=1、 dimensionSize[1]=4、dimensionSize[2]=224、 dimensionSize[3]=224

typedef struct hbDNNQuantiScale {
  int32_t scaleLen;//缩放数据的长度
  float *scaleData;//缩放数据的首地址
  int32_t zeroPointLen;//零点偏移数据的长度
  int32_t *zeroPointData;//零点偏移数据的首地址
} hbDNNQuantiScale;

typedef enum {
  NONE,  // 没有量化
  SCALE//量化类型为 SCALE
} hbDNNQuantiType;
        了解了这些结构体之后，我们便可以根据结构体参数来定义我们的一些变量，同时由于我们知道我们的模型是单输入的也知道我们输入的数据应该该是NV12，且数据排布是NCHW，同时输入Tensor数据的valid shape应为(1,3,H,W)，所以我们在使用API获取了我们的输入信息之后我们还可以利用这些安全信息进行一些输入的安全检查:

// Step1：首先是输入检查,对于我们的YOLO模型，输入应该仅为1
int32_t input_count = 0;
RDK_CHECK_SUCCESS(
    hbDNNGetInputCount(&input_count, dnn_handle),
    "hbDNNGetInputCount failed");

if (input_count < 1) {
    std::cout << "S100 YOLO model should have at least 1 input, but got " << input_count << std::endl;
    return -1;
} else if (input_count > 1) {
    std::cout << "S100 YOLO model has " << input_count << " inputs, using first input for inference" << std::endl;
} 
// Step2：接着获取模型的输入张量信息
hbDNNTensorProperties input_properties;
RDK_CHECK_SUCCESS(
    hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
    "hbDNNGetInputTensorProperties failed");
std::cout << "✓ input tensor type: " << input_properties.tensorType << std::endl;// S100 UCP 模型需要检测输入格式是否支持
// Step2.1：检测模型的输入格式是否为NV12 (type 3)
if (input_properties.tensorType != 3) {
    std::cout << "[ERROR] This program only supports NV12 input (type 3), but got type: " << input_properties.tensorType << std::endl;
    return -1;
}
// Step2.2：检测输入tensor布局是否为NCHW（与我们转换模型时填入的rt有关）
if (input_properties.validShape.numDimensions == 4) {
    int32_t channels = input_properties.validShape.dimensionSize[3];// NCHW布局，H和W应该在维度1和2位置，且通道数应该为1
    if (channels != 1) {
        std::cout << "[ERROR] This program expects NCHW layout with 1 channel, but got " << channels << " channels" << std::endl;
        return -1;
    }
    std::cout << "✓ input tensor layout: NCHW (verified)" << std::endl;
} else {
    std::cout << "[ERROR] Expected 4D input tensor for NCHW layout, but got " << input_properties.validShape.numDimensions << "D" << std::endl;
    return -1;
}
// Step2.3：获取模型的输入尺寸并检查
int32_t input_H, input_W;
if (input_properties.validShape.numDimensions == 4) {
    input_H = input_properties.validShape.dimensionSize[1];
    input_W = input_properties.validShape.dimensionSize[2];
    std::cout << "✓ input tensor valid shape: (" 
              << input_properties.validShape.dimensionSize[0] << ", "
              << input_H << ", " << input_W << ", "
              << input_properties.validShape.dimensionSize[3] << ")" << std::endl;
} else {
    std::cout << "S100 YOLO model input should be 4D" << std::endl;
    return -1;
}
        完成了对模型的输入张量信息的检查后，我们可以同步完成对模型输出的检查，这部分的逻辑和方式与输入检查一致，只不过是API改了个名字，因此不过多赘述啦，大家直接看代码！

// Step 4: 检查模型输出 - S100 YOLO 按照Readme导出后应该有6个输出
// Step 4: Check model output - S100 YOLO should have 6 outputs according to Readme
int32_t output_count = 0;
RDK_CHECK_SUCCESS(
    hbDNNGetOutputCount(&output_count, dnn_handle),
    "hbDNNGetOutputCount failed");

if (output_count != 6) {
    std::cout << "S100 YOLO model should have 6 outputs, but got " << output_count << std::endl;
    return -1;
}
std::cout << "✓ S100 YOLO model has 6 outputs" << std::endl;
// 打印输出信息并获取正确的输出顺序
std::cout << "\033[32m-> output tensors\033[0m" << std::endl;
for (int i = 0; i < 6; i++) {
    hbDNNTensorProperties output_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
    std::cout << "output[" << i << "] valid shape: (" 
              << output_properties.validShape.dimensionSize[0] << ", "
              << output_properties.validShape.dimensionSize[1] << ", "
              << output_properties.validShape.dimensionSize[2] << ", "
              << output_properties.validShape.dimensionSize[3] << "), ";

    std::cout << "QuantiType: " << output_properties.quantiType << std::endl;
}
        在完成了安全检查之后我们便可以进入模型前处理部分啦！图像的预处理无非就是图像尺寸的转换和图像格式的转换，所以这部分比较简单我就讲的稍微快一点啦，图像尺寸的变换我们采用letterbox的方式，众所周知，OpenCV中有一个图像转换的函数叫resize这个函数可以直接实现图像尺寸的变换，但是由于这个函数的实现过于简单粗暴，因此在图像尺寸不一致的情况下会改变图像的长宽比造成图像的失真，就比如如下情况，可以看到右边图像就发生了扭曲

image-20250713002235477

        而我们使用LetterBox的方式便可以看到，画面并没有产生扭曲变形，因为LetterBox的方式在对图片进行resize时，保持了原图的长宽比进行等比例缩放，当长边 resize 到需要的长度时，短边剩下的部分便采用灰色填充，这样便保持了原始图像的长宽比不变

image-20250713002302955

        但由于我们现在写的是通用代码，因此接下来我们会使用LetterBox和resize两种方式实现图像的预处理，resize不过多说啦使用opencv直接调用函数即可，我们主要讲LetterBox的方式，具体代码如下，其核心思想便是其核心思想便是通过按比例缩放图像以适应目标尺寸，同时保持原始图像的纵横比，为了确保图像在目标尺寸内居中，空白区域将使用填充的方式填充，通常填充色为中性色（如127, 127, 127）。这样，我们可以避免图像在缩放时出现失真，且确保图像的宽高比保持不变

// 前处理参数
float y_scale = 1.0, x_scale = 1.0;
int x_shift = 0, y_shift = 0;
cv::Mat resize_img;
begin_time = std::chrono::system_clock::now();
if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
    // LetterBox前处理
    float scale = std::min(1.0f * input_H / img.rows, 1.0f * input_W / img.cols);
    int new_w = int(img.cols * scale);
    int new_h = int(img.rows * scale);
    // 确保尺寸为偶数
    new_w = (new_w / 2) * 2;
    new_h = (new_h / 2) * 2;
    // 重新计算实际的缩放因子
    x_scale = 1.0f * new_w / img.cols;
    y_scale = 1.0f * new_h / img.rows;
    x_shift = (input_W - new_w) / 2;
    int x_other = input_W - new_w - x_shift;
    y_shift = (input_H - new_h) / 2;
    int y_other = input_H - new_h - y_shift;
    cv::Size targetSize(new_w, new_h);
    cv::resize(img, resize_img, targetSize);
    cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
} else {
    // Resize前处理
    cv::Size targetSize(input_W, input_H);
    cv::resize(img, resize_img, targetSize);
    y_scale = 1.0 * input_H / img.rows;
    x_scale = 1.0 * input_W / img.cols;
}
        完成了图像尺寸的缩放之后由于我们最开始编译模型的时候选择的是nv12的输入，因此我们还需要将我们输入图像从BGR格式转换为NV12格式，NV12格式是YUV420SP格式的一种，它将Y分量（亮度）和交错存储的UV分量（色度）分开存放，具体的实现代码如下：

// BGR转YUV420SP (NV12)
cv::Mat img_nv12;
cv::Mat yuv_mat;
cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
uint8_t *yuv = yuv_mat.ptr<uint8_t>();

img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
int uv_height = input_H / 2;
int uv_width = input_W / 2;
int y_size = input_H * input_W;

// 复制Y平面
memcpy(ynv12, yuv, y_size);

// 交错UV平面
uint8_t *nv12 = ynv12 + y_size;
uint8_t *u_data = yuv + y_size;
uint8_t *v_data = u_data + uv_height * uv_width;
for (int i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
}
        完成了前面对图像的操作之后我们便要开始准备模型的输入数据啦！接下来，我们需要将处理后的图像数据转换为我们的模型可以接受的输入格式，在这个过程中，我们首先要为输入张量分配内存，并将处理后的图像数据（YUV格式）复制到内存中，以确保模型能够正确地访问和使用这些数据。其中涉及到了一个API为hbUCPMallocCached，我们查看一下他的解释以及其中涉及到的结构体定义：

/**
 * @brief Allocate cacheable system memory 申请缓存的系统内存
 * 
 * @param[out] mem Memory pointer.
 * @param[in] size Size of the requested memory.
 * @param[in] deviceId Reserved parameter.
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbUCPMallocCached(hbUCPSysMem *mem, uint64_t size, int32_t deviceId);

typedef struct hbUCPSysMem {
  uint64_t phyAddr;
  void *virAddr;
  uint64_t memSize;
} hbUCPSysMem;

typedef struct hbDNNTensor {
  hbUCPSysMem sysMem;
  hbDNNTensorProperties properties;
} hbDNNTensor;
        根据API所示，我们首先要先创建一个hbUCPSysMem结构体，这个结构体用于描述内存的物理地址(phyAddr)、虚拟地址(virAddr)以及内存的大小(memSize)。接着，我们调用hbUCPMallocCached函数为输入张量分配内存，分配的内存是可缓存的，这意味着硬件可以在处理数据时直接访问此内存，而无需频繁与主内存进行交换，hbDNNTensor是用来存储整个张量信息的结构体，其中包含了多个hbSysMem结构体来描述不同部分的数据（比如输入、输出等）。而hbDNNTensorProperties则存储有关张量的属性信息，如张量的形状、数据类型、量化信息等。了解了以上信息之后我们便可以准备输入数据啦，我们需要创建一个std::vector<hbDNNTensor>类型的容器，用于存储所有的输入张量。每一个hbDNNTensor结构体代表一个输入张量，它包含了用于存储张量数据的内存信息（sysMem）以及张量的相关属性（properties）。在循环中，我们依次为每个输入张量配置其属性，并通过hbUCPMallocCached函数为其分配对应大小的缓存内存。在这个过程中，我们根据YUV格式的特点，将输入分为两个部分：第一个输入为Y分量，尺寸为640×640×1，对于该输入，我们按照张量格式要求设置其有效形状（validShape）为 [1, 640, 640, 1]，并设置步长信息（stride），以确保内存布局与模型期望一致。然后，我们使用hbUCPMallocCached分配对应大小的内存，并将图像的Y分量数据通过memcpy函数复制到分配好的内存中。第二个输入为UV分量，尺寸为320×320×2（U和V分别为一个通道，尺寸为原图的一半），类似地，我们为其设置形状为 [1, 320, 320, 2]，并根据通道和宽度计算正确的步长。之后，同样使用hbUCPMallocCached分配内存，并将原图中对应的UV数据区域复制到该输入张量中。完成数据复制后，为确保数据能够被设备正确读取，我们还需要调用hbUCPMemFlush函数对每块内存进行刷新操作，使用HB_SYS_MEM_CACHE_CLEAN参数以清理并同步缓存，具体代码如下：

std::vector<hbDNNTensor> input_tensors(input_count);
// 分配输入内存
for (int i = 0; i < input_count; i++) {
    // 复制输入tensor属性
    input_tensors[i].properties = input_properties;
    int data_size;
    if (i == 0) {
        // 第一个输入：Y分量 640x640x1
        data_size = input_H * input_W;
        // 设置tensor的stride信息
        input_tensors[i].properties.validShape.dimensionSize[0] = 1;
        input_tensors[i].properties.validShape.dimensionSize[1] = input_H;
        input_tensors[i].properties.validShape.dimensionSize[2] = input_W;
        input_tensors[i].properties.validShape.dimensionSize[3] = 1;
        // 设置stride 
        input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
        input_tensors[i].properties.stride[2] = 1;                    // 通道步长 = stride[3] * size[3] = 1 * 1
        input_tensors[i].properties.stride[1] = input_W;              // 行步长 = stride[2] * size[2] = 1 * 640 = 640
        input_tensors[i].properties.stride[0] = input_W * input_H;    // 整个tensor = stride[1] * size[1] = 640 * 640 = 409600
    } else {
        // 第二个输入：UV分量 320x320x2 (尺寸减半，2通道)
        int uv_h = input_H / 2;  // 320
        int uv_w = input_W / 2;  // 320
        data_size = uv_h * uv_w * 2;  // UV两个通道
        // 设置tensor的stride信息
        input_tensors[i].properties.validShape.dimensionSize[0] = 1;
        input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
        input_tensors[i].properties.validShape.dimensionSize[2] = uv_w; 
        input_tensors[i].properties.validShape.dimensionSize[3] = 2;
        // 设置stride
        input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
        input_tensors[i].properties.stride[2] = 2;                    // 通道步长 = stride[3] * size[3] = 1 * 2 = 2
        input_tensors[i].properties.stride[1] = uv_w * 2;             // 行步长 = stride[2] * size[2] = 2 * 320 = 640
        input_tensors[i].properties.stride[0] = uv_w * uv_h * 2;      // 整个tensor = stride[1] * size[1] = 640 * 320 = 204800
    }
    // 分配内存
    hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0);
    std::cout << "✓ Input tensor " << i << " memory allocated: " << data_size << " bytes" << std::endl;
    // 复制数据
    if (i == 0) {
        // 第一个输入：复制Y分量
        memcpy(input_tensors[i].sysMem.virAddr, ynv12, input_H * input_W);
        std::cout << "✓ Y component data copied to tensor " << i << std::endl;
    } else {
        // 第二个输入：复制UV分量 
        uint8_t *uv_src = ynv12 + input_H * input_W;  // UV数据在Y之后
        memcpy(input_tensors[i].sysMem.virAddr, uv_src, data_size);
        std::cout << "✓ UV component data copied to tensor " << i << std::endl;
    }
    // 刷新内存
    hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
}
        当然，为输入分配了内存我们肯定也要为输出分配内存，相比于输入，这部分就会简单一些，原因在于输出的形状、数据类型、内存对齐方式等信息通常是由模型结构自动决定的，我们只需根据模型输出的属性直接进行内存申请即可。在实际实现中，我们首先通过调用 hbDNNGetOutputTensorProperties 获取每一个输出张量的属性信息，并存入对应的 hbDNNTensorProperties 结构体中。该函数会从模型句柄 dnn_handle 中获取第 i 个输出的详细信息，包括张量的形状、对齐大小（alignedByteSize）、数据类型等，随后，我们从属性中读取该输出张量的对齐后内存大小 alignedByteSize，并使用 hbUCPMallocCached 函数为该输出张量分配对应大小的缓存内存。这里的 alignedByteSize 表示该输出在内存中所需的实际大小，已经根据平台的要求进行了字节对齐，确保后续在硬件访问时不会出现越界或访问异常的情况

// 分配输出内存
for (int i = 0; i < output_count; i++) {
    hbDNNTensorProperties &output_properties = output_tensors[i].properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
    int out_aligned_size = output_properties.alignedByteSize;
    hbUCPSysMem &mem = output_tensors[i].sysMem;
    hbUCPMallocCached(&mem, out_aligned_size, 0);
    std::cout << "✓ Output tensor " << i << " memory allocated: " << out_aligned_size << " bytes" << std::endl;
}
        完成了内存的分配之后，我们就可以正式开始模型推理啦！推理过程主要分为两个阶段：发起推理任务 和 等待推理结果，这中间涉及到几个关键API，包括 hbDNNInferV2、hbUCPSubmitTask 和 hbUCPWaitTaskDone，它们共同组成了推理执行流程

hbDNNInferV2它的作用是创建或绑定一个推理任务，并将输入张量送入模型进行处理，其中：task_handle 是指向任务句柄的指针，在此可以传入空指针以使用同步方式，或者传入一个未提交的任务句柄用于异步多模型推理；output_tensors 是模型输出张量数组的指针；input_tensors 是输入张量数组；dnn_handle 是模型的句柄，代表一个已经加载的模型，其可以运行在同步或异步模式下
/**
 * @brief DNN inference
 * 
 * @param[in/out] taskHandle:
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given *taskHandle is not nullptr, attach task to task handle, which represents multi model task. The given *taskHandle must be obtained through case1 and not already committed or released.
 * case3: given taskHandle is nullptr to run in sync mode with default ctrl param
 * @param[out] output Pointer to the output tensor array, the size of array should be equal to $(`hbDNNGetOutputCount`)
 * @param[in] input Input tensor array, the size of array should be equal to  $(`hbDNNGetInputCount`)
 * @param[in] dnnHandle Pointer to the dnn handle which represents model handle
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInferV2(hbUCPTaskHandle_t *taskHandle, hbDNNTensor *output,
                     hbDNNTensor const *input, hbDNNHandle_t dnnHandle);



HB_UCP_INITIALIZE_SCHED_PARAM，该宏用于快速初始化调度参数结构体 hbUCPSchedParam，控制任务调度的优先级、运行设备等属性
#define HB_UCP_INITIALIZE_SCHED_PARAM(param)    \
  {                                             \
    (param)->priority = HB_UCP_PRIORITY_LOWEST; \
    (param)->deviceId = 0U;                     \
    (param)->customId = 0;                      \
    (param)->backend = HB_UCP_CORE_ANY;         \
  }
typedef struct hbUCPSchedParam {
  int32_t priority;
  int64_t customId;
  uint64_t backend;
  uint32_t deviceId;
} hbUCPSchedParam;
hbUCPSubmitTask，该函数将一个已经通过 hbDNNInferV2 创建的任务提交给计算平台调度执行，其中taskHandle：指向已创建但尚未提交的任务句柄；schedParam：指向调度参数结构体，用于指定执行优先级、运行设备等。
/**
 * @brief Submit task to Unified Computing Platform with scheduling parameters
 * 
 * @param[in] taskHandle pointer to the task
 * @param[in] schedParam task schedule parameter
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPSubmitTask(hbUCPTaskHandle_t taskHandle,
                        hbUCPSchedParam *schedParam);
hbUCPWaitTaskDone，该函数阻塞当前线程，直到所提交的推理任务完成或超时，其中taskHandle：指向待等待的任务句柄；timeout：超时时间，单位为毫秒（ms），超时后函数返回。
/**
 * @brief Wait util task completed.
 * 
 * @param[in] taskHandle pointer to the task
 * @param[in] timeout timeout for waiting task, unit is milliseconds(ms)
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPWaitTaskDone(hbUCPTaskHandle_t taskHandle, int32_t timeout);
        这四个函数组合在一起，就组成了异步推理的核心流程：先使用 hbDNNInferV2 创建任务；再初始化调度参数；接着使用 hbUCPSubmitTask 提交任务；最后使用 hbUCPWaitTaskDone 等待结果；同步模式下也可以只使用 hbDNNInferV2，传入 nullptr 作为 taskHandle 即可完成一键式推理。

        之后我们便可以开始完成我们推理的代码部分啦，首先我们调用 hbDNNInferV2 发起推理请求，同时生成一个推理任务句柄（task_handle），用于后续提交与控制。注意，此时我们使用的是异步模式（传入了 task_handle 的地址），因此需要确保句柄非空，接着我们通过调用宏 HB_UCP_INITIALIZE_SCHED_PARAM 初始化调度参数结构体，设置调度优先级、设备ID、后端核心类型等信息，设置完调度参数后便可以提交推理任务啦，让统一计算平台（UCP）调度执行，最后调用 hbUCPWaitTaskDone 阻塞等待任务完成，我们将超时时间设为 10000 毫秒（即10秒），任务完成后便能安全地访问 output_tensors 中的结果，具体的代码实现如下：

// Step 7: 推理
// Step 7: Inference
std::cout << "\033[32m-> Starting inference\033[0m" << std::endl;
begin_time = std::chrono::system_clock::now();
// 生成任务句柄
hbUCPTaskHandle_t task_handle = nullptr;
int infer_ret = hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
if (infer_ret != 0) {
    std::cout << "[ERROR] hbDNNInferV2 failed with error code: " << infer_ret << std::endl;
    return -1;
}
//确保句柄非空
if (task_handle == nullptr) {
    std::cout << "[ERROR] task_handle is null after hbDNNInferV2" << std::endl;
    return -1;
}
std::cout << "✓ Inference task created successfully" << std::endl;
// 设置UCP调度参数
hbUCPSchedParam ctrl_param;
HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
ctrl_param.backend = HB_UCP_BPU_CORE_ANY;  // 使用任意BPU核心
// 提交任务到UCP
int submit_ret = hbUCPSubmitTask(task_handle, &ctrl_param);
if (submit_ret != 0) {
    std::cout << "[ERROR] hbUCPSubmitTask failed with error code: " << submit_ret << std::endl;
    return -1;
}
std::cout << "✓ Inference task submitted successfully" << std::endl;
// 等待任务完成，设置合理的超时时间(10秒)
int wait_ret = hbUCPWaitTaskDone(task_handle, 10000);
if (wait_ret != 0) {
    std::cout << "[ERROR] hbUCPWaitTaskDone failed with error code: " << wait_ret << std::endl;
    return -1;
}
std::cout << "✓ Inference task completed successfully" << std::endl;
std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) 
          << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
          << " ms\033[0m" << std::endl;
        在这个阶段，我们的主要目标是将模型的原始输出（通常是特征图或候选框回归结果）还原为实际图像坐标系中的检测框或结果，也就是通常说的 目标框解析、置信度判断、NMS等处理，而为了正确地解码每个输出位置对应的预测框，我们首先需要为每个输出特征图预先计算锚点（anchor）坐标，YOLO 系列模型采用的是一种密集预测机制，每个输出特征图上的位置都对应着一个锚点（即一个候选框的中心），模型的输出是相对于这些锚点的偏移量（如中心坐标偏移、宽高缩放等），因此我们需要提前根据特征图的尺寸（如 80×80、40×40、20×20）和对应的 stride（如 8、16、32）计算每个位置的中心坐标 (w + 0.5, h + 0.5)，为后续将偏移量还原为原图尺度下的目标框做好准备，具体的代码如下：

// 预计算锚点
// s_anchor: 80x80 (stride=8)
std::vector<std::pair<float, float>> s_anchor(80 * 80);
for (int h = 0; h < 80; h++) {
    for (int w = 0; w < 80; w++) {
        s_anchor[h * 80 + w] = {w + 0.5f, h + 0.5f};
    }
}
// m_anchor: 40x40 (stride=16) 
std::vector<std::pair<float, float>> m_anchor(40 * 40);
for (int h = 0; h < 40; h++) {
    for (int w = 0; w < 40; w++) {
        m_anchor[h * 40 + w] = {w + 0.5f, h + 0.5f};
    }
}
// l_anchor: 20x20 (stride=32)
std::vector<std::pair<float, float>> l_anchor(20 * 20);
for (int h = 0; h < 20; h++) {
    for (int w = 0; w < 20; w++) {
        l_anchor[h * 20 + w] = {w + 0.5f, h + 0.5f};
    }
}
        然后我们便可以开始处理每个尺度的输出啦！在 YOLO 的结构中，通常会有多个输出分支，分别对应不同感受野的特征图（如 80×80、40×40、20×20），我们需要逐个处理它们，提取出有效的目标候选框。首先，我们遍历三个尺度的输出，对每个尺度，根据其在输出张量中的索引位置（例如类别分支在 0、2、4，边界框分支在 1、3、5）提取出类别预测数据和边框预测数据，并根据当前尺度确定 stride 与 grid 大小；然后，通过 hbUCPMemFlush 刷新 BPU 缓存，确保我们读取到的输出数据是最新的内容；接下来，通过指针分别获取类别预测（cls_data）和边框预测（bbox_data）的内存地址，并准备好浮点 scale 信息用于后续解码；在处理前，我们先进行第一步筛选：遍历当前尺度下所有 anchor 点（共 grid_size² 个），找出在所有类别中得分最高的类别，并判断其 raw 值是否超过设定的阈值（CONF_THRES_RAW）。如果通过筛选，就将该 anchor 的索引、预测类别和 Sigmoid 分数保存下来，供下一步做目标框的解码和拼接，通过这一轮筛选，我们从原始的密集预测中选出了可能包含目标的 anchor 点，大大减少了后续处理的数据量

        接着第二步我们继续处理有效检测的边界框。首先，对于每个通过置信度阈值筛选的 anchor，我们从边框预测数据中读取对应的离散回归分布（DFL），对每个边界框的四条边分别进行 softmax 计算，得到概率分布后计算期望值以恢复真实的距离偏移。然后结合预先计算的锚点坐标和当前尺度的 stride，将偏移转换为原图尺度下的边界框左上角和右下角坐标（xyxy格式）。最后，我们判断该边界框是否有效（宽高均大于0），如果合法则将边框、对应置信度和类别 ID 分别存入 all_bboxes、all_scores 和 all_ids 三个容器中，方便后续进行非极大值抑制（NMS）和结果输出。这样，通过逐尺度处理，我们完成了模型输出的解码和有效目标框的初步筛选。

// 计算置信度阈值的原始值（利用Sigmoid函数的单调性）
float CONF_THRES_RAW = -std::log(1.0f / SCORE_THRESHOLD - 1.0f);
// 处理3个特征层的输出
std::vector<cv::Rect2d> all_bboxes;
std::vector<float> all_scores;
std::vector<int> all_ids;
// 处理每个尺度
for (int scale = 0; scale < 3; scale++) {
    int cls_idx = scale * 2;     // 0, 2, 4
    int bbox_idx = scale * 2 + 1; // 1, 3, 5
    int stride = (scale == 0) ? 8 : (scale == 1) ? 16 : 32;
    int grid_size = (scale == 0) ? 80 : (scale == 1) ? 40 : 20;
    // 刷新BPU内存
    hbUCPMemFlush(&(output_tensors[cls_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    hbUCPMemFlush(&(output_tensors[bbox_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    // 获取输出数据指针
    auto *cls_data = reinterpret_cast<float *>(output_tensors[cls_idx].sysMem.virAddr);
    auto *bbox_data = reinterpret_cast<int32_t *>(output_tensors[bbox_idx].sysMem.virAddr);
    auto *bbox_scale = reinterpret_cast<float *>(output_tensors[bbox_idx].properties.scale.scaleData);
    int total_anchors = grid_size * grid_size;
    // 第一步：找到所有超过阈值的位置
    std::vector<int> valid_indices;
    std::vector<int> valid_class_ids;
    std::vector<float> valid_scores;
    for (int i = 0; i < total_anchors; i++) {
        float *cur_cls = cls_data + i * CLASSES_NUM;
        // 找到最大分数和对应类别
        int max_cls_id = 0;
        for (int c = 1; c < CLASSES_NUM; c++) {
            if (cur_cls[c] > cur_cls[max_cls_id]) {
                max_cls_id = c;
            }
        }
        // 检查是否超过阈值（raw值比较）
        if (cur_cls[max_cls_id] >= CONF_THRES_RAW) {
            valid_indices.push_back(i);
            valid_class_ids.push_back(max_cls_id);
            // 计算Sigmoid分数
            float score = 1.0f / (1.0f + std::exp(-cur_cls[max_cls_id]));
            valid_scores.push_back(score);
        }
    }
    // 第二步：处理有效检测的边界框
    for (size_t idx = 0; idx < valid_indices.size(); idx++) {
        int anchor_idx = valid_indices[idx];
        int32_t *cur_bbox = bbox_data + anchor_idx * (REG * 4);
        // DFL计算 - 对每条边进行处理
        float ltrb[4];
        for (int i = 0; i < 4; i++) {
            float dfl_values[REG];
            float softmax_values[REG];
            // 反量化DFL值
            for (int j = 0; j < REG; j++) {
                int scale_idx = i * REG + j;
                dfl_values[j] = float(cur_bbox[scale_idx]) * bbox_scale[scale_idx];
            }
            // Softmax
            softmax(dfl_values, softmax_values, REG);
            // 计算期望值（DFL到距离的转换）
            ltrb[i] = 0.0f;
            for (int j = 0; j < REG; j++) {
                ltrb[i] += softmax_values[j] * j;
            }
        }
        // 获取锚点坐标
        float anchor_x, anchor_y;
        if (scale == 0) {
            anchor_x = s_anchor[anchor_idx].first;
            anchor_y = s_anchor[anchor_idx].second;
        } else if (scale == 1) {
            anchor_x = m_anchor[anchor_idx].first;
            anchor_y = m_anchor[anchor_idx].second;
        } else {
            anchor_x = l_anchor[anchor_idx].first;
            anchor_y = l_anchor[anchor_idx].second;
        }
        // ltrb转xyxy坐标
        double x1 = (anchor_x - ltrb[0]) * stride;
        double y1 = (anchor_y - ltrb[1]) * stride;
        double x2 = (anchor_x + ltrb[2]) * stride;
        double y2 = (anchor_y + ltrb[3]) * stride;
        // 检查边界框合法性
        if (x2 > x1 && y2 > y1) {
            all_bboxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            all_scores.push_back(valid_scores[idx]);
            all_ids.push_back(valid_class_ids[idx]);
        }
    }
}
        后处理最后一步就是分类别的NMS啦，我们为每个类别分别创建容器，收集该类别所有的检测框和对应置信度分数，同时保存它们在全局结果中的索引。然后调用 OpenCV 提供的 cv::dnn::NMSBoxes 函数，传入该类别的边界框和分数，设置置信度阈值和重叠阈值（SCORE_THRESHOLD 和 NMS_THRESHOLD），函数返回经过 NMS 后保留的边界框索引，最后，我们将这些类别内保留的索引映射回全局检测结果的索引，并统计 NMS 后总的检测数量这样我们便得到了去重且可信的目标检测框，具体代码如下：

// Step 9: 分类别NMS处理
// Step 9: Class-wise NMS processing
std::vector<std::vector<int>> nms_indices(CLASSES_NUM);
int total_detections_before_nms = all_bboxes.size();
int total_detections_after_nms = 0;

for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
    // 收集该类别的所有检测
    std::vector<cv::Rect2d> class_bboxes;
    std::vector<float> class_scores;
    std::vector<int> original_indices;

    for (size_t i = 0; i < all_bboxes.size(); i++) {
        if (all_ids[i] == cls_id) {
            class_bboxes.push_back(all_bboxes[i]);
            class_scores.push_back(all_scores[i]);
            original_indices.push_back(i);
        }
    }

    if (!class_bboxes.empty()) {
        std::vector<int> class_nms_indices;
        cv::dnn::NMSBoxes(class_bboxes, class_scores, 
                         SCORE_THRESHOLD, NMS_THRESHOLD, class_nms_indices);

        // 将类别内的索引转换为全局索引
        for (int idx : class_nms_indices) {
            nms_indices[cls_id].push_back(original_indices[idx]);
        }
        total_detections_after_nms += class_nms_indices.size();
    }
}
        最后我们便只需要根据NMS之后的结果绘制图像和打印输出就可以啦！！！具体代码如下，主要是CV操作这里就不赘述啦！

// Step 10: 绘制结果
// Step 10: Draw results
for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
    for (int global_idx : nms_indices[cls_id]) {
        // 坐标转换回原图
        float x1 = (all_bboxes[global_idx].x - x_shift) / x_scale;
        float y1 = (all_bboxes[global_idx].y - y_shift) / y_scale;
        float x2 = x1 + all_bboxes[global_idx].width / x_scale;
        float y2 = y1 + all_bboxes[global_idx].height / y_scale;
        float score = all_scores[global_idx];
        // 边界检查
        x1 = std::max(0.0f, std::min((float)img.cols - 1, x1));
        y1 = std::max(0.0f, std::min((float)img.rows - 1, y1));
        x2 = std::max(0.0f, std::min((float)img.cols - 1, x2));
        y2 = std::max(0.0f, std::min((float)img.rows - 1, y2));
        // 绘制边界框
        cv::Scalar color = rdk_colors[cls_id % 20];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, LINE_SIZE);
        // 绘制标签
        std::string label = object_names[cls_id] + ": " + std::to_string(int(score * 100)) + "%";
        int baseline;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_THICKNESS, &baseline);
        cv::Point label_pos(x1, y1 - 10 > textSize.height ? y1 - 10 : y1 + textSize.height + 10);
        cv::rectangle(img, label_pos + cv::Point(0, baseline), 
                     label_pos + cv::Point(textSize.width, -textSize.height), color, cv::FILLED);
        cv::putText(img, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 0), FONT_THICKNESS);
        // 打印检测结果
        std::cout << "(" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ") -> " 
                  << object_names[cls_id] << ": " << std::fixed << std::setprecision(2) << score << std::endl;
    }
}
cv::imwrite(IMG_SAVE_PATH, img);
        至此我们的推理便完成了，但是在彻底结束前我们还需要对我们推理的资源进行释放，这里主要设计到了三个函数，分别用于释放任务句柄、清理内存以及释放模型句柄

/**
 * @brief Release a task and its related resources.
 * 
 * @param[in] taskHandle pointer to the task
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPReleaseTask(hbUCPTaskHandle_t taskHandle);

/**
 * @brief Free mem
 * 
 * @param[in] mem Memory pointer.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPFree(hbUCPSysMem *mem);

/**
 * @brief Release DNN Networks in a given packed handle
 * 
 * @param[in] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNRelease(hbDNNPackedHandle_t dnnPackedHandle);
        释放的代码如下：

// Step 12: 资源释放
// Step 12: Release resources
std::cout << "\033[32m-> Cleaning up resources\033[0m" << std::endl;

// 释放任务句柄
hbUCPReleaseTask(task_handle);

// 释放输入内存
for (int i = 0; i < input_count; i++) {
    hbUCPFree(&(input_tensors[i].sysMem));
}

// 释放输出内存
for (int i = 0; i < output_count; i++) {
    hbUCPFree(&(output_tensors[i].sysMem));
}

// 释放模型
hbDNNRelease(packed_dnn_handle);
        至此推理的全流程代码我们便完成啦！！！CMakeLists及完整的代码如下：

cmake_minimum_required(VERSION 3.0)
project(rdk_s100_yolo_detect)
# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置编译器标志
# libdnn.so depends on system software dynamic link library, use -Wl,-unresolved-symbols=ignore-in-shared-libs to shield during compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -Werror -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -Werror -O3 ")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# 设置OpenCV包
find_package(OpenCV REQUIRED)
# S100 UCP库路径配置
set(HOBOT_INCLUDE_PATH "/usr/include")
set(HOBOT_LIB_PATH "/usr/hobot/lib")
# 包含头文件路径
include_directories(${HOBOT_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
# 链接库路径
link_directories(${HOBOT_LIB_PATH})
# 添加可执行文件
add_executable(main main.cc)
# 链接所需的库
target_link_libraries(main
                      ${OpenCV_LIBS}    # OpenCV库
                      dnn               # S100 DNN推理库
                      hbucp             # S100 UCP统一计算平台库
                      pthread           # 多线程支持
                      rt                # 实时库
                      dl                # 动态链接库支持
                      m                 # 数学库
                      )
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2025，SkyXZ D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// 注意: 此程序在RDK S100板端运行
// Attention: This program runs on RDK S100 board.
// D-Robotics S100 *.hbm 模型路径
// Path of D-Robotics S100 *.hbm model.
#define MODEL_PATH "rdk_model_zoo_s/samples/Vision/ultralytics_YOLO_Detect/source/reference_hbm_models/yolov5nu_detect_nashe_640x640_nv12.hbm"
// 推理使用的测试图片路径
// Path of the test image used for inference.
#define TEST_IMG_PATH "rdk_model_zoo_s/resource/datasets/COCO2017/assets/bus.jpg"
// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0 
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE
// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "cpp_result.jpg"
// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 80
// NMS的阈值, 默认0.7
// Non-Maximum Suppression (NMS) threshold, default is 0.7
#define NMS_THRESHOLD 0.7
// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.25
// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16
// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0
// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0
// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0
// C/C++ Standard Libraries
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
// Third Party Libraries
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
// RDK S100 UCP API
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
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
// COCO Names
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
// S100定制颜色
std::vector<cv::Scalar> rdk_colors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
};
// Softmax function for DFL calculation
void softmax(float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    } 
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}
int main()
{
    // Step 0: 加载S100 hbm模型
    // Step 0: Load S100 hbm model
    auto begin_time = std::chrono::system_clock::now();
    hbDNNPackedHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics S100 Quantize model time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 1: 打印基本信息
    // Step 1: Print basic information
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;
    // Step 2: 获取模型句柄
    // Step 2: Get model handle
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");
    if (model_count > 1) {
        std::cout << "This model file have more than 1 model, only use model 0." << std::endl;
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;
    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");
    // Step 3: 检查模型输入
    // Step 3: Check model input
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");
    if (input_count < 1) {
        std::cout << "S100 YOLO model should have at least 1 input, but got " << input_count << std::endl;
        return -1;
    } else if (input_count > 1) {
        std::cout << "S100 YOLO model has " << input_count << " inputs, using first input for inference" << std::endl;
    } 
    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");
    // S100 UCP 模型需要检测输入格式是否支持
    std::cout << "✓ input tensor type: " << input_properties.tensorType << std::endl; 
    // 检测输入格式是否为NV12 (type 3)
    if (input_properties.tensorType != 3) {
        std::cout << "[ERROR] This program only supports NV12 input (type 3), but got type: " << input_properties.tensorType << std::endl;
        return -1;
    }
    // 检测输入tensor布局为NCHW
    if (input_properties.validShape.numDimensions == 4) {
        // NCHW布局，H和W应该在维度1和2位置，且通道数应该为1
        int32_t channels = input_properties.validShape.dimensionSize[3];
        if (channels != 1) {
            std::cout << "[ERROR] This program expects NCHW layout with 1 channel, but got " << channels << " channels" << std::endl;
            return -1;
        }
        std::cout << "✓ input tensor layout: NCHW (verified)" << std::endl;
    } else {
        std::cout << "[ERROR] Expected 4D input tensor for NCHW layout, but got " << input_properties.validShape.numDimensions << "D" << std::endl;
        return -1;
    }
    // 获取输入尺寸
    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4) {
        input_H = input_properties.validShape.dimensionSize[1];
        input_W = input_properties.validShape.dimensionSize[2];
        std::cout << "✓ input tensor valid shape: (" 
                  << input_properties.validShape.dimensionSize[0] << ", "
                  << input_H << ", " << input_W << ", "
                  << input_properties.validShape.dimensionSize[3] << ")" << std::endl;
    } else {
        std::cout << "S100 YOLO model input should be 4D" << std::endl;
        return -1;
    }
    // Step 4: 检查模型输出 - S100 YOLO 按照Readme导出后应该有6个输出
    // Step 4: Check model output - S100 YOLO should have 6 outputs according to Readme
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");
    if (output_count != 6) {
        std::cout << "S100 YOLO model should have 6 outputs, but got " << output_count << std::endl;
        return -1;
    }
    std::cout << "✓ S100 YOLO model has 6 outputs" << std::endl;
    // 打印输出信息并获取正确的输出顺序
    std::cout << "\033[32m-> output tensors\033[0m" << std::endl;
    for (int i = 0; i < 6; i++) {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        std::cout << "output[" << i << "] valid shape: (" 
                  << output_properties.validShape.dimensionSize[0] << ", "
                  << output_properties.validShape.dimensionSize[1] << ", "
                  << output_properties.validShape.dimensionSize[2] << ", "
                  << output_properties.validShape.dimensionSize[3] << "), "; 
        std::cout << "QuantiType: " << output_properties.quantiType << std::endl;
    }
    // Step 5: 前处理 - 读取图像并转换为YUV420SP
    // Step 5: Preprocessing - Load image and convert to YUV420SP
    std::cout << "\033[32m-> Starting preprocessing\033[0m" << std::endl;
    cv::Mat img = cv::imread(TEST_IMG_PATH);
    if (img.empty()) {
        std::cout << "Failed to load image: " << TEST_IMG_PATH << std::endl;
        return -1;
    }
    std::cout << "✓ img path: " << TEST_IMG_PATH << std::endl;
    std::cout << "✓ img (rows, cols, channels): (" << img.rows << ", " << img.cols << ", " << img.channels() << ")" << std::endl;
    // 前处理参数
    float y_scale = 1.0, x_scale = 1.0;
    int x_shift = 0, y_shift = 0;
    cv::Mat resize_img;
    begin_time = std::chrono::system_clock::now();
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // LetterBox前处理
        float scale = std::min(1.0f * input_H / img.rows, 1.0f * input_W / img.cols);
        int new_w = int(img.cols * scale);
        int new_h = int(img.rows * scale);
        // 确保尺寸为偶数
        new_w = (new_w / 2) * 2;
        new_h = (new_h / 2) * 2;
        // 重新计算实际的缩放因子
        x_scale = 1.0f * new_w / img.cols;
        y_scale = 1.0f * new_h / img.rows;
        x_shift = (input_W - new_w) / 2;
        int x_other = input_W - new_w - x_shift;
        y_shift = (input_H - new_h) / 2;
        int y_other = input_H - new_h - y_shift;
        cv::Size targetSize(new_w, new_h);
        cv::resize(img, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        // Resize前处理
        cv::Size targetSize(input_W, input_H);
        cv::resize(img, resize_img, targetSize);
        y_scale = 1.0 * input_H / img.rows;
        x_scale = 1.0 * input_W / img.cols;
    }
    std::cout << "✓ y_scale = " << y_scale << ", x_scale = " << x_scale << std::endl;
    std::cout << "✓ y_shift = " << y_shift << ", x_shift = " << x_shift << std::endl;
    // BGR转YUV420SP (NV12)
    cv::Mat img_nv12;
    cv::Mat yuv_mat;
    cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int uv_height = input_H / 2;
    int uv_width = input_W / 2;
    int y_size = input_H * input_W;  
    // 复制Y平面
    memcpy(ynv12, yuv, y_size);   
    // 交错UV平面
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    std::cout << "\033[31m pre process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 6: 准备输入tensor
    // Step 6: Prepare input tensor
    std::vector<hbDNNTensor> input_tensors(input_count);
    std::vector<hbDNNTensor> output_tensors(output_count);
    // 分配输入内存
    for (int i = 0; i < input_count; i++) {
        // 复制输入tensor属性
        input_tensors[i].properties = input_properties; 
        int data_size;
        if (i == 0) {
            // 第一个输入：Y分量 640x640x1
            data_size = input_H * input_W;
            // 设置tensor的stride信息
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = input_H;
            input_tensors[i].properties.validShape.dimensionSize[2] = input_W;
            input_tensors[i].properties.validShape.dimensionSize[3] = 1;
            // 设置stride 
            input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
            input_tensors[i].properties.stride[2] = 1;                    // 通道步长 = stride[3] * size[3] = 1 * 1
            input_tensors[i].properties.stride[1] = input_W;              // 行步长 = stride[2] * size[2] = 1 * 640 = 640
            input_tensors[i].properties.stride[0] = input_W * input_H;    // 整个tensor = stride[1] * size[1] = 640 * 640 = 409600
        } else {
            // 第二个输入：UV分量 320x320x2 (尺寸减半，2通道)
            int uv_h = input_H / 2;  // 320
            int uv_w = input_W / 2;  // 320
            data_size = uv_h * uv_w * 2;  // UV两个通道
            // 设置tensor的stride信息
            input_tensors[i].properties.validShape.dimensionSize[0] = 1;
            input_tensors[i].properties.validShape.dimensionSize[1] = uv_h;
            input_tensors[i].properties.validShape.dimensionSize[2] = uv_w; 
            input_tensors[i].properties.validShape.dimensionSize[3] = 2;
            // 设置stride
            input_tensors[i].properties.stride[3] = 1;                    // 每个元素1字节
            input_tensors[i].properties.stride[2] = 2;                    // 通道步长 = stride[3] * size[3] = 1 * 2 = 2
            input_tensors[i].properties.stride[1] = uv_w * 2;             // 行步长 = stride[2] * size[2] = 2 * 320 = 640
            input_tensors[i].properties.stride[0] = uv_w * uv_h * 2;      // 整个tensor = stride[1] * size[1] = 640 * 320 = 204800
        }
        // 分配内存
        hbUCPMallocCached(&input_tensors[i].sysMem, data_size, 0);
        std::cout << "✓ Input tensor " << i << " memory allocated: " << data_size << " bytes" << std::endl; 
        // 复制数据
        if (i == 0) {
            // 第一个输入：复制Y分量
            memcpy(input_tensors[i].sysMem.virAddr, ynv12, input_H * input_W);
            std::cout << "✓ Y component data copied to tensor " << i << std::endl;
        } else {
            // 第二个输入：复制UV分量 
            uint8_t *uv_src = ynv12 + input_H * input_W;  // UV数据在Y之后
            memcpy(input_tensors[i].sysMem.virAddr, uv_src, data_size);
            std::cout << "✓ UV component data copied to tensor " << i << std::endl;
        }     
        // 刷新内存
        hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }
    // 分配输出内存
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output_tensors[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbUCPSysMem &mem = output_tensors[i].sysMem;
        hbUCPMallocCached(&mem, out_aligned_size, 0);
        std::cout << "✓ Output tensor " << i << " memory allocated: " << out_aligned_size << " bytes" << std::endl;
    }
    // Step 7: 推理
    // Step 7: Inference
    std::cout << "\033[32m-> Starting inference\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    // 生成任务句柄
    hbUCPTaskHandle_t task_handle = nullptr;
    int infer_ret = hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle);
    if (infer_ret != 0) {
        std::cout << "[ERROR] hbDNNInferV2 failed with error code: " << infer_ret << std::endl;
        return -1;
    }
    if (task_handle == nullptr) {
        std::cout << "[ERROR] task_handle is null after hbDNNInferV2" << std::endl;
        return -1;
    }
    std::cout << "✓ Inference task created successfully" << std::endl;
    // 设置UCP调度参数
    hbUCPSchedParam ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;  // 使用任意BPU核心
    // 提交任务到UCP
    int submit_ret = hbUCPSubmitTask(task_handle, &ctrl_param);
    if (submit_ret != 0) {
        std::cout << "[ERROR] hbUCPSubmitTask failed with error code: " << submit_ret << std::endl;
        return -1;
    }
    std::cout << "✓ Inference task submitted successfully" << std::endl;
    // 等待任务完成，设置合理的超时时间(10秒)
    int wait_ret = hbUCPWaitTaskDone(task_handle, 10000);
    if (wait_ret != 0) {
        std::cout << "[ERROR] hbUCPWaitTaskDone failed with error code: " << wait_ret << std::endl;
        return -1;
    }
    std::cout << "✓ Inference task completed successfully" << std::endl;
    std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 8: 后处理
    // Step 8: Post-processing
    std::cout << "\033[32m-> Starting post-processing\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now();
    // 计算置信度阈值的原始值（利用Sigmoid函数的单调性）
    float CONF_THRES_RAW = -std::log(1.0f / SCORE_THRESHOLD - 1.0f);
    // 预计算锚点
    // s_anchor: 80x80 (stride=8)
    std::vector<std::pair<float, float>> s_anchor(80 * 80);
    for (int h = 0; h < 80; h++) {
        for (int w = 0; w < 80; w++) {
            s_anchor[h * 80 + w] = {w + 0.5f, h + 0.5f};
        }
    }
    // m_anchor: 40x40 (stride=16) 
    std::vector<std::pair<float, float>> m_anchor(40 * 40);
    for (int h = 0; h < 40; h++) {
        for (int w = 0; w < 40; w++) {
            m_anchor[h * 40 + w] = {w + 0.5f, h + 0.5f};
        }
    }   
    // l_anchor: 20x20 (stride=32)
    std::vector<std::pair<float, float>> l_anchor(20 * 20);
    for (int h = 0; h < 20; h++) {
        for (int w = 0; w < 20; w++) {
            l_anchor[h * 20 + w] = {w + 0.5f, h + 0.5f};
        }
    }
    // 处理3个特征层的输出
    std::vector<cv::Rect2d> all_bboxes;
    std::vector<float> all_scores;
    std::vector<int> all_ids;
    // 处理每个尺度
    for (int scale = 0; scale < 3; scale++) {
        int cls_idx = scale * 2;     // 0, 2, 4
        int bbox_idx = scale * 2 + 1; // 1, 3, 5
        int stride = (scale == 0) ? 8 : (scale == 1) ? 16 : 32;
        int grid_size = (scale == 0) ? 80 : (scale == 1) ? 40 : 20;
        // 刷新BPU内存
        hbUCPMemFlush(&(output_tensors[cls_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        hbUCPMemFlush(&(output_tensors[bbox_idx].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        // 获取输出数据指针
        auto *cls_data = reinterpret_cast<float *>(output_tensors[cls_idx].sysMem.virAddr);
        auto *bbox_data = reinterpret_cast<int32_t *>(output_tensors[bbox_idx].sysMem.virAddr);
        auto *bbox_scale = reinterpret_cast<float *>(output_tensors[bbox_idx].properties.scale.scaleData);
        int total_anchors = grid_size * grid_size; 
        // 第一步：找到所有超过阈值的位置
        std::vector<int> valid_indices;
        std::vector<int> valid_class_ids;
        std::vector<float> valid_scores; 
        for (int i = 0; i < total_anchors; i++) {
            float *cur_cls = cls_data + i * CLASSES_NUM;
            // 找到最大分数和对应类别
            int max_cls_id = 0;
            for (int c = 1; c < CLASSES_NUM; c++) {
                if (cur_cls[c] > cur_cls[max_cls_id]) {
                    max_cls_id = c;
                }
            }           
            // 检查是否超过阈值（raw值比较）
            if (cur_cls[max_cls_id] >= CONF_THRES_RAW) {
                valid_indices.push_back(i);
                valid_class_ids.push_back(max_cls_id);
                // 计算Sigmoid分数
                float score = 1.0f / (1.0f + std::exp(-cur_cls[max_cls_id]));
                valid_scores.push_back(score);
            }
        }       
        // 第二步：处理有效检测的边界框
        for (size_t idx = 0; idx < valid_indices.size(); idx++) {
            int anchor_idx = valid_indices[idx];
            int32_t *cur_bbox = bbox_data + anchor_idx * (REG * 4);            
            // DFL计算 - 对每条边进行处理
            float ltrb[4];
            for (int i = 0; i < 4; i++) {
                float dfl_values[REG];
                float softmax_values[REG];                
                // 反量化DFL值
                for (int j = 0; j < REG; j++) {
                    int scale_idx = i * REG + j;
                    dfl_values[j] = float(cur_bbox[scale_idx]) * bbox_scale[scale_idx];
                }                
                // Softmax
                softmax(dfl_values, softmax_values, REG);                
                // 计算期望值（DFL到距离的转换）
                ltrb[i] = 0.0f;
                for (int j = 0; j < REG; j++) {
                    ltrb[i] += softmax_values[j] * j;
                }
            }           
            // 获取锚点坐标
            float anchor_x, anchor_y;
            if (scale == 0) {
                anchor_x = s_anchor[anchor_idx].first;
                anchor_y = s_anchor[anchor_idx].second;
            } else if (scale == 1) {
                anchor_x = m_anchor[anchor_idx].first;
                anchor_y = m_anchor[anchor_idx].second;
            } else {
                anchor_x = l_anchor[anchor_idx].first;
                anchor_y = l_anchor[anchor_idx].second;
            }            
            // ltrb转xyxy坐标
            double x1 = (anchor_x - ltrb[0]) * stride;
            double y1 = (anchor_y - ltrb[1]) * stride;
            double x2 = (anchor_x + ltrb[2]) * stride;
            double y2 = (anchor_y + ltrb[3]) * stride;            
            // 检查边界框合法性
            if (x2 > x1 && y2 > y1) {
                all_bboxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                all_scores.push_back(valid_scores[idx]);
                all_ids.push_back(valid_class_ids[idx]);
            }
        }
    }
    // Step 9: 分类别NMS处理
    // Step 9: Class-wise NMS processing
    std::vector<std::vector<int>> nms_indices(CLASSES_NUM);
    int total_detections_before_nms = all_bboxes.size();
    int total_detections_after_nms = 0;    
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        // 收集该类别的所有检测
        std::vector<cv::Rect2d> class_bboxes;
        std::vector<float> class_scores;
        std::vector<int> original_indices;        
        for (size_t i = 0; i < all_bboxes.size(); i++) {
            if (all_ids[i] == cls_id) {
                class_bboxes.push_back(all_bboxes[i]);
                class_scores.push_back(all_scores[i]);
                original_indices.push_back(i);
            }
        }        
        if (!class_bboxes.empty()) {
            std::vector<int> class_nms_indices;
            cv::dnn::NMSBoxes(class_bboxes, class_scores, 
                             SCORE_THRESHOLD, NMS_THRESHOLD, class_nms_indices);            
            // 将类别内的索引转换为全局索引
            for (int idx : class_nms_indices) {
                nms_indices[cls_id].push_back(original_indices[idx]);
            }
            total_detections_after_nms += class_nms_indices.size();
        }
    }  
    std::cout << "✓ Detections before NMS: " << total_detections_before_nms << std::endl;
    std::cout << "✓ Detections after NMS: " << total_detections_after_nms << std::endl;
    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 10: 绘制结果
    // Step 10: Draw results
    std::cout << "\033[32m-> Drawing results\033[0m" << std::endl;
    begin_time = std::chrono::system_clock::now(); 
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        for (int global_idx : nms_indices[cls_id]) {
            // 坐标转换回原图
            float x1 = (all_bboxes[global_idx].x - x_shift) / x_scale;
            float y1 = (all_bboxes[global_idx].y - y_shift) / y_scale;
            float x2 = x1 + all_bboxes[global_idx].width / x_scale;
            float y2 = y1 + all_bboxes[global_idx].height / y_scale;
            float score = all_scores[global_idx];       
            // 边界检查
            x1 = std::max(0.0f, std::min((float)img.cols - 1, x1));
            y1 = std::max(0.0f, std::min((float)img.rows - 1, y1));
            x2 = std::max(0.0f, std::min((float)img.cols - 1, x2));
            y2 = std::max(0.0f, std::min((float)img.rows - 1, y2)); 
            // 绘制边界框
            cv::Scalar color = rdk_colors[cls_id % 20];
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, LINE_SIZE);         
            // 绘制标签
            std::string label = object_names[cls_id] + ": " + std::to_string(int(score * 100)) + "%";
            int baseline;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_THICKNESS, &baseline);   
            cv::Point label_pos(x1, y1 - 10 > textSize.height ? y1 - 10 : y1 + textSize.height + 10);
            cv::rectangle(img, label_pos + cv::Point(0, baseline), 
                         label_pos + cv::Point(textSize.width, -textSize.height), color, cv::FILLED);
            cv::putText(img, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 0), FONT_THICKNESS);         
            // 打印检测结果
            std::cout << "(" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ") -> " 
                      << object_names[cls_id] << ": " << std::fixed << std::setprecision(2) << score << std::endl;
        }
    } 
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;
    // Step 11: 保存结果
    // Step 11: Save result
    cv::imwrite(IMG_SAVE_PATH, img);
    std::cout << "\033[32m✓ saved in path: \"" << IMG_SAVE_PATH << "\"\033[0m" << std::endl;
    // Step 12: 资源释放
    // Step 12: Release resources
    std::cout << "\033[32m-> Cleaning up resources\033[0m" << std::endl;    
    // 释放任务句柄
    hbUCPReleaseTask(task_handle);    
    // 释放输入内存
    for (int i = 0; i < input_count; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    // 释放输出内存
    for (int i = 0; i < output_count; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    // 释放模型
    hbDNNRelease(packed_dnn_handle);   
    std::cout << "\033[32m✓ Program completed successfully\033[0m" << std::endl;
    return 0;
}
        我们将CPP文件和CMake文件放到一个文件夹之后在宏定义部分配置好我们的模型路径、测试图片以及类别数量和类别名即可运行以下命令实现推理啦！

mkdir build && cd build
cmake .. && make 
./main
        之后我们便可以在build文件夹下看到推理结果啦！！！
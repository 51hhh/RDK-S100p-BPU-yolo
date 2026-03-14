cmake_minimum_required(VERSION 3.0)
project(ultralytics_yolo)
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for header file search
include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    utils/inc
    inc/
    src/
)

# Add library search path for linker
link_directories(/usr/hobot/lib)

# Find packages (required)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Libraries to link with targets
SET(LINK_libs dnn hbucp gflags fmt OpenMP::OpenMP_CXX Threads::Threads ${OpenCV_LIBS})

# ==================== 公共工具源文件 ====================
set(UTILS_SRC
    utils/src/common_utils.cc
    utils/src/postprocess_utils.cc
    utils/src/preprocess_utils.cc
)

# ==================== 目标1: yolov8_usb_camera ====================
# USB摄像头实时YOLOv8/YOLOv11目标检测
add_executable(yolov8_usb_camera
    src/main.cc
    src/ultralytics_yolo.cc
    ${UTILS_SRC}
)
target_link_libraries(yolov8_usb_camera ${LINK_libs})

# ==================== 目标2: example (可选，用于兼容原有示例) ====================
# 如果 example 目录存在则编译
if(EXISTS "${CMAKE_SOURCE_DIR}/../example/main.cc")
    add_executable(ultralytics_yolo11_example
        ${CMAKE_SOURCE_DIR}/../example/main.cc
        ${CMAKE_SOURCE_DIR}/../example/ultralytics_yolo11.cc
        ${UTILS_SRC}
    )
    target_include_directories(ultralytics_yolo11_example PRIVATE ${CMAKE_SOURCE_DIR}/../example/)
    target_link_libraries(ultralytics_yolo11_example ${LINK_libs})
endif()

# ==================== 安装规则 ====================
install(TARGETS yolov8_usb_camera DESTINATION bin)

# ==================== 打印构建信息 ====================
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "========================================")

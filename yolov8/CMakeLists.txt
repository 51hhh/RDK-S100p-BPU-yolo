cmake_minimum_required(VERSION 3.10)
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
find_package(yaml-cpp REQUIRED)
find_package(realsense2 QUIET)
find_package(rclcpp QUIET)
find_package(sensor_msgs QUIET)
find_package(volleyball_interfaces QUIET)
set(ROS2_LINK_LIBS "")

set(REALSENSE_LINK_LIBS "")
if(TARGET realsense2::realsense2)
    set(REALSENSE_LINK_LIBS realsense2::realsense2)
elseif(realsense2_FOUND AND realsense2_LIBRARY)
    set(REALSENSE_LINK_LIBS ${realsense2_LIBRARY})
endif()
if(rclcpp_FOUND AND sensor_msgs_FOUND AND volleyball_interfaces_FOUND AND
   TARGET sensor_msgs::sensor_msgs__rosidl_typesupport_cpp AND
   TARGET volleyball_interfaces::volleyball_interfaces__rosidl_typesupport_cpp)
    add_compile_definitions(HAVE_VOLLEYBALL_ROS2=1)
    list(APPEND ROS2_LINK_LIBS
         rclcpp::rclcpp
         sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
         volleyball_interfaces::volleyball_interfaces__rosidl_typesupport_cpp)
    message(STATUS "D435 observation and IMU publishers: enabled")
else()
    message(STATUS "D435 observation and IMU publishers: disabled")
endif()

if(NOT REALSENSE_LINK_LIBS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(REALSENSE2 QUIET IMPORTED_TARGET realsense2)
        if(TARGET PkgConfig::REALSENSE2)
            set(REALSENSE_LINK_LIBS PkgConfig::REALSENSE2)
        endif()
    endif()
endif()

if(REALSENSE_LINK_LIBS)
    add_compile_definitions(HAVE_REALSENSE2=1)
    message(STATUS "RealSense RGB-D: enabled")
else()
    message(WARNING "librealsense2 not found: RGB-D capture will be unavailable")
endif()

# Libraries to link with targets
SET(LINK_libs dnn hbucp gflags fmt yaml-cpp OpenMP::OpenMP_CXX Threads::Threads ${OpenCV_LIBS} ${REALSENSE_LINK_LIBS} ${ROS2_LINK_LIBS})

# ==================== 公共工具源文件 ====================
set(UTILS_SRC
    utils/src/common_utils.cc
    utils/src/postprocess_utils.cc
    utils/src/preprocess_utils.cc
    utils/src/rgbd_utils.cc
    utils/src/trajectory_utils.cc
    utils/src/d435_observation_publisher.cc
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

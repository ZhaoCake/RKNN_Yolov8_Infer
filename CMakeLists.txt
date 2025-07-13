cmake_minimum_required(VERSION 3.16)
project(YOLOv8CircleDetection VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 生成compile_commands.json用于clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找依赖包
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)

# 查找ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS 
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
        /opt/onnxruntime/include
        ${CMAKE_PREFIX_PATH}/include/onnxruntime
    NO_DEFAULT_PATH
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
        /usr/local/lib
        /usr/lib
        /opt/onnxruntime/lib
        ${CMAKE_PREFIX_PATH}/lib
    NO_DEFAULT_PATH
)

# 如果上面没找到，再尝试系统默认路径
if(NOT ONNXRUNTIME_INCLUDE_DIR)
    find_path(ONNXRUNTIME_INCLUDE_DIR 
        NAMES onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime
    )
endif()

if(NOT ONNXRUNTIME_LIB)
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(WARNING "ONNX Runtime not found. Please install it manually or run install_onnxruntime.sh")
    message(STATUS "Expected locations:")
    message(STATUS "  Headers: /usr/local/include/onnxruntime/")
    message(STATUS "  Library: /usr/local/lib/libonnxruntime.so")
    set(ONNXRUNTIME_FOUND FALSE)
else()
    message(STATUS "Found ONNX Runtime:")
    message(STATUS "  Headers: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  Library: ${ONNXRUNTIME_LIB}")
    set(ONNXRUNTIME_FOUND TRUE)
endif()

# 包含目录
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

if(ONNXRUNTIME_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    add_definitions(-DHAVE_ONNXRUNTIME)
endif()

# 源文件
set(SOURCES
    src/main.cc
    src/yolov8_detector.cc
)

# 头文件
set(HEADERS
    include/yolov8_detector.h
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

if(ONNXRUNTIME_FOUND)
    target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
endif()

# 设置运行时库路径
if(UNIX AND NOT APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:/usr/local/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# 安装规则
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "  OpenCV Libraries: ${OpenCV_LIBRARIES}")
message(STATUS "  ONNX Runtime Found: ${ONNXRUNTIME_FOUND}")
message(STATUS "")

# 如果没有找到ONNX Runtime，提供安装建议
if(NOT ONNXRUNTIME_FOUND)
    message(STATUS "To install ONNX Runtime, run:")
    message(STATUS "  chmod +x install_onnxruntime.sh")
    message(STATUS "  sudo ./install_onnxruntime.sh")
    message(STATUS "")
endif()

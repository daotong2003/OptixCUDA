cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(OptixSimEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

SET(OPTIX_8_0_PATH $ENV{OPTIX_8_0_PATH} CACHE PATH "Path to OptiX SDK")
if (NOT OPTIX_8_0_PATH)
    message(FATAL_ERROR "OPTIX_8_0_PATH variable not set" )
endif()

# ==========================================
# 1. 收集主机端源文件 (不再包含 .cu 文件)
# ==========================================
set(HOST_SOURCES 
    main.cpp 
    Engine/Core/Context.cpp         
    Engine/Scene/GeometryManager.cpp 
    Engine/Tracer/RayTracer.cpp
# ---- 新增的点云转换器源文件 ----
    Engine/Scene/PointCloudConverter.cpp
    Engine/Scene/SceneManager.cpp
    Engine/Debug/Exporter3D.h
)

# ==========================================
# 2. 强制 NVCC 纯 PTX 编译 (核心修复区)
# ==========================================
set(PTX_SOURCES Ptx/device_programs.cu)

# 告诉 CMake 这是一个独立的对象库，专门用来处理 CUDA 文件
add_library(DevicePrograms OBJECT ${PTX_SOURCES})
# 启用原生 PTX 生成模式，并指定架构 (等同于 -ptx 和 -arch=compute_50)
set_target_properties(DevicePrograms PROPERTIES 
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES 50
)

# 传递必须的头文件路径
target_include_directories(DevicePrograms PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${OPTIX_8_0_PATH}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 指定一个绝对的输出目录
set(PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/Ptx")
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# 创建拦截节点：使用生成器表达式自动获取生成的 .ptx 文件，并复制到你期望的目录
add_custom_target(CompilePTX ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:DevicePrograms> "${PTX_OUTPUT_DIR}/"
    DEPENDS DevicePrograms
    COMMENT "正在将生成的 PTX 文件提取到固定目录..."
)

# ==========================================
# 3. 定义可执行文件
# ==========================================
# 注意：这里只有 HOST_SOURCES，干净利落！
add_executable(OptixSim ${HOST_SOURCES})
# 告诉 CMake：在打包 exe 之前，必须等 CompilePTX 将文件复制完毕
add_dependencies(OptixSim CompilePTX)

# ==========================================
# 4. 路径与库配置
# ==========================================
target_include_directories(OptixSim PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${OPTIX_8_0_PATH}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 终极魔法：将 PTX 输出文件夹的绝对路径，变成一个 C++ 宏传进代码里！
# 这样 C++ 永远能在硬盘上准确找到生成的 PTX 文件。
target_compile_definitions(OptixSim PRIVATE "PTX_DIR=\"${PTX_OUTPUT_DIR}\"")

target_link_libraries(OptixSim PRIVATE CUDA::cudart)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(OptixSimEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

SET(OPTIX_8_0_PATH $ENV{OPTIX_8_0_PATH} CACHE PATH "Path to OptiX SDK")
if (NOT OPTIX_8_0_PATH)
    message(FATAL_ERROR "OPTIX_8_0_PATH variable not set" )
endif()


# 1. 收集主机端源文件 (不再包含 .cu 文件)

set(HOST_SOURCES 
    # main.cpp 
    Engine/Core/Context.cpp         
    Engine/Scene/GeometryManager.cpp 
    Engine/Tracer/RayTracer.cpp

    Engine/Tracer/RayGenerator.cpp

    Engine/Tracer/ImageMethodSolver.cpp   # <--- [新增这一行！核心修复]
# ---- 新增的点云转换器源文件 ----
    Engine/Scene/PointCloudConverter.cpp
    Engine/Scene/SceneManager.cpp
    Engine/Debug/Exporter3D.h
    Engine/Debug/PathExporter.h
    Engine/Tracer/ImageMethodKernel.cu 
)

# 构建静态库供复用
add_library(OptixEngineCore STATIC ${HOST_SOURCES})
target_include_directories(OptixEngineCore PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${OPTIX_8_0_PATH}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(OptixEngineCore PUBLIC CUDA::cudart)

# 2. 强制 NVCC 纯 PTX 编译 (核心修复区)

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
    COMMENT "moving PTX to target document."
)


# 原生 C++ 可执行程序 (你的传统入口)
# 只编译 main.cpp，核心逻辑去链接 OptixEngineCore
add_executable(OptixSim main.cpp)
target_link_libraries(OptixSim PRIVATE OptixEngineCore)
add_dependencies(OptixSim CompilePTX)
target_compile_definitions(OptixSim PRIVATE "PTX_DIR=\"${PTX_OUTPUT_DIR}\"")

# Python 绑定层：Nanobind 模块
# ==========================================
# 确保在运行 CMake 前，当前的 Python 环境已安装 nanobind (pip install nanobind)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
# 打印出 CMake 实际抓到的 Python 路径，防止它阳奉阴违
message(STATUS ">>> [CMake Debug]  Python : ${Python_EXECUTABLE}")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)

    # 打印出获取到的 nanobind 目录
message(STATUS ">>> [CMake Debug]  Nanobind CMake: ${NB_DIR}")

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# 编译胶水代码
nanobind_add_module(optix_backend Bindings/OptixSionnaBinding.cpp)

# 核心链接：让绑定模块能够调用引擎的核心函数
target_link_libraries(optix_backend PRIVATE OptixEngineCore)

# 确保 PTX 在 Python 模块加载前也准备好
add_dependencies(optix_backend CompilePTX)
target_compile_definitions(optix_backend PRIVATE "PTX_DIR=\"${PTX_OUTPUT_DIR}\"")
#include "SceneManager.h"
#include "../Core/CudaError.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

namespace Engine {
	namespace Core {
		SceneManager::SceneManager(OptixDeviceContext context)
			: optixContext(context) {
		}

		SceneManager::~SceneManager() {
			// 严谨的 RAII：生命周期结束时统一释放持久化显存
			if (d_instances) cudaFree(d_instances);
			if (d_temp_buffer) cudaFree(d_temp_buffer);
			if (d_ias_output_buffer) cudaFree(d_ias_output_buffer);
		}

		void SceneManager::addInstance(int32_t instance_id, OptixTraversableHandle gas_handle) {
			// 如果传入了无效句柄，直接拦截
			if (gas_handle == 0) return;

			OptixInstance optixInst = {};

			// 1. 初始化为单位矩阵 (无旋转、无平移，位于原点)
			// OptiX 的 transform 是一个 3x4 的按行优先存储的仿射变换矩阵
			const float identity[12] = {
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f
			};
			std::memcpy(optixInst.transform, identity, sizeof(float) * 12);

			// 2. 设置实例专属属性
			optixInst.instanceId = instance_id;        // 开放给 Shader 读取的 ID (optixGetInstanceId())
			optixInst.sbtOffset = 0;                   // 着色器绑定表偏移量 (后续设置多材质时会用到)
			optixInst.visibilityMask = 255;            // 射线可见性掩码 (默认全可见)
			optixInst.flags = OPTIX_INSTANCE_FLAG_NONE; // 实例标志位
			optixInst.traversableHandle = gas_handle;  // 挂载底层的 GAS 树！

			// 3. 记录到 Host 端内存与映射表中
			instanceIndexMap[instance_id] = hostInstances.size();
			hostInstances.push_back(optixInst);
		}

		void SceneManager::updateTransform(int32_t instance_id, const float transform[12]) {
			auto it = instanceIndexMap.find(instance_id);
			if (it != instanceIndexMap.end()) {
				size_t index = it->second;
				// 直接覆盖该部件的 3x4 矩阵
				std::memcpy(hostInstances[index].transform, transform, sizeof(float) * 12);
			}
			else {
				std::cerr << "[Warning] 试图移动未注册的部件 ID: " << instance_id << std::endl;
			}
		}

		// 核心显存复用逻辑
		void SceneManager::allocateDeviceMemory(size_t instanceCount) {
			// [优化] 只有当实例数量增加，导致显存不够时，才重新分配 d_instances
			static size_t current_allocated_instances = 0;
			if (instanceCount > current_allocated_instances) {
				if (d_instances) cudaFree(d_instances);
				CUDA_CHECK(cudaMalloc(&d_instances, instanceCount * sizeof(OptixInstance)));
				current_allocated_instances = instanceCount;
			}
		}

		void SceneManager::buildIAS() {
			if (hostInstances.empty()) return;

			size_t numInstances = hostInstances.size();

			// 1. 确保显存足够，并将最新的实例矩阵推送到 GPU
			allocateDeviceMemory(numInstances);
			CUDA_CHECK(cudaMemcpy(
				d_instances,
				hostInstances.data(),
				numInstances * sizeof(OptixInstance),
				cudaMemcpyHostToDevice
			));

			// 2. 配置构建输入，告诉 OptiX 这是一颗由 Instance 构成的树
			OptixBuildInput buildInput = {};
			buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			buildInput.instanceArray.instances = (CUdeviceptr)d_instances;
			buildInput.instanceArray.numInstances = static_cast<uint32_t>(numInstances);

			// 3. 配置构建选项
			OptixAccelBuildOptions accelOptions = {};
			// OPTIX_BUILD_FLAG_PREFER_FAST_TRACE: 提示显卡优先优化追踪速度
			// OPTIX_BUILD_FLAG_ALLOW_UPDATE: 如果未来您想用 optixAccelBuild 的 update 模式追求更极端的性能，需加上此标志
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

			// 4. 计算 IAS 构建所需的显存大小
			OptixAccelBufferSizes bufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				optixContext,
				&accelOptions,
				&buildInput,
				1, // numBuildInputs (IAS 总是只有一个 buildInput)
				&bufferSizes
			));

			// 5. 极速缓存管理 (避免每一帧都重新申请显存)
			static size_t current_temp_size = 0;
			static size_t current_output_size = 0;

			if (bufferSizes.tempSizeInBytes > current_temp_size) {
				if (d_temp_buffer) cudaFree(d_temp_buffer);
				CUDA_CHECK(cudaMalloc(&d_temp_buffer, bufferSizes.tempSizeInBytes));
				current_temp_size = bufferSizes.tempSizeInBytes;
			}

			if (bufferSizes.outputSizeInBytes > current_output_size) {
				if (d_ias_output_buffer) cudaFree(d_ias_output_buffer);
				CUDA_CHECK(cudaMalloc(&d_ias_output_buffer, bufferSizes.outputSizeInBytes));
				current_output_size = bufferSizes.outputSizeInBytes;
			}

			// 6. 瞬间重建 IAS
			OPTIX_CHECK(optixAccelBuild(
				optixContext,
				0,                  // 默认 CUDA stream
				&accelOptions,
				&buildInput,
				1,                  // numBuildInputs
				(CUdeviceptr)d_temp_buffer,
				bufferSizes.tempSizeInBytes,
				(CUdeviceptr)d_ias_output_buffer,
				bufferSizes.outputSizeInBytes,
				&iasHandle,
				nullptr,
				0
			));

			// 同步确保矩阵更新到位
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	} // namespace Core
} // namespace Engine
#pragma once
#pragma once

#include <optix.h>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace Engine {
	namespace Core {
		// 类 SceneManager
		// 职责：管理顶层加速结构 (IAS)，控制各个分离的点云部件 (GAS) 在 3D 空间中的位置与旋转
		class SceneManager {
		public:
			SceneManager(OptixDeviceContext context);
			~SceneManager();

			// ==========================================
			// 场景节点构建接口
			// ==========================================

			// 注册一个实例。
			// instance_id: 您的部件 ID (如机械臂大臂 = 1)
			// gas_handle: 上一步从 GeometryManager 获取的底层加速结构句柄
			void addInstance(int32_t instance_id, OptixTraversableHandle gas_handle);

			// ==========================================
			// 核心动画/移动接口
			// ==========================================

			// 更新指定部件的 3x4 仿射变换矩阵
			// transform: 长度为 12 的浮点数组，按行优先排列。如果不调用此函数，默认使用单位矩阵(原点位置)。
			void updateTransform(int32_t instance_id, const float transform[12]);

			// ==========================================
			// IAS 构建与获取
			// ==========================================

			// 构建或更新顶层加速结构 (IAS)。
			// 在您每次调用 updateTransform 移动了物体之后，都需要调用一次此函数来刷新 GPU 端状态。
			void buildIAS();

			// 获取最终的场景总句柄 (将这个句柄传给您的 RayTracer 进行射线追踪)
			OptixTraversableHandle getIasHandle() const { return iasHandle; }

		private:
			OptixDeviceContext optixContext = nullptr;
			OptixTraversableHandle iasHandle = 0;

			// ==========================================
			// 数据与显存管理
			// ==========================================

			// 在 Host 端维护的实例配置数组
			std::vector<OptixInstance> hostInstances;

			// 映射表：方便通过 instance_id 快速找到在 hostInstances 数组中的索引，从而修改其矩阵
			std::unordered_map<int32_t, size_t> instanceIndexMap;

			// GPU 显存指针
			void* d_instances = nullptr;       // 存放 OptixInstance 数组的显存
			void* d_ias_output_buffer = nullptr; // IAS 树形结构的显存
			void* d_temp_buffer = nullptr;       // 构建时的临时显存

			// 内部辅助函数
			void allocateDeviceMemory(size_t instanceCount);
		};
	} // namespace Core
} // namespace Engine
#pragma once
//  接收 PointCloudConverter 输出的复杂网格结构
// 实现严格的按 instance_id 分组构建 GAS
#include <optix.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "MeshTypes.h"

namespace Engine {
	namespace Core {
		class GeometryManager {
		public:

			// ==========================================
			// 显存管理机制 (RAII 防泄漏)
			// ==========================================
			// 【防坑修复】：移至 public 顶部，避免后续作用域推导报错
			// 记录单个 GAS 树相关的显存指针，用于析构时自动释放
			struct GasRecord {
				std::vector<void*> d_vertices_list;
				std::vector<void*> d_indices_list;

				// [Step 3 新增] 记录微观物理映射数组的显存指针
				std::vector<void*> d_pointOffsets_list;
				std::vector<void*> d_pointCounts_list;
				std::vector<void*> d_pointIndices_list;

				void* d_gas_output_buffer = nullptr;
				// [新增] 存储每个网格的平面标签 (CPU 端数据，无需分配显存)
				std::vector<int32_t> plane_label_list;
			};

			// 初始化
			GeometryManager(OptixDeviceContext context);
			~GeometryManager();

			// ==========================================
			// 核心构建接口
			// ==========================================

			// 批量构建场景 GAS
			// 接收 PointCloudConverter 的输出作为输入，实现管线无缝对接
			void buildSceneGAS(const std::unordered_map<int32_t, std::vector<Geometry::TriangleMesh>>& meshesByInstance);

			// ==========================================
			// 句柄查询接口 (为后续构建 IAS / 移动物体提供支持)
			// ==========================================

			// 获取指定 instance_id 的底层加速结构句柄
			OptixTraversableHandle getGasHandle(int32_t instance_id) const;

			// 获取全部已构建的 GAS 字典
			const std::unordered_map<int32_t, OptixTraversableHandle>& getAllGasHandles() const {
				return gasHandles;
			}

			// 获取所有底层 GAS 的显存分配记录，供 SBT 打包使用
			const std::unordered_map<int32_t, GasRecord>& getGasRecords() const {
				return gasRecords;
			}

		private:
			OptixDeviceContext optixContext = nullptr;

			// 存储 instance_id 到对应 OptixTraversableHandle 的映射表
			std::unordered_map<int32_t, OptixTraversableHandle> gasHandles;

			// 存储所有已分配的显存记录
			std::unordered_map<int32_t, GasRecord> gasRecords;

			// ==========================================
			// 内部工作流程
			// ==========================================

			// 封装底层的 optixAccelBuild 逻辑，处理显存申请和构建输入
			OptixTraversableHandle buildSingleGAS(const std::vector<Geometry::TriangleMesh>& meshes, GasRecord& outRecord);
		};
	} // namespace Core
} // namespace Engine
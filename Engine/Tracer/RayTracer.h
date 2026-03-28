#pragma once

#include <optix.h>
#include <string>
#include "../../NimbusCommon/LaunchParams.h"
#include "../Scene/GeometryManager.h"
#include "../../NimbusCommon/SbtData.h"

namespace Engine {
	namespace Tracer {
		// 定义 SBT 记录的基本结构 (OptiX 强制要求的内存对齐格式)
		template <typename T>
		struct SbtRecord {
			alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			T data;
		};
		typedef SbtRecord<int> EmptySbtRecord; // 我们这次不需要传额外数据，用空的占位

		class RayTracer {
		public:
			RayTracer(OptixDeviceContext context, OptixTraversableHandle gasHandle);
			~RayTracer();

#ifdef ENABLE_ENGINE_DEBUG
			void shootRay(float ox, float oy, float oz, float dx, float dy, float dz,
				Engine::Geometry::Point* d_globalCloud = nullptr);
#endif
			// 修改函数签名：追加 geometryManager 参数
			void initPipelineAndSBT(const std::string& ptxPath, const Engine::Core::GeometryManager& geometryManager);

			// SBR 0326 专用的射线发射接口
			// out_hostPath: 这是一个 CPU 端的指针，函数执行完毕后，内部会将显卡跑完的路径数据拷贝到这里
			void shootRaySBR(float ox, float oy, float oz, float dx, float dy, float dz,
				Engine::Tracer::RayPath* out_hostPath,
				Engine::Geometry::Point* d_globalCloud = nullptr);

			// SBR 批量发射接口：一次性把成千上万条射线推给 GPU，并返回它们各自的拓扑账本
			void shootRaysBatchSBR(
				const std::vector<float3>& host_txRays,    // 斐波那契射线束
				float tx_ox, float tx_oy, float tx_oz,     // Tx 发射机坐标
				float rx_ox, float rx_oy, float rx_oz,     // Rx 捕获球坐标
				float rx_radius,                           // Rx 捕获球半径
				std::vector<Engine::Tracer::PathTopology>& out_topologies, // 输出的拓扑账本数组
				Engine::Geometry::Point* d_globalCloud = nullptr);

			// [阶段三] 发射阴影射线，对精确路径进行最终的物理遮挡判决
			void validatePathsOptiX(std::vector<Engine::Tracer::ExactPath>& paths);

		private:
			OptixDeviceContext optixContext;
			OptixTraversableHandle sceneHandle;

			OptixModule module = nullptr;
			OptixPipeline pipeline = nullptr;

			OptixProgramGroup raygenPG = nullptr;
			OptixProgramGroup missPG = nullptr;
			OptixProgramGroup hitgroupPG = nullptr;

			OptixShaderBindingTable sbt = {};

			// [SBR 新增] SBR 输出路径在 GPU 显存上的缓存指针
			Engine::Tracer::RayPath* d_outSbrPath = nullptr;

			// 显存指针，用于传递参数和接收结果
			Engine::LaunchParams* d_params = nullptr;
			int* d_outHitStatus = nullptr;
			float* d_outHitX = nullptr;
			float* d_outHitY = nullptr;
			float* d_outHitZ = nullptr;

			// 微观物理校准输出缓存 (新增)
			float* d_outHitNormalX = nullptr;
			float* d_outHitNormalY = nullptr;
			float* d_outHitNormalZ = nullptr;
			uint8_t* d_outHitMaterial = nullptr;

			// [阶段一新增] 显存指针与容量管理
			float3* d_txRayDirections = nullptr;
			Engine::Tracer::PathTopology* d_outCandidateTopologies = nullptr;
			size_t allocatedRayCapacity = 0; // 记录当前显卡上分配了多少条射线的空间
		};
	} // namespace Tracer
} // namespace Engine
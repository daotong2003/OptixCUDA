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

			// 在 RayTracer 类中
			void shootRay(float ox, float oy, float oz, float dx, float dy, float dz,
				Engine::Geometry::Point* d_globalCloud = nullptr);

			// 修改函数签名：追加 geometryManager 参数
			void initPipelineAndSBT(const std::string& ptxPath, const Engine::Core::GeometryManager& geometryManager);

		private:
			OptixDeviceContext optixContext;
			OptixTraversableHandle sceneHandle;

			OptixModule module = nullptr;
			OptixPipeline pipeline = nullptr;

			OptixProgramGroup raygenPG = nullptr;
			OptixProgramGroup missPG = nullptr;
			OptixProgramGroup hitgroupPG = nullptr;

			OptixShaderBindingTable sbt = {};

			// 显存指针，用于传递参数和接收结果
			Engine::LaunchParams* d_params = nullptr;
			int* d_outHitStatus = nullptr;
			float* d_outHitX = nullptr;
			float* d_outHitY = nullptr;
			float* d_outHitZ = nullptr;

			// ==================== [Step 5 补充声明] ====================
			// 微观物理校准输出缓存 (新增)
			float* d_outHitNormalX = nullptr;
			float* d_outHitNormalY = nullptr;
			float* d_outHitNormalZ = nullptr;
			uint8_t* d_outHitMaterial = nullptr;
			// =========================================================
		};
	} // namespace Tracer
} // namespace Engine
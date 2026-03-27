#pragma once
#include <optix.h>
#include <cstdint>
#include "Engine/Scene/MeshTypes.h"
#include "Engine/Tracer/SbrTypes.h"

namespace Engine {
	namespace Geometry {
		struct Point;
	}
}

namespace Engine {
	struct LaunchParams {
		OptixTraversableHandle handle;

		// ==================== [系统与物理参数] ====================
		float tmax;
		int maxBounceDepth;
		Geometry::Point* globalPointCloud;

		// ==================== [Tx 发射机配置 (批量)] ====================
		float rayOrigin_x, rayOrigin_y, rayOrigin_z; // 发射机坐标

		// 以前是单一方向，现在改为接收 CPU 算好的斐波那契射线束指针
		float3* txRayDirections;
		unsigned int numRays; // 总发射射线数量

		// 兼容旧版：保留单一射线方向，用于单步调试
		float rayDirection_x, rayDirection_y, rayDirection_z;

		// ==================== [Rx 接收机配置 (虚拟捕获球)] ====================
		float rxPosition_x, rxPosition_y, rxPosition_z;
		float rxRadius; // 虚拟捕获球的半径

		// ==================== [输出缓存] ====================
		// 1. 旧版 LOS 输出缓存 (保留)
		int* outHitStatus;
		float* outHitPosition_x; float* outHitPosition_y; float* outHitPosition_z;
		float* outHitNormal_x; float* outHitNormal_y; float* outHitNormal_z;
		uint8_t* outHitMaterial;

		// 2. 旧版 SBR 单线物理缓存 (保留)
		Tracer::RayPath* outSbrPath;

		// 3. [阶段一新增] 粗搜拓扑候选者大数组，大小将等于 numRays
		Tracer::PathTopology* outCandidateTopologies;

		// [阶段三专属] 遮挡验证 (Shadow Ray Validation)
		// =========================================================
		Engine::Tracer::ExactPath* validationPaths; // 需要验证的精确路径数组
		int numValidationPaths;                     // 路径总数
	};
} // namespace Engine
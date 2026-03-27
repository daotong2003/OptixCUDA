#pragma once
#include "SbrTypes.h"
#include "Engine/Scene/MeshTypes.h" // 引入点云 Geometry::Point 定义
#include <vector>
#include <unordered_map>

namespace Engine {
	namespace Tracer {
		class ImageMethodSolver {
		public:
			// 核心算法：(你之前写的 solvePath 保留)
			static ExactPath solvePath(const float3& tx, const float3& rx, const std::vector<PlaneEquation>& planes);

			// ==================== [阶段二数据基建] ====================
			// 遍历全局点云，抹平噪点，为每个 label 提取出绝对平滑的平面方程字典
			static std::unordered_map<int32_t, PlaneEquation> buildPlaneMapFromCloud(
				const std::vector<Engine::Geometry::Point>& globalCloud);

			// 全自动 GPU 批量解析接口
			static std::vector<ExactPath> solvePathsGPU(
				const std::vector<PathTopology>& uniqueTopologies,
				const std::unordered_map<int32_t, PlaneEquation>& planeMap,
				const float3& tx, const float3& rx);

		private:
			static float3 mirrorPoint(const float3& pt, const PlaneEquation& plane);
			static float3 intersectLinePlane(const float3& p1, const float3& p2, const PlaneEquation& plane, bool& out_intersected);
		};
	} // namespace Tracer
} // namespace Engine
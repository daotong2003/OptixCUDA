#pragma once
#include "MeshTypes.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace Engine {
	namespace Geometry {
		class PointCloudConverter {
		public:
			PointCloudConverter(float config_grid_size = 0.055f);
			~PointCloudConverter() = default;

			bool loadFromBinaryPLY(const std::string& filepath, std::vector<Point>& outPoints);

			// 核心转换管线接口保持不变
			std::unordered_map<int32_t, std::vector<TriangleMesh>> convertToMeshes(const std::vector<Point>& rawPoints);

		private:
			float m_gridSize;

			struct PlaneBasis {
				float3 origin;
				float3 normal;
				float3 axisU;
				float3 axisV;
			};

			// --- [优化 1] 改为返回“全局索引”的二维分组 ---
			std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<uint32_t>>>
				groupPointIndices(const std::vector<Point>& rawPoints);

			// --- [优化 2] 传入全局点云池与当前平面的索引列表 ---
			TriangleMesh triangulatePlaneGreedy(const std::vector<Point>& globalPoints, const std::vector<uint32_t>& planeIndices);

			// --- [新增] 兜底机制：将零散残余点直接 1:1 三角化 ---
			TriangleMesh triangulateResidualPoints(const std::vector<Point>& globalPoints, const std::vector<uint32_t>& residualIndices);

			// --- 辅助函数同步更新 ---
			PlaneBasis computePlaneBasis(const std::vector<Point>& globalPoints, const std::vector<uint32_t>& indices);
			float3 calculate3DPos(const PlaneBasis& basis, float u, float v);
		};
	} // namespace Geometry
} // namespace Engine
// 点云转化为 OptiX 能够处理的几何面片
#include "PointCloudConverter.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector_functions.h>

// 辅助向量数学
inline float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float3 normalize(float3 v) {
	float len = std::sqrt(dot(v, v));
	return len > 0 ? make_float3(v.x / len, v.y / len, v.z / len) : make_float3(0.0f, 0.0f, 0.0f);
}

inline float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator*(float a, float3 b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

inline float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
namespace Engine {
	namespace {
		// [调试专用] 面片映射详情全量打印函数 (采用单行紧凑格式防刷屏)
		inline void debugPrintPatchMapping(int patchIndex, int w, int h, size_t totalPts, size_t tri0Pts, size_t tri1Pts) {
			std::cout << "[Debug Patch " << patchIndex << "] "
				<< "Grid " << w << "x" << h << " | "
				<< "Total Points: " << totalPts << " -> "
				<< "Tri0: " << tri0Pts << " + Tri1: " << tri1Pts << " "
				<< (totalPts == (tri0Pts + tri1Pts) ? "[PASS]" : "[FAIL!]")
				<< "\n";
		}
	}

	namespace Geometry {
		PointCloudConverter::PointCloudConverter(float config_grid_size)
			: m_gridSize(config_grid_size) {
		}

		bool PointCloudConverter::loadFromBinaryPLY(const std::string& filepath, std::vector<Point>& outPoints) {
			std::ifstream file(filepath, std::ios::binary);
			if (!file.is_open()) {
				std::cerr << "Failed to open point cloud file: " << filepath << std::endl;
				return false;
			}

			//  跳过 ASCII 文件头，解析总点数
			std::string line;
			size_t vertexCount = 0;
			while (std::getline(file, line)) {
				if (line.find("element vertex") != std::string::npos) {
					sscanf(line.c_str(), "element vertex %zu", &vertexCount);
				}
				if (line == "end_header" || line == "end_header\r") break;
			}

			if (vertexCount == 0) return false;

			// --- 在这里加一句打印 ---
			// std::cout << "[Debug] Parsed vertex count: " << vertexCount << std::endl;
			// std::cout << "[Debug] Expected memory: " << (vertexCount * sizeof(Point)) / (1024 * 1024) << " MB" << std::endl;

			outPoints.resize(vertexCount);

			// 2. 内存 Zero-Copy 映射：利用一字节对齐的威力，光速载入数千万点云
			outPoints.resize(vertexCount);
			file.read(reinterpret_cast<char*>(outPoints.data()), vertexCount * sizeof(Point));
			return true;
		}

		std::unordered_map<int32_t, std::vector<TriangleMesh>>
			PointCloudConverter::convertToMeshes(const std::vector<Point>& rawPoints) {
			auto groupedIndices = groupPointIndices(rawPoints);
			std::unordered_map<int32_t, std::vector<TriangleMesh>> resultMeshes;

			std::vector<uint32_t> allResidualIndices;

			for (const auto& instancePair : groupedIndices) {
				int32_t instId = instancePair.first;
				for (const auto& labelPair : instancePair.second) {

					// 这里的 labelPair.first 就是最纯正的平面标签
					int32_t planeLabel = labelPair.first;
					const std::vector<uint32_t>& planeIndices = labelPair.second;

					if (planeIndices.size() < 10) {
						allResidualIndices.insert(allResidualIndices.end(), planeIndices.begin(), planeIndices.end());
						continue;
					}

					TriangleMesh mesh = triangulatePlaneGreedy(rawPoints, planeIndices);
					mesh.instance_id = instId;
					mesh.material_id = rawPoints[planeIndices[0]].material;

					// [新增] 严格赋权！这个网格生成的所有三角形，绝对只属于这个 Label
					mesh.plane_label = planeLabel;
					resultMeshes[instId].push_back(mesh);
				}
			}

			if (!allResidualIndices.empty()) {
				std::cout << "[Converter] 触发防漏底机制，正在为 " << allResidualIndices.size()
					<< " 个离群碎片点构建微网格..." << std::endl;
				TriangleMesh residualMesh = triangulateResidualPoints(rawPoints, allResidualIndices);
				resultMeshes[9999].push_back(residualMesh); // 放入专属的 Instance ID 9999
			}

			return resultMeshes;
		}

		std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<uint32_t>>>
			PointCloudConverter::groupPointIndices(const std::vector<Point>& rawPoints) {
			std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<uint32_t>>> groups;
			for (uint32_t i = 0; i < rawPoints.size(); ++i) {
				const auto& pt = rawPoints[i];
				groups[pt.instance_id][pt.label].push_back(i);
			}
			return groups;
		}

		PointCloudConverter::PlaneBasis PointCloudConverter::computePlaneBasis(const std::vector<Point>& globalPoints, const std::vector<uint32_t>& indices) {
			PlaneBasis basis;

			float3 center = make_float3(0, 0, 0);
			float3 avgNormal = make_float3(0, 0, 0);
			for (uint32_t idx : indices) {
				center = center + globalPoints[idx].getPos();
				avgNormal = avgNormal + globalPoints[idx].getNormal();
			}
			float invN = 1.0f / indices.size();
			basis.origin = make_float3(center.x * invN, center.y * invN, center.z * invN);
			basis.normal = normalize(avgNormal);

			// 绝不使用 PCA，保证所有的墙壁包围盒都是绝对横平竖直的，消除倾斜！
			float3 worldUp = make_float3(0.0f, 0.0f, 1.0f);

			if (std::abs(basis.normal.z) > 0.95f) {
				// 如果是地板或天花板，U轴指向 X
				basis.axisU = make_float3(1.0f, 0.0f, 0.0f);
				basis.axisV = make_float3(0.0f, 1.0f, 0.0f);
			}
			else {
				// 如果是墙壁，强制 U 轴平行于真实地面
				basis.axisU = normalize(cross(worldUp, basis.normal));
				basis.axisV = normalize(cross(basis.normal, basis.axisU));
			}
			return basis;
		}

		float3 PointCloudConverter::calculate3DPos(const PlaneBasis& basis, float u, float v) {
			return basis.origin + (u * basis.axisU) + (v * basis.axisV);
		}

		TriangleMesh PointCloudConverter::triangulatePlaneGreedy(const std::vector<Point>& globalPoints, const std::vector<uint32_t>& planeIndices) {
			TriangleMesh mesh;
			PlaneBasis basis = computePlaneBasis(globalPoints, planeIndices);

			float minU = 1e9f, maxU = -1e9f, minV = 1e9f, maxV = -1e9f;
			for (uint32_t globalIdx : planeIndices) {
				float3 localP = globalPoints[globalIdx].getPos() - basis.origin;
				float u = dot(localP, basis.axisU);
				float v = dot(localP, basis.axisV);
				minU = std::min(minU, u); maxU = std::max(maxU, u);
				minV = std::min(minV, v); maxV = std::max(maxV, v);
			}

			int cols = static_cast<int>(std::ceil((maxU - minU) / m_gridSize)) + 1;
			int rows = static_cast<int>(std::ceil((maxV - minV) / m_gridSize)) + 1;

			// 将点云分配到网格中
			std::vector<std::vector<uint32_t>> gridPoints(cols * rows);
			for (uint32_t globalIdx : planeIndices) {
				float3 localP = globalPoints[globalIdx].getPos() - basis.origin;
				int i = static_cast<int>((dot(localP, basis.axisU) - minU) / m_gridSize);
				int j = static_cast<int>((dot(localP, basis.axisV) - minV) / m_gridSize);
				if (i >= 0 && i < cols && j >= 0 && j < rows) {
					gridPoints[j * cols + i].push_back(globalIdx);
				}
			}

			// 【核心修复】彻底抛弃贪心合并与孔洞填补，采用“独立精确包围盒”策略
			for (int j = 0; j < rows; ++j) {
				for (int i = 0; i < cols; ++i) {
					int cellIndex = j * cols + i;
					const auto& cellPts = gridPoints[cellIndex];

					// 1. 遇到空白区域（真实缝隙、窗户）直接跳过，绝不填补
					if (cellPts.empty()) continue;

					//// 2. 计算该网格内点云的【绝对精确】局部边界，实现 100% 紧致贴合，绝不越界
					//float cellMinU = 1e9f, cellMaxU = -1e9f;
					//float cellMinV = 1e9f, cellMaxV = -1e9f;
					//for (uint32_t globalIdx : cellPts) {
					//	float3 localP = globalPoints[globalIdx].getPos() - basis.origin;
					//	float u = dot(localP, basis.axisU);
					//	float v = dot(localP, basis.axisV);
					//	cellMinU = std::min(cellMinU, u); cellMaxU = std::max(cellMaxU, u);
					//	cellMinV = std::min(cellMinV, v); cellMaxV = std::max(cellMaxV, v);
					//}

					// 2. [修复] 强制使用理论网格边界，确保相邻三角形 100% 共边，消灭缝隙！
					// 取消局部收缩，直接用循环索引 (i, j) 乘以 grid_size 推算绝对坐标
					float cellMinU = minU + i * m_gridSize;
					// 保护最边缘的网格不要超出真实点云的最大边界
					float cellMaxU = std::min(minU + (i + 1) * m_gridSize, maxU);

					float cellMinV = minV + j * m_gridSize;
					float cellMaxV = std::min(minV + (j + 1) * m_gridSize, maxV);
					// 极小值保护（防止点完全共线导致三角形面积为0）
					float pad = 0.1f;
					cellMinU -= pad; cellMaxU += pad;
					cellMinV -= pad; cellMaxV += pad;

					// 3. 生成紧致的 4 个顶点
					uint32_t vIdx = static_cast<uint32_t>(mesh.vertices.size());
					mesh.vertices.push_back(calculate3DPos(basis, cellMinU, cellMinV));
					mesh.vertices.push_back(calculate3DPos(basis, cellMaxU, cellMinV));
					mesh.vertices.push_back(calculate3DPos(basis, cellMaxU, cellMaxV));
					mesh.vertices.push_back(calculate3DPos(basis, cellMinU, cellMaxV));

					mesh.indices.push_back({ vIdx, vIdx + 1, vIdx + 2 });
					mesh.indices.push_back({ vIdx, vIdx + 2, vIdx + 3 });

					// 4. 将点分配给两个三角形
					std::vector<uint32_t> tri0_indices;
					std::vector<uint32_t> tri1_indices;
					float du = cellMaxU - cellMinU;
					float dv = cellMaxV - cellMinV;

					for (uint32_t globalIdx : cellPts) {
						float3 localP = globalPoints[globalIdx].getPos() - basis.origin;
						float u = dot(localP, basis.axisU);
						float v = dot(localP, basis.axisV);
						float norm_u = (du > 1e-6f) ? (u - cellMinU) / du : 0.0f;
						float norm_v = (dv > 1e-6f) ? (v - cellMinV) / dv : 0.0f;

						if (norm_u > norm_v) tri0_indices.push_back(globalIdx);
						else tri1_indices.push_back(globalIdx);
					}

					// 5. 【防零点机制】如果对角线划分恰好导致某个三角形内没有点
					// 从隔壁三角形借用一个真实点作为代理，消灭 Original CPU: 0
					if (tri0_indices.empty() && !tri1_indices.empty()) tri0_indices.push_back(tri1_indices.front());
					if (tri1_indices.empty() && !tri0_indices.empty()) tri1_indices.push_back(tri0_indices.front());

					mesh.pointOffsets.push_back(static_cast<uint32_t>(mesh.pointIndices.size()));
					mesh.pointCounts.push_back(static_cast<uint32_t>(tri0_indices.size()));
					mesh.pointIndices.insert(mesh.pointIndices.end(), tri0_indices.begin(), tri0_indices.end());

					mesh.pointOffsets.push_back(static_cast<uint32_t>(mesh.pointIndices.size()));
					mesh.pointCounts.push_back(static_cast<uint32_t>(tri1_indices.size()));
					mesh.pointIndices.insert(mesh.pointIndices.end(), tri1_indices.begin(), tri1_indices.end());
				}
			}
			return mesh;
		}

		// 1:1 独立成面，确保 100% 物理还原 ---
		TriangleMesh PointCloudConverter::triangulateResidualPoints(const std::vector<Point>& globalPoints, const std::vector<uint32_t>& residualIndices) {
			TriangleMesh mesh;
			mesh.instance_id = 9999;
			mesh.material_id = 0;
			float proxySize = m_gridSize * 0.1f; // 以降采样网格的一半作为微平面半径

			for (uint32_t globalIdx : residualIndices) {
				const auto& pt = globalPoints[globalIdx];
				float3 p = pt.getPos();
				float3 n = pt.getNormal();

				float3 up = (std::abs(n.y) > 0.99f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
				float3 t = normalize(cross(up, n));
				float3 b = cross(n, t);

				uint32_t vIdx = static_cast<uint32_t>(mesh.vertices.size());
				mesh.vertices.push_back(p + (proxySize * t));
				mesh.vertices.push_back(p - (proxySize * t) + (proxySize * b));
				mesh.vertices.push_back(p - (proxySize * t) - (proxySize * b));

				mesh.indices.push_back({ vIdx, vIdx + 1, vIdx + 2 });

				mesh.pointOffsets.push_back(static_cast<uint32_t>(mesh.pointIndices.size()));
				mesh.pointCounts.push_back(1);
				mesh.pointIndices.push_back(globalIdx); // 直接映射唯一的全局索引
			}
			return mesh;
		}
	}
} // namespace Geometry
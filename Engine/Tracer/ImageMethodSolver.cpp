#include "ImageMethodSolver.h"
#include <cmath>

// [新增这行] 引入 CUDA 的 host 端向量构造函数
#include <vector_functions.h>
#include <cuda_runtime.h>
#include "../Core/CudaError.h"
#include "ImageMethodKernel.h"
#include <algorithm> // 引入 std::sort

namespace Engine {
	namespace Tracer {
		float3 ImageMethodSolver::mirrorPoint(const float3& pt, const PlaneEquation& plane) {
			// 距离公式：dist = Ax + By + Cz + D
			float dist = plane.normal.x * pt.x + plane.normal.y * pt.y + plane.normal.z * pt.z + plane.d;
			// 镜像点：P' = P - 2 * dist * N
			return make_float3(
				pt.x - 2.0f * dist * plane.normal.x,
				pt.y - 2.0f * dist * plane.normal.y,
				pt.z - 2.0f * dist * plane.normal.z
			);
		}

		float3 ImageMethodSolver::intersectLinePlane(const float3& p1, const float3& p2, const PlaneEquation& plane, bool& out_intersected) {
			float3 dir = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
			float dotDirNormal = dir.x * plane.normal.x + dir.y * plane.normal.y + dir.z * plane.normal.z;

			// 如果射线和平面几乎平行，则无交点
			if (std::abs(dotDirNormal) < 1e-6f) {
				out_intersected = false;
				return make_float3(0, 0, 0);
			}

			float distP1 = plane.normal.x * p1.x + plane.normal.y * p1.y + plane.normal.z * p1.z + plane.d;
			float t = -distP1 / dotDirNormal;

			out_intersected = true;
			return make_float3(p1.x + t * dir.x, p1.y + t * dir.y, p1.z + t * dir.z);
		}

		ExactPath ImageMethodSolver::solvePath(const float3& tx, const float3& rx, const std::vector<PlaneEquation>& planes) {
			ExactPath path;
			path.isValid = false;
			path.vertexCount = 0;

			int numBounces = static_cast<int>(planes.size());
			if (numBounces > MAX_BOUNCE_DEPTH) return path;

			// 如果是视距直射 (LOS)
			if (numBounces == 0) {
				path.vertices[0] = tx;
				path.vertices[1] = rx;
				path.vertexCount = 2;
				path.isValid = true;
				return path;
			}

			// 1. 正向求镜像 (存储每一步的虚拟源)
			std::vector<float3> images(numBounces);
			images[0] = mirrorPoint(tx, planes[0]);
			for (int i = 1; i < numBounces; ++i) {
				images[i] = mirrorPoint(images[i - 1], planes[i]);
			}

			// 2. 反向求交点
			std::vector<float3> hitPoints(numBounces);
			float3 currentTarget = rx;

			for (int i = numBounces - 1; i >= 0; --i) {
				// 连线的起点永远是【当前平面的虚拟镜像源】
				float3 rayOrigin = images[i];
				bool intersected = false;
				float3 p = intersectLinePlane(rayOrigin, currentTarget, planes[i], intersected);

				if (!intersected) return path; // 数学死胡同，路径失效
				hitPoints[i] = p;
				currentTarget = p; // 下一个平面的目标点变成当前交点
			}

			// 3. 组装最终的折线顶点
			path.vertices[0] = tx;
			for (int i = 0; i < numBounces; ++i) {
				path.vertices[i + 1] = hitPoints[i];
			}
			path.vertices[numBounces + 1] = rx;
			path.vertexCount = numBounces + 2;
			path.isValid = true;

			return path;
		}

		// ==================== [提取全局平面情报库] ====================
		std::unordered_map<int32_t, PlaneEquation> ImageMethodSolver::buildPlaneMapFromCloud(
			const std::vector<Engine::Geometry::Point>& globalCloud)
		{
			std::unordered_map<int32_t, PlaneEquation> planeMap;

			// 用于累加数据的临时字典
			std::unordered_map<int32_t, float3> normalSum;
			std::unordered_map<int32_t, float3> centroidSum;
			std::unordered_map<int32_t, int> countMap;

			// 1. 第一次遍历：按 label 累加坐标和法线
			for (const auto& pt : globalCloud) {
				int32_t label = pt.label;
				countMap[label]++;

				normalSum[label].x += pt.nx;
				normalSum[label].y += pt.ny;
				normalSum[label].z += pt.nz;

				centroidSum[label].x += pt.x;
				centroidSum[label].y += pt.y;
				centroidSum[label].z += pt.z;
			}

			// 2. 结算提取：求均值，计算常数 D
			for (const auto& pair : countMap) {
				int32_t label = pair.first;
				int n = pair.second;

				// 计算宏观平均法线 (A, B, C)
				float3 avgN = make_float3(normalSum[label].x / n, normalSum[label].y / n, normalSum[label].z / n);

				// 严格归一化法向量
				float len = std::sqrt(avgN.x * avgN.x + avgN.y * avgN.y + avgN.z * avgN.z);
				if (len > 1e-6f) {
					avgN.x /= len; avgN.y /= len; avgN.z /= len;
				}

				// 计算质心坐标 P(x, y, z)
				float3 avgP = make_float3(centroidSum[label].x / n, centroidSum[label].y / n, centroidSum[label].z / n);

				// 代数几何公式求常数项：D = -(A*x + B*y + C*z)
				float d = -(avgN.x * avgP.x + avgN.y * avgP.y + avgN.z * avgP.z);

				PlaneEquation eq;
				eq.normal = avgN;
				eq.d = d;
				planeMap[label] = eq;
			}

			return planeMap;
		}

		std::vector<ExactPath> ImageMethodSolver::solvePathsGPU(
			const std::vector<PathTopology>& uniqueTopologies,
			const std::unordered_map<int32_t, PlaneEquation>& planeMap,
			const float3& tx, const float3& rx)
		{
			std::vector<ExactPath> results(uniqueTopologies.size());
			if (uniqueTopologies.empty()) return results;

			// 1. 将平面字典展平并排序 (极其重要，因为 GPU 要用二分查找)
			std::vector<PlaneDictEntry> dictArray;
			dictArray.reserve(planeMap.size());
			for (const auto& pair : planeMap) dictArray.push_back({ pair.first, pair.second });
			std::sort(dictArray.begin(), dictArray.end(), [](const PlaneDictEntry& a, const PlaneDictEntry& b) {
				return a.label < b.label;
				});

			// 2. 申请显存
			PathTopology* d_topologies = nullptr;
			PlaneDictEntry* d_plane_dict = nullptr;
			ExactPath* d_out_paths = nullptr;

			size_t topoBytes = uniqueTopologies.size() * sizeof(PathTopology);
			size_t dictBytes = dictArray.size() * sizeof(PlaneDictEntry);
			size_t exactBytes = results.size() * sizeof(ExactPath);

			cudaMalloc((void**)&d_topologies, topoBytes);
			cudaMalloc((void**)&d_plane_dict, dictBytes);
			cudaMalloc((void**)&d_out_paths, exactBytes);

			// 3. 搬运数据进 GPU
			cudaMemcpy(d_topologies, uniqueTopologies.data(), topoBytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_plane_dict, dictArray.data(), dictBytes, cudaMemcpyHostToDevice);

			// 4. 发射核弹！(调用纯 CUDA Kernel)
			launchImageMethodKernel(
				d_topologies, static_cast<int>(uniqueTopologies.size()),
				d_plane_dict, static_cast<int>(dictArray.size()),
				tx, rx, d_out_paths
			);
			cudaDeviceSynchronize(); // 等待所有流处理器算完

			// 5. 将算好的精确折线拉回 CPU
			cudaMemcpy(results.data(), d_out_paths, exactBytes, cudaMemcpyDeviceToHost);

			// 6. 打扫战场
			cudaFree(d_topologies); cudaFree(d_plane_dict); cudaFree(d_out_paths);

			return results;
		}

		// [TDD 新增] 局部 2D 字典构建测试
		std::unordered_map<int32_t, LocalPlaneDictEntry> ImageMethodSolver::buildLocalPlaneMapFromCloud_TEST(
			const std::vector<Engine::Geometry::Point>& globalCloud)
		{
			std::unordered_map<int32_t, LocalPlaneDictEntry> planeMap;
			std::unordered_map<int32_t, std::vector<uint32_t>> groupedPoints;

			// 1. 按 label 分组收集点索引
			for (uint32_t i = 0; i < globalCloud.size(); ++i) {
				groupedPoints[globalCloud[i].label].push_back(i);
			}

			// 2. 遍历每个平面分组
			for (const auto& pair : groupedPoints) {
				int32_t label = pair.first;
				const auto& indices = pair.second;
				if (indices.empty()) continue;

				int32_t instance_id = globalCloud[indices[0]].instance_id;

				// 计算中心点和平均法线
				float3 center = make_float3(0, 0, 0);
				float3 avgNormal = make_float3(0, 0, 0);
				for (uint32_t idx : indices) {
					center.x += globalCloud[idx].x; center.y += globalCloud[idx].y; center.z += globalCloud[idx].z;
					avgNormal.x += globalCloud[idx].nx; avgNormal.y += globalCloud[idx].ny; avgNormal.z += globalCloud[idx].nz;
				}
				float invN = 1.0f / indices.size();
				center.x *= invN; center.y *= invN; center.z *= invN;

				float len = std::sqrt(avgNormal.x * avgNormal.x + avgNormal.y * avgNormal.y + avgNormal.z * avgNormal.z);
				if (len > 1e-6f) { avgNormal.x /= len; avgNormal.y /= len; avgNormal.z /= len; }
				float d = -(avgNormal.x * center.x + avgNormal.y * center.y + avgNormal.z * center.z);

				// 提取局部基底 (与 PointCloudConverter 保持绝对一致，保证防倾斜)
				float3 worldUp = make_float3(0.0f, 0.0f, 1.0f);
				float3 axisU, axisV;
				if (std::abs(avgNormal.z) > 0.95f) {
					axisU = make_float3(1.0f, 0.0f, 0.0f);
					axisV = make_float3(0.0f, 1.0f, 0.0f);
				}
				else {
					float3 crossU = make_float3(worldUp.y * avgNormal.z - worldUp.z * avgNormal.y,
						worldUp.z * avgNormal.x - worldUp.x * avgNormal.z,
						worldUp.x * avgNormal.y - worldUp.y * avgNormal.x);
					float lenU = std::sqrt(crossU.x * crossU.x + crossU.y * crossU.y + crossU.z * crossU.z);
					axisU = make_float3(crossU.x / lenU, crossU.y / lenU, crossU.z / lenU);

					float3 crossV = make_float3(avgNormal.y * axisU.z - avgNormal.z * axisU.y,
						avgNormal.z * axisU.x - avgNormal.x * axisU.z,
						avgNormal.x * axisU.y - avgNormal.y * axisU.x);
					float lenV = std::sqrt(crossV.x * crossV.x + crossV.y * crossV.y + crossV.z * crossV.z);
					axisV = make_float3(crossV.x / lenV, crossV.y / lenV, crossV.z / lenV);
				}

				// 3. 计算 2D 投影边界
				float min_u = 1e9f, max_u = -1e9f, min_v = 1e9f, max_v = -1e9f;
				for (uint32_t idx : indices) {
					float3 localP = make_float3(globalCloud[idx].x - center.x,
						globalCloud[idx].y - center.y,
						globalCloud[idx].z - center.z);
					float u = localP.x * axisU.x + localP.y * axisU.y + localP.z * axisU.z;
					float v = localP.x * axisV.x + localP.y * axisV.y + localP.z * axisV.z;
					if (u < min_u) min_u = u; if (u > max_u) max_u = u;
					if (v < min_v) min_v = v; if (v > max_v) max_v = v;
				}

				// 4. 自适应网格计算
				float area = (max_u - min_u) * (max_v - min_v);
				float avg_area = area / indices.size();
				float grid_size = std::sqrt(avg_area) * 1.5f;
				if (grid_size < 0.02f) grid_size = 0.055f; // 兜底极小网格保护

				int cols = static_cast<int>(std::ceil((max_u - min_u) / grid_size)) + 1;
				int rows = static_cast<int>(std::ceil((max_v - min_v) / grid_size)) + 1;

				// ==========================================================
								// 5. [TDD Step 2 新增] 分配 CPU 掩膜，执行 3x3 膨胀
								// ==========================================================
				int total_cells = cols * rows;
				// 临时在 CPU 上分配内存并全部初始化为 false
				bool* host_bitmap = new bool[total_cells]();

				for (uint32_t idx : indices) {
					float3 localP = make_float3(globalCloud[idx].x - center.x,
						globalCloud[idx].y - center.y,
						globalCloud[idx].z - center.z);
					float u = localP.x * axisU.x + localP.y * axisU.y + localP.z * axisU.z;
					float v = localP.x * axisV.x + localP.y * axisV.y + localP.z * axisV.z;

					int u_idx = static_cast<int>((u - min_u) / grid_size);
					int v_idx = static_cast<int>((v - min_v) / grid_size);

					// 核心容差机制：3x3 膨胀 (Dilation) 填补微观缝隙
					for (int dv = -1; dv <= 1; ++dv) {
						for (int du = -1; du <= 1; ++du) {
							int nu = u_idx + du;
							int nv = v_idx + dv;
							// 边界保护
							if (nu >= 0 && nu < cols && nv >= 0 && nv < rows) {
								host_bitmap[nv * cols + nu] = true;
							}
						}
					}
				}
				// ==========================================================
				// 6. [TDD Step 3 新增] 将掩膜上传至 GPU 显存
				// ==========================================================
				bool* d_bitmap = nullptr;
				CUDA_CHECK(cudaMalloc((void**)&d_bitmap, total_cells * sizeof(bool)));
				CUDA_CHECK(cudaMemcpy(d_bitmap, host_bitmap, total_cells * sizeof(bool), cudaMemcpyHostToDevice));

				delete[] host_bitmap; // 必须释放 CPU 临时内存防止泄漏！

				// 【修复 2：只保留一个干净的 entry 组装】
				LocalPlaneDictEntry entry;
				entry.instance_id = instance_id;
				entry.label = label;
				entry.eq.normal = avgNormal;
				entry.eq.d = d;
				entry.local_origin = center;
				entry.axisU = axisU;
				entry.axisV = axisV;
				entry.min_u = min_u; entry.max_u = max_u;
				entry.min_v = min_v; entry.max_v = max_v;
				entry.grid_size = grid_size;
				entry.cols = cols;
				entry.rows = rows;

				// 【核心修正】：指向刚分配的 GPU 显存 d_bitmap
				entry.d_occupancy_bitmap = d_bitmap;

				planeMap[label] = entry;
			}
			return planeMap;
		}
		// [TDD Step 3 新增] 新版 GPU 解析器实现
		std::vector<ExactPath> ImageMethodSolver::solvePathsGPU_TEST(
			const std::vector<PathTopology>& uniqueTopologies,
			const std::unordered_map<int32_t, LocalPlaneDictEntry>& planeMap,
			const float3& tx, const float3& rx)
		{
			std::vector<ExactPath> results(uniqueTopologies.size());
			if (uniqueTopologies.empty()) return results;

			// 1. 将包含位图指针的局部平面字典展平并排序
			std::vector<LocalPlaneDictEntry> dictArray;
			dictArray.reserve(planeMap.size());
			for (const auto& pair : planeMap) dictArray.push_back(pair.second);
			std::sort(dictArray.begin(), dictArray.end(), [](const LocalPlaneDictEntry& a, const LocalPlaneDictEntry& b) {
				return a.label < b.label;
				});

			// 2. 申请显存 (注意这里类型变为了 LocalPlaneDictEntry)
			PathTopology* d_topologies = nullptr;
			LocalPlaneDictEntry* d_plane_dict = nullptr;
			ExactPath* d_out_paths = nullptr;

			size_t topoBytes = uniqueTopologies.size() * sizeof(PathTopology);
			size_t dictBytes = dictArray.size() * sizeof(LocalPlaneDictEntry);
			size_t exactBytes = results.size() * sizeof(ExactPath);

			cudaMalloc((void**)&d_topologies, topoBytes);
			cudaMalloc((void**)&d_plane_dict, dictBytes);
			cudaMalloc((void**)&d_out_paths, exactBytes);

			// 3. 搬运数据进 GPU 
			// (注意：这里直接浅拷贝 dictArray 即可，因为 entry 里面的 d_occupancy_bitmap 已经是 GPU 地址了)
			cudaMemcpy(d_topologies, uniqueTopologies.data(), topoBytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_plane_dict, dictArray.data(), dictBytes, cudaMemcpyHostToDevice);

			// 4. 调用新的 TEST CUDA Kernel
			launchImageMethodKernel_TEST(
				d_topologies, static_cast<int>(uniqueTopologies.size()),
				d_plane_dict, static_cast<int>(dictArray.size()),
				tx, rx, d_out_paths
			);
			cudaDeviceSynchronize();

			// 5. 拉回结果
			cudaMemcpy(results.data(), d_out_paths, exactBytes, cudaMemcpyDeviceToHost);

			// 6. 打扫战场 (暂不释放 d_bitmap 以供后续测试)
			cudaFree(d_topologies); cudaFree(d_plane_dict); cudaFree(d_out_paths);

			return results;
		}
	} // namespace Tracer
} // namespace Engine
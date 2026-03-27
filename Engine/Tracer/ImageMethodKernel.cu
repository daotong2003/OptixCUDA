// 整个阶段二的灵魂！在 Engine/Tracer/ 目录下新建这个 .cu 文件。它包含了设备端的二分查找、镜像翻转和光线求交
#include "ImageMethodKernel.h"
#include <cuda_runtime.h>
#include <cmath>

namespace Engine {
	namespace Tracer {
		// ==================== [设备端数学工具] ====================
		__device__ float3 d_mirrorPoint(const float3& pt, const PlaneEquation& plane) {
			float dist = plane.normal.x * pt.x + plane.normal.y * pt.y + plane.normal.z * pt.z + plane.d;
			return make_float3(
				pt.x - 2.0f * dist * plane.normal.x,
				pt.y - 2.0f * dist * plane.normal.y,
				pt.z - 2.0f * dist * plane.normal.z
			);
		}

		__device__ float3 d_intersectLinePlane(const float3& p1, const float3& p2, const PlaneEquation& plane, bool& out_intersected) {
			float3 dir = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
			float dotDirNormal = dir.x * plane.normal.x + dir.y * plane.normal.y + dir.z * plane.normal.z;
			// [核心修复] 必须使用 fabsf()！绝不能用 abs()！
			if (fabsf(dotDirNormal) < 1e-6f) {
				out_intersected = false;
				return make_float3(0, 0, 0);
			}
			float distP1 = plane.normal.x * p1.x + plane.normal.y * p1.y + plane.normal.z * p1.z + plane.d;
			float t = -distP1 / dotDirNormal;
			// 物理防护：交点必须严格在线段内部
			if (t < -0.001f || t > 1.001f) { out_intersected = false; return make_float3(0, 0, 0); }
			out_intersected = true;
			return make_float3(p1.x + t * dir.x, p1.y + t * dir.y, p1.z + t * dir.z);
		}

		// GPU 端高速二分查找平面方程 (要求传入的字典必须按 label 排序)
		__device__ bool d_findPlane(int32_t label, const PlaneDictEntry* dict, int dictSize, PlaneEquation& out_plane) {
			int left = 0, right = dictSize - 1;
			while (left <= right) {
				int mid = left + (right - left) / 2;
				if (dict[mid].label == label) { out_plane = dict[mid].eq; return true; }
				if (dict[mid].label < label) left = mid + 1;
				else right = mid - 1;
			}
			return false;
		}

		// ==================== [并发解算 Kernel] ====================
		__global__ void solveExactPathsKernel(
			const PathTopology* topologies, int num_topologies,
			const PlaneDictEntry* plane_dict, int num_planes,
			float3 tx, float3 rx, ExactPath* out_paths)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= num_topologies) return;

			PathTopology topo = topologies[idx];
			ExactPath path;
			path.isValid = false; path.vertexCount = 0;

			int numBounces = topo.nodeCount;
			if (numBounces > MAX_BOUNCE_DEPTH) { out_paths[idx] = path; return; }

			// 1. 直射 (LOS) 兜底
			if (numBounces == 0) {
				path.vertices[0] = tx; path.vertices[1] = rx;
				path.vertexCount = 2; path.isValid = true;
				out_paths[idx] = path; return;
			}

			// 2. 查字典获取物理平面
			PlaneEquation planes[MAX_BOUNCE_DEPTH];
			for (int i = 0; i < numBounces; ++i) {
				if (!d_findPlane(topo.nodes[i].plane_label, plane_dict, num_planes, planes[i])) {
					out_paths[idx] = path; return; // 异常：找不到平面
				}
			}

			// 3. 核心算法：正向镜像折叠
			float3 images[MAX_BOUNCE_DEPTH];
			images[0] = d_mirrorPoint(tx, planes[0]);
			for (int i = 1; i < numBounces; ++i) images[i] = d_mirrorPoint(images[i - 1], planes[i]);

			// 4. 核心算法：反向连线求交
			float3 hitPoints[MAX_BOUNCE_DEPTH];
			float3 currentTarget = rx;
			for (int i = numBounces - 1; i >= 0; --i) {
				// [终极修复] 连线的起点永远是【当前平面对应的虚拟镜像源】！
				float3 rayOrigin = images[i];

				bool intersected = false;
				float3 p = d_intersectLinePlane(rayOrigin, currentTarget, planes[i], intersected);
				if (!intersected) { out_paths[idx] = path; return; } // 数学死角，判定该多径失效
				hitPoints[i] = p;
				currentTarget = p;
			}

			// 5. 组装写回
			path.vertices[0] = tx;
			for (int i = 0; i < numBounces; ++i) path.vertices[i + 1] = hitPoints[i];
			path.vertices[numBounces + 1] = rx;
			path.vertexCount = numBounces + 2;
			path.isValid = true;

			out_paths[idx] = path;
		}

		// ==================== [C++ 启动器包装] ====================
		void launchImageMethodKernel(
			const PathTopology* d_topologies, int num_topologies,
			const PlaneDictEntry* d_plane_dict, int num_planes,
			float3 tx, float3 rx, ExactPath* d_out_paths)
		{
			int blockSize = 256;
			int gridSize = (num_topologies + blockSize - 1) / blockSize;
			solveExactPathsKernel << <gridSize, blockSize >> > (d_topologies, num_topologies, d_plane_dict, num_planes, tx, rx, d_out_paths);
		}
	} // namespace Tracer
} // namespace Engine
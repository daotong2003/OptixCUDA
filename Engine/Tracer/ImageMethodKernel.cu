// 整个阶段二的灵魂！在 Engine/Tracer/ 目录下新建这个 .cu 文件。它包含了设备端的二分查找、镜像翻转和光线求交
#include "ImageMethodKernel.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
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

		// ========================================================================
		// [TDD Step 3] 局部字典查找与 2D 位图拦截器核心 (修复版)
		// ========================================================================
		__device__ bool d_findLocalPlane_TEST(int32_t label, const LocalPlaneDictEntry* dict, int dictSize, LocalPlaneDictEntry& out_plane) {
			int left = 0, right = dictSize - 1;
			while (left <= right) {
				int mid = left + (right - left) / 2;
				if (dict[mid].label == label) { out_plane = dict[mid]; return true; }
				if (dict[mid].label < label) left = mid + 1;
				else right = mid - 1;
			}
			return false;
		}

		__global__ void solveExactPathsKernel_TEST(
			const PathTopology* topologies, int num_topologies,
			const LocalPlaneDictEntry* plane_dict, int num_planes,
			float3 tx, float3 rx, ExactPath* out_paths)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= num_topologies) return;

			PathTopology topo = topologies[idx];

			// 【修复 1】: PathTopology 的成员是 nodeCount
			int numBounces = topo.nodeCount;

			ExactPath path;

			// 【修复 2】: ExactPath 包含发射点 Tx 和接收点 Rx，所以总顶点数是 bounces + 2
			path.vertexCount = numBounces + 2;
			path.isValid = false;

			if (numBounces == 0 || numBounces > MAX_BOUNCE_DEPTH) { out_paths[idx] = path; return; }

			// 1. 查字典获取局部平面信息
			LocalPlaneDictEntry planes[MAX_BOUNCE_DEPTH];
			for (int i = 0; i < numBounces; ++i) {
				if (!d_findLocalPlane_TEST(topo.nodes[i].plane_label, plane_dict, num_planes, planes[i])) {
					out_paths[idx] = path; return;
				}
			}

			// 2. 正向镜像折叠
			float3 images[MAX_BOUNCE_DEPTH];
			images[0] = d_mirrorPoint(tx, planes[0].eq);
			for (int i = 1; i < numBounces; ++i) {
				images[i] = d_mirrorPoint(images[i - 1], planes[i].eq);
			}

			// 3. 反向连线求交与 2D 占据位图拦截
			float3 hitPoints[MAX_BOUNCE_DEPTH];
			float3 currentTarget = rx;

			for (int i = numBounces - 1; i >= 0; --i) {
				float3 rayOrigin = images[i];
				bool intersected = false;

				// 计算绝对数学交点 p
				float3 p = d_intersectLinePlane(rayOrigin, currentTarget, planes[i].eq, intersected);
				if (!intersected) { out_paths[idx] = path; return; }

				// --- [核心：物理边界极速拦截] ---
				float3 diff = make_float3(p.x - planes[i].local_origin.x,
					p.y - planes[i].local_origin.y,
					p.z - planes[i].local_origin.z);

				float u = diff.x * planes[i].axisU.x + diff.y * planes[i].axisU.y + diff.z * planes[i].axisU.z;
				float v = diff.x * planes[i].axisV.x + diff.y * planes[i].axisV.y + diff.z * planes[i].axisV.z;

				// 【修复 5】：增加 floorf 防止向零截断导致边缘坐标错乱！
				int u_idx = (int)floorf((u - planes[i].min_u) / planes[i].grid_size);
				int v_idx = (int)floorf((v - planes[i].min_v) / planes[i].grid_size);

				// ==========================================================
				// 【核心修正】：接住那些落在 5cm 容差边缘的交点，把它们拉回 0 号网格！
				if (u_idx == -1) u_idx = 0;
				if (u_idx == planes[i].cols) u_idx = planes[i].cols - 1;
				if (v_idx == -1) v_idx = 0;
				if (v_idx == planes[i].rows) v_idx = planes[i].rows - 1;
				// ==========================================================

				if (u_idx < 0 || u_idx >= planes[i].cols || v_idx < 0 || v_idx >= planes[i].rows) {
					out_paths[idx] = path; return; // 拦截：超出物理包围盒
				}

				int bit_idx = v_idx * planes[i].cols + u_idx;
				if (!planes[i].d_occupancy_bitmap[bit_idx]) {
					out_paths[idx] = path; return; // 拦截：落入无点云空洞
				}
				// -------------------------------

				hitPoints[i] = p;
				currentTarget = p;
			}

			// 【修复 3】：严格按照 ExactPath 的 vertices 数组结构，填入 Tx、反射点、Rx
			path.vertices[0] = tx; // 起点
			for (int i = 0; i < numBounces; ++i) {
				path.vertices[i + 1] = hitPoints[i]; // 中间的反射交点
			}
			path.vertices[numBounces + 1] = rx; // 终点

			path.isValid = true;
			out_paths[idx] = path;
		}

		void launchImageMethodKernel_TEST(
			const PathTopology* d_topologies, int num_topologies,
			const LocalPlaneDictEntry* d_plane_dict, int num_planes,
			float3 tx, float3 rx, ExactPath* d_out_paths)
		{
			int blockSize = 256;
			int gridSize = (num_topologies + blockSize - 1) / blockSize;
			solveExactPathsKernel_TEST << <gridSize, blockSize >> > (
				d_topologies, num_topologies, d_plane_dict, num_planes, tx, rx, d_out_paths);

			// 【修复 4】：使用原生 CUDA API 错误捕获，直接打印错误字符串，避免对 CUDA_CHECK 宏的依赖
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("Kernel Launch Error in launchImageMethodKernel_TEST: %s\n", cudaGetErrorString(err));
			}
		}
	} // namespace Tracer
} // namespace Engine
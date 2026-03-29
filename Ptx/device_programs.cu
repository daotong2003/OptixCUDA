#include <optix_device.h>
#include "../Engine/Scene/MeshTypes.h"
#include "../NimbusCommon/LaunchParams.h"
#include "../NimbusCommon/SbtData.h"

extern "C" {
	__constant__ Engine::LaunchParams params;
}

// =========================================================================
// [阶段一核心] 扩充 PRD 寄存器负载
// =========================================================================
struct PerRayData {
	float3 hit_pos;
	float3 hit_normal;
	uint8_t hit_material;
	int hit_status;

	int32_t hit_instance_id;
	int32_t hit_plane_label; // [修复] 确保这里是 hit_plane_label
};

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	return reinterpret_cast<void*>(uptr);
}

static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32; i1 = uptr & 0x00000000ffffffff;
}

/*
// [修复] 增加 int32_t& out_label
	__device__ __inline__ void findNearestPointAndExtract(
		const Engine::HitGroupData* sbtData, unsigned int prim_idx, float3 hit_pt, float3& out_normal, uint8_t& out_material, int32_t& out_label) {
		if (!sbtData->pointOffsets || !sbtData->pointCounts || !sbtData->pointIndices || !params.globalPointCloud) {
			out_normal = make_float3(0.0f, 1.0f, 0.0f); out_material = 0; out_label = -1; return;
		}
		uint32_t offset = sbtData->pointOffsets[prim_idx];
		uint32_t count = sbtData->pointCounts[prim_idx];
		float min_dist_sq = 1e30f;
		float3 best_normal = make_float3(0, 0, 0); uint8_t best_material = 0;
		int32_t best_label = -1;

		for (uint32_t i = 0; i < count; ++i) {
			uint32_t pt_idx = sbtData->pointIndices[offset + i];
			Engine::Geometry::Point pt = params.globalPointCloud[pt_idx];
			float diff_x = hit_pt.x - pt.x; float diff_y = hit_pt.y - pt.y; float diff_z = hit_pt.z - pt.z;
			float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			if (dist_sq < min_dist_sq) {
				min_dist_sq = dist_sq;
				best_normal = make_float3(pt.nx, pt.ny, pt.nz);
				best_material = pt.material;
				best_label = pt.label; // 提取点云的平面标签
			}
		}
		out_normal = best_normal; out_material = best_material; out_label = best_label;
	}
*/
// ... 前面的 PRD 定义和 findNearestPointAndExtract (记得加上 out_label) 保持更新后的样子 ...

extern "C" __global__ void __closesthit__los() {
	const Engine::HitGroupData* sbtData = (Engine::HitGroupData*)optixGetSbtDataPointer();

	float3 ray_orig = optixGetWorldRayOrigin();
	float3 ray_dir = optixGetWorldRayDirection();
	float t = optixGetRayTmax();
	float3 hit_pt = make_float3(ray_orig.x + t * ray_dir.x, ray_orig.y + t * ray_dir.y, ray_orig.z + t * ray_dir.z);

	unsigned int prim_idx = optixGetPrimitiveIndex();

	// ==================== [修复：获取三角形绝对几何法线] ====================
	float3 v[3]; // OptiX API 要求必须传大小为 3 的数组
	optixGetTriangleVertexData(optixGetGASTraversableHandle(), prim_idx, optixGetSbtGASIndex(), 0.0f, v);

	float3 edge1 = make_float3(v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z);
	float3 edge2 = make_float3(v[2].x - v[0].x, v[2].y - v[0].y, v[2].z - v[0].z);

	// 叉乘并手动归一化 (防 CUDA 找不到 normalize)
	float nx = edge1.y * edge2.z - edge1.z * edge2.y;
	float ny = edge1.z * edge2.x - edge1.x * edge2.z;
	float nz = edge1.x * edge2.y - edge1.y * edge2.x;
	float invLen = 1.0f / sqrtf(nx * nx + ny * ny + nz * nz);
	float3 geo_normal = make_float3(nx * invLen, ny * invLen, nz * invLen);
	// =========================================================================

	// ==================== [提取点云标签] ====================
	uint32_t p0 = optixGetPayload_0();
	uint32_t p1 = optixGetPayload_1();
	PerRayData* prd = (PerRayData*)unpackPointer(p0, p1);

	prd->hit_pos = hit_pt;
	prd->hit_normal = geo_normal;
	prd->hit_material = sbtData->material_id; // 从 sbt 直接拿
	prd->hit_status = 1;
	prd->hit_instance_id = sbtData->instance_id;

	// [终极极速] 直接 O(1) 提取专属 Label！彻底消灭微多径抖动！
	prd->hit_plane_label = sbtData->plane_label;

}

extern "C" __global__ void __miss__los() {
	uint32_t p0 = optixGetPayload_0();
	uint32_t p1 = optixGetPayload_1();
	PerRayData* prd = (PerRayData*)unpackPointer(p0, p1);
	prd->hit_status = 0;
}
// =========================================================================
// 3. 射线生成着色器 (Raygen) - 并发核心
// =========================================================================
extern "C" __global__ void __raygen__los() {
	// 获取当前这根射线的独立编号
	uint32_t idx = optixGetLaunchIndex().x;

	// =========================================================================
	// [阶段三核心] 拦截器：视距遮挡验证模式 (特洛伊木马)
	// =========================================================================
	if (params.validationPaths != nullptr) {
		if (idx >= params.numValidationPaths) return;

		Engine::Tracer::ExactPath path = params.validationPaths[idx];
		if (!path.isValid) return; // 已经被阶段二枪毙的，直接跳过

		bool isOccluded = false;

		// 遍历路径中的每一条线段
		for (int i = 0; i < path.vertexCount - 1; ++i) {
			float3 start_pt = path.vertices[i];
			float3 end_pt = path.vertices[i + 1];

			float3 dir = make_float3(end_pt.x - start_pt.x, end_pt.y - start_pt.y, end_pt.z - start_pt.z);
			float dist = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

			if (dist < 1e-4f) continue;
			dir.x /= dist; dir.y /= dist; dir.z /= dist;

			PerRayData prd;
			prd.hit_status = 0; // 0 表示安全
			uint32_t p0, p1;
			packPointer(&prd, p0, p1);

			// 发射阴影射线
			optixTrace(
				params.handle, start_pt, dir,
				0.005f, dist - 0.005f, 0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				0, 1, 0, p0, p1
			);

			if (prd.hit_status == 1) {
				isOccluded = true;
				break; // 一段被挡，整条作废
			}
		}

		if (isOccluded) params.validationPaths[idx].isValid = false;

		return; // 验证结束，直接 Return，不执行后面的 SBR 代码！
	}
	// =========================================================================

	// =========================================================================
	// [阶段一核心] 原有的 SBR 射线弹跳逻辑
	// =========================================================================
	bool isBatchSBR = (params.outCandidateTopologies != nullptr);
	bool isSingleSBR = (params.outSbrPath != nullptr);

	if (isBatchSBR && idx >= params.numRays) return; // 并发越界保护
	if (!isBatchSBR && idx > 0) return;              // 旧模式只准线程0运行

	// 设置当前射线的起点和方向
	float3 current_origin = make_float3(params.rayOrigin_x, params.rayOrigin_y, params.rayOrigin_z);
	float3 current_dir = isBatchSBR ? params.txRayDirections[idx] :
		make_float3(params.rayDirection_x, params.rayDirection_y, params.rayDirection_z);

	int max_depth = (isBatchSBR || isSingleSBR) ? params.maxBounceDepth : 1;

	// 取出该射线专属的账本
	Engine::Tracer::PathTopology* current_topo = nullptr;
	if (isBatchSBR) {
		current_topo = &params.outCandidateTopologies[idx];
		current_topo->hitRx = false;
		current_topo->nodeCount = 0;
	}

	PerRayData prd;
	uint32_t p0, p1;
	int depth = 0;

	for (depth = 0; depth < max_depth; ++depth) {
		prd.hit_status = 0;
		packPointer(&prd, p0, p1);

		// 发射硬件光追
		optixTrace(
			params.handle, current_origin, current_dir,
			0.001f, params.tmax, 0.0f,
			OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			0, 1, 0, p0, p1
		);

		if (prd.hit_status == 1) {
			if (isBatchSBR) {
				current_topo->nodes[depth].instance_id = prd.hit_instance_id;
				current_topo->nodes[depth].plane_label = prd.hit_plane_label;
				// 【核心】：每次击中都实时更新深度，无论最后是否飞向太空，这都是一条有效的残缺拓扑
				current_topo->nodeCount = depth + 1;
			}

			if (isSingleSBR) {
				params.outSbrPath->nodes[depth].position = prd.hit_pos;
			}

			float3 N = prd.hit_normal;
			float dotIN = current_dir.x * N.x + current_dir.y * N.y + current_dir.z * N.z;
			if (dotIN > 0.0f) { N.x = -N.x; N.y = -N.y; N.z = -N.z; dotIN = -dotIN; }

			float3 reflect_dir = make_float3(
				current_dir.x - 2.0f * dotIN * N.x,
				current_dir.y - 2.0f * dotIN * N.y,
				current_dir.z - 2.0f * dotIN * N.z
			);

			current_origin = make_float3(prd.hit_pos.x + N.x * 0.005f, prd.hit_pos.y + N.y * 0.005f, prd.hit_pos.z + N.z * 0.005f);
			current_dir = reflect_dir;

			if (!isBatchSBR && !isSingleSBR) break;
		}
		else {
			// 射线飞向太空，但它之前撞击表面的记录依然保留在 current_topo 中
			if (isSingleSBR) params.outSbrPath->isEscaped = true;
			break;
		}
	}

	if (isSingleSBR) params.outSbrPath->nodeCount = depth;
}
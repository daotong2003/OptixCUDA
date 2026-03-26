#include <optix_device.h>
// 强行引入带有 Point 定义的头文件，解决 pt 和命名空间报错
#include "../Engine/Scene/MeshTypes.h"   // 提供 Engine::Geometry::Point 的完整结构
#include "../NimbusCommon/LaunchParams.h"
#include "../NimbusCommon/SbtData.h"

// 声明全局的 params 快递盒
extern "C" {
	__constant__ Engine::LaunchParams params;
}

// =========================================================================
// 1. 射线生成着色器 (Raygen) - 负责把射线打出去
// =========================================================================
extern "C" __global__ void __raygen__los() {
	// 从 Host 端传来的 params 中提取射线起点和方向
	float3 ray_origin = make_float3(params.rayOrigin_x, params.rayOrigin_y, params.rayOrigin_z);
	float3 ray_dir = make_float3(params.rayDirection_x, params.rayDirection_y, params.rayDirection_z);

	unsigned int p0 = 0; // 载荷 (Payload) 占位符

	// 正式发射射线！
	optixTrace(
		params.handle,
		ray_origin,
		ray_dir,
		0.0f,                      // tmin (最近距离)
		params.tmax,               // tmax (最远距离)
		0.0f,                      // rayTime
		OptixVisibilityMask(255),  // 掩码，全可见
		OPTIX_RAY_FLAG_DISABLE_ANYHIT, // 关闭 AnyHit 提升性能
		0, 1, 0,                   // SBT 的 offset, stride, missSBTIndex
		p0                         // 绑定载荷
	);
}

// =========================================================================
// 2. 未命中着色器 (Miss) - 负责处理射线打向太空的情况
// =========================================================================
extern "C" __global__ void __miss__los() {
	if (params.outHitStatus) {
		*(params.outHitStatus) = 0; // 0 代表未命中 (Miss)
	}
}

// =========================================================================
// 3. 命中着色器与微观校准逻辑 (ClosestHit)
// =========================================================================

// 独立的最近邻查找逻辑 (微观物理校准)
__device__ __inline__ void findNearestPointAndExtract(
	const Engine::HitGroupData* sbtData,
	unsigned int prim_idx,
	float3 hit_pt,
	float3& out_normal,
	uint8_t& out_material)
{
	// 防御性：若面片本身没有点云映射，返回默认值
	if (!sbtData->pointOffsets || !sbtData->pointCounts || !sbtData->pointIndices || !params.globalPointCloud) {
		out_normal = make_float3(0.0f, 1.0f, 0.0f);
		out_material = 0;
		return;
	}

	// 从 SBT 中抽取出该代理三角形对应的局部点云范围
	uint32_t offset = sbtData->pointOffsets[prim_idx];
	uint32_t count = sbtData->pointCounts[prim_idx];

	float min_dist_sq = 1e30f;
	float3 best_normal = make_float3(0, 0, 0);
	uint8_t best_material = 0;

	// 遍历计算欧氏距离，找出真实表面上距离击中点最近的那个原始点
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t pt_idx = sbtData->pointIndices[offset + i];

		// 从全局点云池中提取真实点属性
		Engine::Geometry::Point pt = params.globalPointCloud[pt_idx];

		float diff_x = hit_pt.x - pt.x;
		float diff_y = hit_pt.y - pt.y;
		float diff_z = hit_pt.z - pt.z;
		float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

		if (dist_sq < min_dist_sq) {
			min_dist_sq = dist_sq;
			best_normal = make_float3(pt.nx, pt.ny, pt.nz);
			best_material = pt.material;
		}
	}

	out_normal = best_normal;
	out_material = best_material;
}

extern "C" __global__ void __closesthit__los() {
	// 1. 获取当前命中的 SBT 记录 (读取我们在 Host 端绑定的显存指针)
	const Engine::HitGroupData* sbtData = (Engine::HitGroupData*)optixGetSbtDataPointer();

	// 2. 获取射线在 3D 空间中的命中交点坐标
	float3 ray_orig = optixGetWorldRayOrigin();
	float3 ray_dir = optixGetWorldRayDirection();
	float t = optixGetRayTmax();
	float3 hit_pt = make_float3(ray_orig.x + t * ray_dir.x,
		ray_orig.y + t * ray_dir.y,
		ray_orig.z + t * ray_dir.z);

	// 3. 获取命中的三角形索引 ID
	unsigned int prim_idx = optixGetPrimitiveIndex();

	// 4. 执行微观物理校准，提取高精度法线和材质
	float3 real_normal;
	uint8_t real_material;
	findNearestPointAndExtract(sbtData, prim_idx, hit_pt, real_normal, real_material);

	// 5. 将宏观坐标与微观物理数据一起写回 LaunchParams 输出通道
	if (params.outHitStatus) {
		*(params.outHitStatus) = 1;

		*(params.outHitPosition_x) = hit_pt.x;
		*(params.outHitPosition_y) = hit_pt.y;
		*(params.outHitPosition_z) = hit_pt.z;

		if (params.outHitNormal_x) {
			*(params.outHitNormal_x) = real_normal.x;
			*(params.outHitNormal_y) = real_normal.y;
			*(params.outHitNormal_z) = real_normal.z;
		}
		if (params.outHitMaterial) {
			*(params.outHitMaterial) = real_material;
		}
	}
}
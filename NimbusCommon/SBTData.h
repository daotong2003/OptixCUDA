// 用于严格定义 CPU 和 GPU 共用的 SBT 内存布局。
// OptiX 要求 Record 结构体必须遵循严格的 16 字节对齐。
#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

namespace Engine {
	struct EmptyData {};// 射线生成与未命中着色器不需要携带专属数据，用空结构体占位

	// 命中组 (HitGroup) 专属数据包 - CPU端装填指针，GPU端解包读取
	struct HitGroupData {
		float3* vertices;       // 代理网格顶点数组
		uint3* indices;         // 代理网格索引数组
		uint8_t material_id;    // 材质标签
		int32_t instance_id;    // 部件标签

		// 微观物理校准指针
		uint32_t* pointOffsets;
		uint32_t* pointCounts;
		uint32_t* pointIndices;

		// [新增] O(1) 极速查找的专属平面标签
		int32_t plane_label;
	};

	// OptiX 严格要求对齐的 SBT 记录模板
	template <typename T>
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
		char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	// 为三种着色器定义具体的 Record 类型
	typedef SbtRecord<EmptyData> RaygenRecord;
	typedef SbtRecord<EmptyData> MissRecord;
	typedef SbtRecord<HitGroupData> HitGroupRecord;
} // namespace Engine
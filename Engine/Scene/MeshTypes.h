#pragma once
#include <cstdint>            // 提供 uint8_t, int32_t 等定宽整数
#include <vector_types.h>     // 提供 CUDA 内置的 float3, uint3 等
#include <vector_functions.h> // 提供 make_float3 函数定义

// 【隔离防线 1】仅在 CPU 宿主端编译时，才引入复杂的标准库
#ifndef __CUDACC__
#include <vector>
#include <iostream>
#endif

namespace Engine {
	namespace Geometry {
		// 1. 原始点云结构体 (CPU 和 GPU 共享的纯净数据)
#pragma pack(push, 1)
		struct Point {
			float x, y, z;          // 坐标
			float nx, ny, nz;       // 法向量
			uint8_t material;       // 材质标签
			int32_t label;          // 平面标签
			int32_t instance_id;    // 实例/部件标签

			// 打上双端标签，允许显卡底层直接调用
			__host__ __device__ inline float3 getPos() const { return make_float3(x, y, z); }
			__host__ __device__ inline float3 getNormal() const { return make_float3(nx, ny, nz); }
		};
#pragma pack(pop)

		// 【隔离防线 2】将包含 std::vector 的三角形网格对显卡完全隐身，防止内存越界报错
#ifndef __CUDACC__
		// 2. 转换后的纯净三角形网格数据 (仅供 Host 端 CPU 消费)
		struct TriangleMesh {
			std::vector<float3> vertices;
			std::vector<uint3> indices;

			uint8_t material_id = 0;
			int32_t instance_id = -1;

			std::vector<uint32_t> pointOffsets;
			std::vector<uint32_t> pointCounts;
			std::vector<uint32_t> pointIndices;

			void printMappingStatus() const {
				std::cout << "========== [TDD Check] TriangleMesh Mapping ==========\n"
					<< "  Triangle Count (indices)  : " << indices.size() << "\n"
					<< "  pointOffsets size         : " << pointOffsets.size() << "\n"
					<< "  pointCounts size          : " << pointCounts.size() << "\n"
					<< "  pointIndices total pool   : " << pointIndices.size() << "\n"
					<< "======================================================\n";
			}
		};
#endif // __CUDACC__
	} // End of namespace Geometry (闭合必须严谨)
} // End of namespace Engine
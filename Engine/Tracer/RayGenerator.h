#pragma once
#include <vector>
#include <vector_types.h>

namespace Engine {
	namespace Tracer {
		class RayGenerator {
		public:
			// 核心算法：生成均匀分布在单位球面上的射线方向（斐波那契球面采样）
			// numRays: 需要生成的射线总数
			static std::vector<float3> generateFibonacciSphere(size_t numRays);
		};
	} // namespace Tracer
} // namespace Engine
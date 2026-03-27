#include "RayGenerator.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 添加以下头文件以定义 make_float3
#include <cuda_runtime.h>

namespace Engine {
	namespace Tracer {
		std::vector<float3> RayGenerator::generateFibonacciSphere(size_t numRays) {
			std::vector<float3> rays;
			if (numRays == 0) return rays;
			rays.reserve(numRays);

			// 黄金角 (Golden Angle)
			const float phi = static_cast<float>(M_PI * (3.0 - std::sqrt(5.0)));

			for (size_t i = 0; i < numRays; ++i) {
				// y 轴坐标从 1 到 -1 均匀分布
				float y = 1.0f - (static_cast<float>(i) / static_cast<float>(numRays - 1)) * 2.0f;
				if (numRays == 1) y = 0.0f; // 兜底：如果只要1条射线，发向水平

				// 当前高度 y 下的截面圆半径
				float radius = std::sqrt(1.0f - y * y);

				// 绕 Y 轴的旋转角
				float theta = phi * static_cast<float>(i);

				// 计算 X 和 Z 坐标
				float x = std::cos(theta) * radius;
				float z = std::sin(theta) * radius;

				// 使用 CUDA 内置的 make_float3 打包
				rays.push_back(make_float3(x, y, z));
			}

			return rays;
		}
	} // namespace Tracer
} // namespace Engine
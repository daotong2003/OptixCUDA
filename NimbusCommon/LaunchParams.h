#pragma once
#include <optix.h>
#include <cstdint> // [Step 1 新增] 引入 cstdint 以支持 uint8_t 材质类型
#include "Engine/Scene/MeshTypes.h"

// =======================================================
// 【修复 1】前向声明：提前告诉编译器存在这个命名空间和结构体
// =======================================================
namespace Engine {
	namespace Geometry {
		struct Point;
	}
}

namespace Engine {
	// 这个结构体是 CPU 和 GPU 之间的数据“快递盒”
	struct LaunchParams {
		OptixTraversableHandle handle; // 场景的句柄

		// 发射机  设置
		float rayOrigin_x, rayOrigin_y, rayOrigin_z;
		float rayDirection_x, rayDirection_y, rayDirection_z;
		float tmax;

		// 接收端 输出缓存
		int* outHitStatus;
		float* outHitPosition_x;
		float* outHitPosition_y;
		float* outHitPosition_z;

		// ==================== [Step 1 新增] ====================
		// 接收端 (Rx) 输出缓存 - 微观物理校准数据 (高精度法线与材质)
		float* outHitNormal_x;
		float* outHitNormal_y;
		float* outHitNormal_z;
		uint8_t* outHitMaterial;

		Geometry::Point* globalPointCloud;
	};
} // namespace Engine
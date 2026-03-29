#pragma once
#include "SbrTypes.h"
#include <vector_types.h>
// C++ 调度器和纯 CUDA Kernel 之间的通讯桥梁
namespace Engine {
	namespace Tracer {
		// 暴露给 C++ 宿主机的 CUDA Kernel 启动器
		void launchImageMethodKernel(
			const PathTopology* d_topologies, int num_topologies,
			const PlaneDictEntry* d_plane_dict, int num_planes,
			float3 tx, float3 rx,
			ExactPath* d_out_paths
		);
		// [TDD Step 3 新增] 带有位图拦截器的新版 Kernel 启动器
		// =========================================================
		void launchImageMethodKernel_TEST(
			const PathTopology* d_topologies, int num_topologies,
			const LocalPlaneDictEntry* d_plane_dict, int num_planes,
			float3 tx, float3 rx,
			ExactPath* d_out_paths
		);
	} // namespace Tracer
} // namespace Engine
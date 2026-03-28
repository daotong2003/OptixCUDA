#include "Engine/Core/Context.h"
#include "Engine/Scene/GeometryManager.h"
#include "Engine/Tracer/RayTracer.h"
#include "Engine/Scene/PointCloudConverter.h"
#include "Engine/Scene/SceneManager.h"
#include "Engine/Core/CudaError.h"
#include "Engine/Tracer/RayGenerator.h"
#include "Engine/Tracer/ImageMethodSolver.h"
#include "Engine/Debug/PathExporter.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// ========================================================================
// 引擎调试开关 (如果需要开启导出 OBJ 或单步诊断，取消下一行的注释即可)
// 或者推荐在 CMakeLists.txt 中通过 target_compile_definitions 全局配置
// ========================================================================
// #define ENABLE_ENGINE_DEBUG

#ifdef ENABLE_ENGINE_DEBUG
#include "Engine/Debug/Exporter3D.h"

// 诊断函数 (仅在开启调试宏时编译)
void runSBRDiagnostics(
	bool enable,
	const std::unordered_map<int32_t, std::vector<Engine::Geometry::TriangleMesh>>& sceneMeshes,
	Engine::Tracer::RayTracer& tracer,
	Engine::Geometry::Point* d_globalCloud)
{
	if (!enable) return;

	std::cout << "\n>>> [TDD Check] ----------------------------------------\n";
	std::cout << ">>> [TDD Check] 开启 SBR 追踪器诊断模式...\n";

	auto firstInstance = sceneMeshes.begin();
	while (firstInstance != sceneMeshes.end() && firstInstance->second.empty()) {
		firstInstance++;
	}
	if (firstInstance == sceneMeshes.end()) {
		std::cout << ">>> [致命错误] sceneMeshes 为空，无法进行诊断！\n";
		return;
	}

	auto& testMesh = firstInstance->second.front();
	float3 v0 = testMesh.vertices[testMesh.indices[0].x];
	float3 v1 = testMesh.vertices[testMesh.indices[0].y];
	float3 v2 = testMesh.vertices[testMesh.indices[0].z];

	float tx = (v0.x + v1.x + v2.x) / 3.0f;
	float ty = (v0.y + v1.y + v2.y) / 3.0f;
	float tz = (v0.z + v1.z + v2.z) / 3.0f;

	float ox = tx + 0.1f, oy = ty + 0.1f, oz = tz + 0.1f;
	float dx = tx - ox, dy = ty - oy, dz = tz - oz;
	float len = std::sqrt(dx * dx + dy * dy + dz * dz);
	if (len > 0.0001f) { dx /= len; dy /= len; dz /= len; }

	std::cout << ">>> [诊断信息] 发射源坐标: (" << ox << ", " << oy << ", " << oz << ")\n";

	Engine::Tracer::RayPath sbrResult = {};
	tracer.shootRaySBR(ox, oy, oz, dx, dy, dz, &sbrResult, d_globalCloud);

	std::cout << "\n>>> [SBR Check] 射线追踪结束，路径拓扑分析:\n";
	std::cout << "  记录实际弹跳次数: " << sbrResult.nodeCount << "\n";
	std::cout << "  是否逃逸(未命中): " << (sbrResult.isEscaped ? "Yes" : "No") << "\n";
	for (int i = 0; i < sbrResult.nodeCount; ++i) {
		std::cout << "  [Bounce " << i << "] Pos: ("
			<< sbrResult.nodes[i].position.x << ", "
			<< sbrResult.nodes[i].position.y << ", "
			<< sbrResult.nodes[i].position.z << ") | Dist: "
			<< sbrResult.nodes[i].distance << "\n";
	}
	std::cout << ">>> [TDD Check] ----------------------------------------\n\n";
}
#endif // ENABLE_ENGINE_DEBUG

// ========================================================================
// 辅助工具：拓扑去重处理器
// ========================================================================
std::vector<Engine::Tracer::PathTopology> extractUniqueTopologies(const std::vector<Engine::Tracer::PathTopology>& candidateTopologies) {
	std::vector<Engine::Tracer::PathTopology> uniqueTopologies;
	int hitCount = 0;
	for (const auto& topo : candidateTopologies) {
		if (topo.hitRx) {
			hitCount++;
			// 提示：后续可在此处根据 topo.nodes 的 plane_label 序列进行严格的 Hash 去重
			uniqueTopologies.push_back(topo);
		}
	}
	std::cout << "  -> 统计完成：共有 " << hitCount << " 条射线到达 Rx 捕获球" << std::endl;
	return uniqueTopologies;
}

int main() {
	try {
		// ========================================================
		// [阶段零] 引擎初始化与物理场景构建 (Infrastructure)
		// ========================================================
		Engine::Core::OptixContextManager optixManager;
		Engine::Core::GeometryManager geometryManager(optixManager.getContext());
		Engine::Core::SceneManager sceneManager(optixManager.getContext());
		Engine::Geometry::PointCloudConverter meshConverter(0.15f);

		std::vector<Engine::Geometry::Point> rawCloud;
		std::unordered_map<int32_t, std::vector<Engine::Geometry::TriangleMesh>> sceneMeshes;

		std::cout << "[System] 正在加载点云数据..." << std::endl;
		if (meshConverter.loadFromBinaryPLY("E:/RT_software/Clanguage/OptixCUDA/SY1101.ply", rawCloud)) {
			sceneMeshes = meshConverter.convertToMeshes(rawCloud);

#ifdef ENABLE_ENGINE_DEBUG
			// 耗时操作：导出 OBJ 仅在调试模式下执行
			Engine::Debug::Exporter3D::exportMeshesToOBJ(sceneMeshes, "debug_greedy_meshes.obj");
#endif

			geometryManager.buildSceneGAS(sceneMeshes);
		}

		// 装载部件并构建 IAS
		for (const auto& pair : geometryManager.getAllGasHandles()) {
			sceneManager.addInstance(pair.first, pair.second);
		}
		sceneManager.buildIAS();

		// 提炼宏观平面方程
		auto planeDictionary = Engine::Tracer::ImageMethodSolver::buildPlaneMapFromCloud(rawCloud);
		std::cout << "  [PASS] 成功提炼出 " << planeDictionary.size() << " 个绝对平整的物理平面！\n";

		// 初始化光追核心管线
		int32_t targetInstanceId = sceneMeshes.begin()->first;
		OptixTraversableHandle traceHandle = geometryManager.getGasHandle(targetInstanceId);
		Engine::Tracer::RayTracer tracer(optixManager.getContext(), traceHandle);
		tracer.initPipelineAndSBT("E:/RT_software/Clanguage/OptixCUDA/out/build/x64-Debug/Ptx/device_programs.ptx", geometryManager);

		// 将全局点云推入显存供射线使用
		Engine::Geometry::Point* d_globalCloud = nullptr;
		size_t cloudBytes = rawCloud.size() * sizeof(Engine::Geometry::Point);
		CUDA_CHECK(cudaMalloc(&d_globalCloud, cloudBytes));
		CUDA_CHECK(cudaMemcpy(d_globalCloud, rawCloud.data(), cloudBytes, cudaMemcpyHostToDevice));

#ifdef ENABLE_ENGINE_DEBUG
		// 单步诊断模块 (仅在调试模式下可用)
		runSBRDiagnostics(false, sceneMeshes, tracer, d_globalCloud);
#endif

		// [阶段一] OptiX SBR 粗搜拓扑 (Topology Search)
		size_t testRayCount = 20000000;
		std::cout << "\n>>> [TDD Check] 阶段一：开始批量推送 " << testRayCount << " 条射线至 OptiX 管线...\n";

		std::vector<float3> testRays = Engine::Tracer::RayGenerator::generateFibonacciSphere(testRayCount);
		std::vector<Engine::Tracer::PathTopology> candidateTopologies;

		// 定义收发机参数
		float3 tx_test = make_float3(9.0f, 6.0f, 1.0f);
		float3 rx_test = make_float3(1.0f, 2.2f, 1.0f);
		float rx_r = 0.5f;

		// 发射加特林机枪
		tracer.shootRaysBatchSBR(
			testRays,
			tx_test.x, tx_test.y, tx_test.z,
			rx_test.x, rx_test.y, rx_test.z, rx_r,
			candidateTopologies,
			d_globalCloud
		);

		// 提取唯一拓扑
		auto uniqueTopologies = extractUniqueTopologies(candidateTopologies);

		// [阶段二] 纯 CUDA 镜像法精确解析 (Image Method Optimization)
		std::vector<Engine::Tracer::ExactPath> finalPaths =
			Engine::Tracer::ImageMethodSolver::solvePathsGPU(uniqueTopologies, planeDictionary, tx_test, rx_test);

		int validCount = 0;
		for (const auto& p : finalPaths) {
			if (p.isValid) validCount++;
		}
		std::cout << "  [PASS] 并发结算完成！共计算 " << finalPaths.size() << " 条多径，严丝合缝的绝对有效路径有 " << validCount << " 条！\n";

		// [阶段三] OptiX 视距遮挡判决 (Shadow Ray Validation)
		tracer.validatePathsOptiX(finalPaths);

		int ultimateValidCount = 0;
		for (const auto& p : finalPaths) {
			if (p.isValid) ultimateValidCount++;
		}
		std::cout << "  [PASS] 终极判决完成！没有被遮挡的 [物理多径] 有: " << ultimateValidCount << " 条！\n";

		// [新增] 导出纯金物理多径到 OBJ 文件，供 Python 渲染验证
		Engine::Debug::PathExporter::exportPathsToOBJ(finalPaths, "debug_ray_paths.obj");
		// 清理在 main 中申请的显存
		CUDA_CHECK(cudaFree(d_globalCloud));
	}
	catch (const std::exception& e) {
		std::cerr << "Engine Error: " << e.what() << std::endl;
	}
	return 0;
}
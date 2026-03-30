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
#include <algorithm> // 添加此行以包含 std::sort 的定义

#include <unordered_set>

// 1. 定义拓扑序列的 Hash 函数
struct TopologyHash {
	std::size_t operator()(const Engine::Tracer::PathTopology& topo) const {
		std::size_t hash = 0;
		for (int i = 0; i < topo.nodeCount; ++i) {
			hash ^= std::hash<int32_t>()(topo.nodes[i].instance_id) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
			hash ^= std::hash<int32_t>()(topo.nodes[i].plane_label) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		}
		return hash;
	}
};

// 2. 定义序列相等判定
struct TopologyEqual {
	bool operator()(const Engine::Tracer::PathTopology& a, const Engine::Tracer::PathTopology& b) const {
		if (a.nodeCount != b.nodeCount) return false;
		for (int i = 0; i < a.nodeCount; ++i) {
			if (a.nodes[i].instance_id != b.nodes[i].instance_id ||
				a.nodes[i].plane_label != b.nodes[i].plane_label) return false;
		}
		return true;
	}
};
// ========================================================================
// 辅助工具：拓扑去重与子路径爆破处理器 (纯血 SBR+IM 核心)
// ========================================================================
std::vector<Engine::Tracer::PathTopology> extractUniqueTopologies(const std::vector<Engine::Tracer::PathTopology>& candidateTopologies) {
	std::unordered_set<Engine::Tracer::PathTopology, TopologyHash, TopologyEqual> uniqueSet;

	// 直射 (LOS) 永远作为一条必选拓扑加入候选名单
	Engine::Tracer::PathTopology losTopo = {};
	losTopo.nodeCount = 0;
	uniqueSet.insert(losTopo);

	for (const auto& topo : candidateTopologies) {
		// SBR+IM 核心奥义：射线每弹跳一次，都可能连通 Rx！
		// 只要节点数大于 0 (即击中过真实的平面)，就把沿途的每一次撞击都拆解为独立的候选路径
		if (topo.nodeCount > 0) {
			Engine::Tracer::PathTopology subTopo = {};
			for (int i = 0; i < topo.nodeCount; ++i) {
				subTopo.nodes[i] = topo.nodes[i];
				subTopo.nodeCount = i + 1;
				uniqueSet.insert(subTopo); // unordered_set 会自动抛弃重复序列
			}
		}
	}

	std::cout << "  -> 去重完成：从千万级射线中提取出 " << uniqueSet.size() << " 条【绝对唯一】的物理拓扑序列" << std::endl;
	return std::vector<Engine::Tracer::PathTopology>(uniqueSet.begin(), uniqueSet.end());
}
// ========================================================================
// 引擎调试开关 (如果需要开启导出 OBJ 或单步诊断，取消下一行的注释即可)
// 或者推荐在 CMakeLists.txt 中通过 target_compile_definitions 全局配置
// ========================================================================
// #define ENABLE_ENGINE_DEBUG

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

int main() {
	try {
		// ========================================================
		// [阶段零] 引擎初始化与物理场景构建 (Infrastructure)
		// ========================================================
		Engine::Core::OptixContextManager optixManager;
		Engine::Core::GeometryManager geometryManager(optixManager.getContext());
		Engine::Core::SceneManager sceneManager(optixManager.getContext());
		Engine::Geometry::PointCloudConverter meshConverter(0.3f);

		std::vector<Engine::Geometry::Point> rawCloud;
		std::unordered_map<int32_t, std::vector<Engine::Geometry::TriangleMesh>> sceneMeshes;

		std::cout << "[System] 正在加载点云数据..." << std::endl;
		if (meshConverter.loadFromBinaryPLY("E:/RT_software/Clanguage/sy.ply", rawCloud)) {
			sceneMeshes = meshConverter.convertToMeshes(rawCloud);

			// 耗时操作：导出 OBJ 仅在调试模式下执行
			Engine::Debug::Exporter3D::exportMeshesToOBJ(sceneMeshes, "debug_greedy_meshes.obj");

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

		auto localPlaneDict_TEST = Engine::Tracer::ImageMethodSolver::buildLocalPlaneMapFromCloud_TEST(rawCloud);

		// ==================== [修复后] ====================
		// 从 sceneManager 中获取包含全场景的顶级 IAS 句柄
		// (注意：这里假设您的获取函数叫 getIASHandle()，如果叫 getRootHandle() 或其他名字请自行替换)
		OptixTraversableHandle traceHandle = sceneManager.getIasHandle();
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
		float3 tx_test = make_float3(26.5f, 4.0f, 2.0f);
		float3 rx_test = make_float3(25.5f, 12.2f, 2.0f);
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
			Engine::Tracer::ImageMethodSolver::solvePathsGPU_TEST(uniqueTopologies, localPlaneDict_TEST, tx_test, rx_test);

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

		// ========================================================================
			// [TDD 终极核查] 按反射阶数排序，全景打印 Label 序列与途经坐标
			// ========================================================================
		std::cout << "\n>>> [Debug] 最终存活多径全景核查 (按反射阶数排序)：\n";

		// 1. 提取有效路径的索引，并按反射阶数（顶点数）升序排序
		std::vector<size_t> sortedIndices;
		for (size_t i = 0; i < finalPaths.size(); ++i) {
			if (finalPaths[i].isValid) sortedIndices.push_back(i);
		}
		std::sort(sortedIndices.begin(), sortedIndices.end(), [&](size_t a, size_t b) {
			return finalPaths[a].vertexCount < finalPaths[b].vertexCount;
			});

		// 2. 格式化打印每一条路径的完整履历
		int pathIndex = 0;
		for (size_t idx : sortedIndices) {
			pathIndex++;
			const auto& path = finalPaths[idx];
			const auto& topo = uniqueTopologies[idx];

			// 反射阶数 = 顶点数 - 2 (减去 Tx 和 Rx)
			int bounces = path.vertexCount - 2;

			std::cout << "  [Path " << pathIndex << " | " << bounces << " 阶] ";
			std::cout << "Label序列: [ ";
			for (int j = 0; j < topo.nodeCount; ++j) {
				std::cout << topo.nodes[j].plane_label << " ";
			}
			std::cout << "] \n    -> 途经坐标: ";

			// 打印所有的中间反射点
			for (int j = 1; j < path.vertexCount - 1; ++j) {
				std::cout << "(" << path.vertices[j].x << ", "
					<< path.vertices[j].y << ", "
					<< path.vertices[j].z << ")";
				if (j < path.vertexCount - 2) std::cout << " -> ";
			}
			std::cout << "\n\n";
		}
		std::cout << ">>> ------------------------------------------------\n\n";
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
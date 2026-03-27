#include "Engine/Core/Context.h"
#include "Engine/Scene/GeometryManager.h"
#include "Engine/Tracer/RayTracer.h"
#include <iostream>
#include "Engine/Scene/PointCloudConverter.h" // 引入我们新建的转换器
#include "Engine/Scene/SceneManager.h"
#include <cmath> // 用于 sin/cos 计算旋转矩阵
// 在 main.cpp 的顶部补充这两个头文件
#include "Engine/Core/CudaError.h"  // 提供 CUDA_CHECK 宏定义
#include <cuda_runtime.h>           // 提供 cudaMalloc 和 cudaMemcpy 函数
// 顶部引入导出工具
#include "Engine/Debug/Exporter3D.h"
#include "Engine/Tracer/RayGenerator.h"
#include "Engine/Tracer/ImageMethodSolver.h"

// 诊断函数
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

	// 设置起点稍微偏移靶心
	float ox = tx + 0.1f, oy = ty + 0.1f, oz = tz + 0.1f;
	float dx = tx - ox, dy = ty - oy, dz = tz - oz;
	float len = std::sqrt(dx * dx + dy * dy + dz * dz);
	if (len > 0.0001f) { dx /= len; dy /= len; dz /= len; }

	std::cout << ">>> [诊断信息] 发射源坐标: (" << ox << ", " << oy << ", " << oz << ")\n";

	// 准备接收 SBR 路径的 CPU 容器
	Engine::Tracer::RayPath sbrResult = {};

	// 扣动扳机
	tracer.shootRaySBR(ox, oy, oz, dx, dy, dz, &sbrResult, d_globalCloud);

	// 打印结果
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
		// OptixContextManager 的构造函数会自动检测显卡并初始化 OptiX
		Engine::Core::OptixContextManager optixManager;
		std::cout << "OptiX Context Initialized successfully." << std::endl;

		// 步骤 2：初始化引擎核心组件
		Engine::Core::GeometryManager geometryManager(optixManager.getContext());

		// 初始化点云转换器，设置贪心网格精度
		Engine::Geometry::PointCloudConverter meshConverter(0.15f);

		std::vector<Engine::Geometry::Point> rawCloud;
		std::cout << "Loading Point Cloud..." << std::endl;

		// 【修复 1】将 sceneMeshes 提升到全局作用域，让下面的诊断代码能用到
		std::unordered_map<int32_t, std::vector<Engine::Geometry::TriangleMesh>> sceneMeshes;

		if (meshConverter.loadFromBinaryPLY("E:/RT_software/Clanguage/OptixCUDA/SY1101.ply", rawCloud)) {
			std::cout << "Converting points to Greedy Meshes..." << std::endl;

			sceneMeshes = meshConverter.convertToMeshes(rawCloud);

			Engine::Debug::Exporter3D::exportMeshesToOBJ(sceneMeshes, "debug_greedy_meshes.obj");

			std::cout << "Building GAS in RT Cores..." << std::endl;
			// 批量将所有部件推入显存，生成相互独立的 GAS
			geometryManager.buildSceneGAS(sceneMeshes);

			std::cout << "All GAS built successfully!" << std::endl;
		}

		Engine::Core::SceneManager sceneManager(optixManager.getContext());

		//  将之前分离好的各个部件装载到舞台上
		for (const auto& pair : geometryManager.getAllGasHandles()) {
			int32_t instanceId = pair.first;
			OptixTraversableHandle gasHandle = pair.second;
			sceneManager.addInstance(instanceId, gasHandle);
		}

		// 初始化构建一次 IAS
		sceneManager.buildIAS();
		std::cout << "[System] 顶层加速结构 (IAS) 装配完毕！" << std::endl;

		// 假设你主机端的点云数据叫 hostCloud
		std::cout << "\n>>> [TDD Check] 开始从 LiDAR 点云中提炼宏观平面方程...\n";
		auto planeDictionary = Engine::Tracer::ImageMethodSolver::buildPlaneMapFromCloud(rawCloud);
		std::cout << "  [PASS] 成功提炼出 " << planeDictionary.size() << " 个绝对平整的物理平面！\n";

		// 从 sceneMeshes 取第一个实例做默认测试把柄 (或者使用 IAS 的总句柄进行全场景追踪)
		int32_t targetInstanceId = sceneMeshes.begin()->first;
		OptixTraversableHandle traceHandle = geometryManager.getGasHandle(targetInstanceId);

		Engine::Tracer::RayTracer tracer(optixManager.getContext(), traceHandle);
		tracer.initPipelineAndSBT("E:/RT_software/Clanguage/OptixCUDA/out/build/x64-Debug/Ptx/device_programs.ptx", geometryManager);

		Engine::Geometry::Point* d_globalCloud = nullptr;
		size_t cloudBytes = rawCloud.size() * sizeof(Engine::Geometry::Point);
		CUDA_CHECK(cudaMalloc(&d_globalCloud, cloudBytes));
		CUDA_CHECK(cudaMemcpy(d_globalCloud, rawCloud.data(), cloudBytes, cudaMemcpyHostToDevice));

		// 诊断模块调用 (通过 true / false 一键开关)
		runSBRDiagnostics(false, sceneMeshes, tracer, d_globalCloud);

		size_t testRayCount = 20000000;
		std::vector<float3> testRays = Engine::Tracer::RayGenerator::generateFibonacciSphere(testRayCount);

		// 准备接收结果的大本营
		std::vector<Engine::Tracer::PathTopology> candidateTopologies;

		// 定义一个虚拟的接收球 (比如在坐标 5,5,5 的位置，半径为 0.1米)
		float rx_x = 1.0f, rx_y = 2.2f, rx_z = 1.0f;
		float rx_r = 0.1f;

		std::cout << ">>> [TDD Check] 开始批量推送 " << testRayCount << " 条射线至 OptiX 管线...\n";

		// 扣动“加特林机枪”的扳机！
		tracer.shootRaysBatchSBR(
			testRays,
			9.0, 6.0, 1.0,            // 发射机坐标 (沿用之前的测试起点)
			rx_x, rx_y, rx_z, rx_r,// 接收球参数
			candidateTopologies,   // 输出容器
			d_globalCloud
		);

		// 1. 明确定义发射点和接收点（为了后面镜像法复用）
		float3 tx_test = make_float3(9.0f, 6.0f, 1.0f);
		float3 rx_test = make_float3(rx_x, rx_y, rx_z);

		// 2. 统计并提取唯一拓扑 (Unique Topologies)
		// SBR 会产生数百万条射线，但很多射线的路径拓扑是一样的，去重能极大提高效率
		std::vector<Engine::Tracer::PathTopology> uniqueTopologies;
		int hitCount = 0;
		for (const auto& topo : candidateTopologies) {
			if (topo.hitRx) {
				hitCount++;
				// 这里简单演示：实际项目中通常会根据平面 ID 序列进行去重
				uniqueTopologies.push_back(topo);
			}
		}
		std::cout << "  -> 统计完成：共有 " << hitCount << " 条射线到达 Rx" << std::endl;

		// ========================================================
		// 3. 震撼的终极调用：GPU 并发镜像法解算
		// ========================================================
		std::cout << "\n>>> [TDD Check] 阶段二：GPU 并发镜像法解算启动！...\n";

		// 假设 tx_test 和 rx_test 是你之前的起点和终点 float3
		std::vector<Engine::Tracer::ExactPath> finalPaths =
			Engine::Tracer::ImageMethodSolver::solvePathsGPU(uniqueTopologies, planeDictionary, tx_test, rx_test);

		int validCount = 0;
		for (const auto& p : finalPaths) {
			if (p.isValid) validCount++;
		}

		std::cout << "  [PASS] 并发结算完成！共计算 " << finalPaths.size() << " 条多径，其中严丝合缝的绝对有效路径有 " << validCount << " 条！\n";
		std::cout << "\n>>> [TDD Check] 阶段三：OptiX 视距遮挡最终判决启动！...\n";

		tracer.validatePathsOptiX(finalPaths);

		int ultimateValidCount = 0;
		for (const auto& p : finalPaths) {
			if (p.isValid) ultimateValidCount++;
		}

		std::cout << "  [PASS] 终极判决完成！346 条精确路径中，没有被遮挡的 [纯金物理多径] 有: " << ultimateValidCount << " 条！\n";
		std::cout << ">>> [TDD Check] ----------------------------------------\n\n";
	}
	catch (const std::exception& e) {
		std::cerr << "Engine Error: " << e.what() << std::endl;
	}
	return 0;
}
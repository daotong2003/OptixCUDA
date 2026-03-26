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

int main() {
	try {
		// ==========================================
		// 步骤 1：启动底层硬件上下文
		// ==========================================
		// OptixContextManager 的构造函数会自动检测显卡并初始化 OptiX
		Engine::Core::OptixContextManager optixManager;
		std::cout << "OptiX Context Initialized successfully." << std::endl;

		// ==========================================
		// 步骤 2：初始化引擎核心组件
		// ==========================================
		// 将解包后的 context 句柄注入给 GeometryManager
		Engine::Core::GeometryManager geometryManager(optixManager.getContext());

		// 初始化点云转换器，设置贪心网格精度 (例如 0.055)
		Engine::Geometry::PointCloudConverter meshConverter(0.15f);

		// ==========================================
		// 步骤 3：数据加载与降维转换 (Host 端)
		// ==========================================
		std::vector<Engine::Geometry::Point> rawCloud;
		std::cout << "Loading Point Cloud..." << std::endl;

		// 【修复 1】将 sceneMeshes 提升到全局作用域，让下面的诊断代码能用到
		std::unordered_map<int32_t, std::vector<Engine::Geometry::TriangleMesh>> sceneMeshes;

		if (meshConverter.loadFromBinaryPLY("E:/RT_software/Clanguage/OptixCUDA/SY1101.ply", rawCloud)) {
			std::cout << "Converting points to Greedy Meshes..." << std::endl;

			sceneMeshes = meshConverter.convertToMeshes(rawCloud);

			Engine::Debug::Exporter3D::exportMeshesToOBJ(sceneMeshes, "debug_greedy_meshes.obj");

			// ==========================================
			// 步骤 4：硬件加速结构构建 (Device 端)
			// ==========================================
			std::cout << "Building GAS in RT Cores..." << std::endl;
			// 批量将所有部件推入显存，生成相互独立的 GAS
			geometryManager.buildSceneGAS(sceneMeshes);

			std::cout << "All GAS built successfully!" << std::endl;
		}

		Engine::Core::SceneManager sceneManager(optixManager.getContext());

		// 1. 将之前分离好的各个部件装载到舞台上
		for (const auto& pair : geometryManager.getAllGasHandles()) {
			int32_t instanceId = pair.first;
			OptixTraversableHandle gasHandle = pair.second;
			sceneManager.addInstance(instanceId, gasHandle);
		}

		// 初始化构建一次 IAS
		sceneManager.buildIAS();
		std::cout << "[System] 顶层加速结构 (IAS) 装配完毕！" << std::endl;

		// ==========================================================
		// [TDD 验证] 终极诊断：贴脸瞄准与黑盒解密
		// ==========================================================
		std::cout << "\n>>> [TDD Check] ----------------------------------------\n";
		std::cout << ">>> [TDD Check] 开始执行终极诊断流程...\n";

		// 【排查 1】确保网格字典里有数据
		auto firstInstance = sceneMeshes.begin(); // 假设你转换点云后存入了这个变量
		while (firstInstance != sceneMeshes.end() && firstInstance->second.empty()) {
			firstInstance++;
		}
		if (firstInstance == sceneMeshes.end()) {
			std::cout << ">>> [致命错误] sceneMeshes 完全为空！请检查点云转换流程。\n";
			return -1;
		}

		int32_t targetInstanceId = firstInstance->first;
		auto& testMesh = firstInstance->second.front();

		// 【排查 2】检查显卡底层的 GAS 句柄是否合法
		OptixTraversableHandle traceHandle = geometryManager.getGasHandle(targetInstanceId);
		std::cout << ">>> [诊断信息] 目标 Instance ID: " << targetInstanceId << "\n";
		std::cout << ">>> [诊断信息] 获取到的 GAS Handle 值为: " << traceHandle << "\n";
		if (traceHandle == 0) {
			std::cout << ">>> [致命错误] Handle 为 0！说明底层的 OptiX GAS 加速结构构建失败或已被销毁！\n";
		}

		// 初始化 Tracer
		Engine::Tracer::RayTracer tracer(optixManager.getContext(), traceHandle);
		tracer.initPipelineAndSBT("E:/RT_software/Clanguage/OptixCUDA/out/build/x64-Debug/Ptx/device_programs.ptx", geometryManager);

		// 将全局点云推入显存
		Engine::Geometry::Point* d_globalCloud = nullptr;
		size_t cloudBytes = rawCloud.size() * sizeof(Engine::Geometry::Point);
		CUDA_CHECK(cudaMalloc(&d_globalCloud, cloudBytes));
		CUDA_CHECK(cudaMemcpy(d_globalCloud, rawCloud.data(), cloudBytes, cudaMemcpyHostToDevice));

		// 【排查 3】提取并打印真实的三角形坐标，防止虚空打靶
		float3 v0 = testMesh.vertices[testMesh.indices[0].x];
		float3 v1 = testMesh.vertices[testMesh.indices[0].y];
		float3 v2 = testMesh.vertices[testMesh.indices[0].z];

		float tx = (v0.x + v1.x + v2.x) / 3.0f;
		float ty = (v0.y + v1.y + v2.y) / 3.0f;
		float tz = (v0.z + v1.z + v2.z) / 3.0f;

		std::cout << ">>> [诊断信息] 提取网格中的第 1 个真实三角形:\n"
			<< "      v0: (" << v0.x << ", " << v0.y << ", " << v0.z << ")\n"
			<< "      v1: (" << v1.x << ", " << v1.y << ", " << v1.z << ")\n"
			<< "      v2: (" << v2.x << ", " << v2.y << ", " << v2.z << ")\n"
			<< "      绝对靶心: (" << tx << ", " << ty << ", " << tz << ")\n";

		// 【排查 4】贴脸开大！(Point-Blank Range)
		// 如果从 (0,0,0) 发射，距离太远可能会因为浮点精度误差导致射线擦过边缘。
		// 我们把射线起点强行设置在距离靶心仅 0.1 单位的地方！
		float ox = tx + 0.1f;
		float oy = ty + 0.1f;
		float oz = tz + 0.1f;

		float dx = tx - ox;
		float dy = ty - oy;
		float dz = tz - oz;
		float len = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (len > 0.0001f) { dx /= len; dy /= len; dz /= len; }

		std::cout << ">>> [诊断信息] 正在从坐标 (" << ox << ", " << oy << ", " << oz << ") 贴脸射击靶心...\n";

		// 扣动扳机
		tracer.shootRay(ox, oy, oz, dx, dy, dz, d_globalCloud);

		// Engine::Debug::Exporter3D::exportMeshesToOBJ(sceneMeshes, "debug_greedy_meshes.obj");

		CUDA_CHECK(cudaFree(d_globalCloud));
		std::cout << ">>> [TDD Check] ----------------------------------------\n\n";
		// ==========================================================

		// =====================================
		// 模拟主循环：让特定部件动起来！
		// =====================================
		float angle = 0.0f;
		for (int frame = 0; frame < 5; ++frame) {
			angle += 0.05f;

			// 构造一个绕 Y 轴旋转，并沿 X 轴平移的矩阵
			float s = std::sin(angle);
			float c = std::cos(angle);

			float movingMatrix[12] = {
				c,    0.0f, s,    5.0f * c, // 平移 X
				0.0f, 1.0f, 0.0f, 0.0f,
			   -s,    0.0f, c,    0.0f
			};

			// 假设 instance_id = 1 是一个可以移动的小车或部件
			sceneManager.updateTransform(1, movingMatrix);

			// 调用此函数，OptiX 会瞬间应用新的矩阵！
			sceneManager.buildIAS();

			std::cout << "Frame " << frame << " rendered." << std::endl;
		}

		// ==========================================
		// 步骤 5：光追主循环...
		// ==========================================
		// while (!window.shouldClose()) { ... }
	}
	catch (const std::exception& e) {
		std::cerr << "Engine Error: " << e.what() << std::endl;
	}

	// 离开作用域时，RAII 机制会完美工作：
	// 1. geometryManager 析构，清理所有 GAS 树的显存 (d_vertices, d_indices 等)
	// 2. optixManager 析构，安全销毁 OptiX 设备上下文
	return 0;
}
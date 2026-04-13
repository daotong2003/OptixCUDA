#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>
#include <algorithm>   // <--- [新增] 加上这个头文件！
// 引入你的引擎头文件
#include "../Engine/Core/Context.h"
#include "../Engine/Scene/GeometryManager.h"
#include "../Engine/Tracer/RayTracer.h"
#include "../Engine/Scene/PointCloudConverter.h"
#include "../Engine/Scene/SceneManager.h"
#include "../Engine/Core/CudaError.h"
#include "../Engine/Tracer/RayGenerator.h"
#include "../Engine/Tracer/ImageMethodSolver.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Engine;

// ========================================================================
// 辅助工具：提取唯一的物理拓扑序列 (从你 main.cpp 迁移过来)
// ========================================================================
struct TopologyHash {
	std::size_t operator()(const Tracer::PathTopology& topo) const {
		std::size_t hash = 0;
		for (int i = 0; i < topo.nodeCount; ++i) {
			hash ^= std::hash<int32_t>()(topo.nodes[i].plane_label) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		}
		return hash;
	}
};

struct TopologyEqual {
	bool operator()(const Tracer::PathTopology& a, const Tracer::PathTopology& b) const {
		if (a.nodeCount != b.nodeCount) return false;
		for (int i = 0; i < a.nodeCount; ++i) {
			if (a.nodes[i].plane_label != b.nodes[i].plane_label) return false;
		}
		return true;
	}
};

std::vector<Tracer::PathTopology> extractUniqueTopologies(const std::vector<Tracer::PathTopology>& candidateTopologies) {
	std::unordered_set<Tracer::PathTopology, TopologyHash, TopologyEqual> uniqueSet;
	Tracer::PathTopology losTopo = {};
	losTopo.nodeCount = 0;
	uniqueSet.insert(losTopo);

	for (const auto& topo : candidateTopologies) {
		if (topo.nodeCount > 0) {
			Tracer::PathTopology subTopo = {};
			for (int i = 0; i < topo.nodeCount; ++i) {
				subTopo.nodes[i] = topo.nodes[i];
				subTopo.nodeCount = i + 1;
				uniqueSet.insert(subTopo);
			}
		}
	}
	return std::vector<Tracer::PathTopology>(uniqueSet.begin(), uniqueSet.end());
}

// ========================================================================
// 核心桥接类：管理 OptiX 生命周期，并提供零拷贝 Python 接口
// ========================================================================
class OptixEngineBridge {
private:
	std::unique_ptr<Core::OptixContextManager> optixManager;
	std::unique_ptr<Core::GeometryManager> geometryManager;
	std::unique_ptr<Core::SceneManager> sceneManager;
	std::unique_ptr<Tracer::RayTracer> tracer;

	std::unordered_map<int32_t, Tracer::LocalPlaneDictEntry> localPlaneDict;
	Geometry::Point* d_globalCloud = nullptr;

public:
	// 构造函数：执行原本 main.cpp 里的 [阶段零] 引擎初始化与物理场景构建
	OptixEngineBridge(const std::string& ply_path, const std::string& ptx_path) {
		std::cout << ">>> [Bridge] 初始化 OptiX 引擎环境...\n";
		optixManager = std::make_unique<Core::OptixContextManager>();
		geometryManager = std::make_unique<Core::GeometryManager>(optixManager->getContext());
		sceneManager = std::make_unique<Core::SceneManager>(optixManager->getContext());

		Geometry::PointCloudConverter meshConverter(0.3f);
		std::vector<Geometry::Point> rawCloud;

		if (!meshConverter.loadFromBinaryPLY(ply_path, rawCloud)) {
			throw std::runtime_error("Failed to load PLY file!");
		}

		auto sceneMeshes = meshConverter.convertToMeshes(rawCloud);
		geometryManager->buildSceneGAS(sceneMeshes);

		// 装载部件并构建 IAS
		std::vector<int32_t> sorted_instances;
		for (const auto& pair : geometryManager->getAllGasHandles()) {
			sorted_instances.push_back(pair.first);
		}
		std::sort(sorted_instances.begin(), sorted_instances.end());

		uint32_t current_sbt_offset = 0;
		const auto& allGasRecords = geometryManager->getGasRecords();
		for (int32_t inst_id : sorted_instances) {
			sceneManager->addInstance(inst_id, geometryManager->getGasHandle(inst_id), current_sbt_offset);
			current_sbt_offset += allGasRecords.at(inst_id).d_vertices_list.size();
		}
		sceneManager->buildIAS();

		localPlaneDict = Tracer::ImageMethodSolver::buildLocalPlaneMapFromCloud_TEST(rawCloud);

		OptixTraversableHandle traceHandle = sceneManager->getIasHandle();
		tracer = std::make_unique<Tracer::RayTracer>(optixManager->getContext(), traceHandle);
		tracer->initPipelineAndSBT(ptx_path, *geometryManager);

		// 推入显存
		size_t cloudBytes = rawCloud.size() * sizeof(Geometry::Point);
		CUDA_CHECK(cudaMalloc(&d_globalCloud, cloudBytes));
		CUDA_CHECK(cudaMemcpy(d_globalCloud, rawCloud.data(), cloudBytes, cudaMemcpyHostToDevice));
		std::cout << ">>> [Bridge] 引擎就绪！随时可以接收 Python 计算请求。\n";
	}

	~OptixEngineBridge() {
		if (d_globalCloud) cudaFree(d_globalCloud);
	}

	// 暴露给 Python 的计算接口：执行 SBR 追踪并返回 ExactPath 字节流
	nb::ndarray<nb::numpy, uint8_t> compute_paths(
		float tx_x, float tx_y, float tx_z,
		float rx_x, float rx_y, float rx_z,
		float rx_r, int num_rays)
	{
		std::vector<float3> testRays = Tracer::RayGenerator::generateFibonacciSphere(num_rays);
		std::vector<Tracer::PathTopology> candidateTopologies;

		// [阶段一] OptiX SBR 粗搜拓扑
		tracer->shootRaysBatchSBR(testRays, tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, rx_r, candidateTopologies, d_globalCloud);
		auto uniqueTopologies = extractUniqueTopologies(candidateTopologies);

		// [阶段二] 纯 CUDA 镜像法精确解析
		float3 tx = make_float3(tx_x, tx_y, tx_z);
		float3 rx = make_float3(rx_x, rx_y, rx_z);
		auto finalPaths = Tracer::ImageMethodSolver::solvePathsGPU_TEST(uniqueTopologies, localPlaneDict, tx, rx);

		// [阶段三] OptiX 视距遮挡判决
		tracer->validatePathsOptiX(finalPaths);

		// 提取绝对有效的物理多径，推入堆内存，供 Python 零拷贝接管
		auto* valid_paths = new std::vector<Tracer::ExactPath>();
		for (const auto& p : finalPaths) {
			if (p.isValid) {
				valid_paths->push_back(p);
			}
		}

		nb::capsule deleter(valid_paths, [](void* p) noexcept {
			delete static_cast<std::vector<Tracer::ExactPath>*>(p);
			});

		size_t total_bytes = valid_paths->size() * sizeof(Tracer::ExactPath);

		return nb::ndarray<nb::numpy, uint8_t>(
			valid_paths->data(), { total_bytes }, deleter
		);
	}
};

// ========================================================================
// 模块注册
// ========================================================================
NB_MODULE(optix_backend, m) {
	nb::class_<OptixEngineBridge>(m, "OptixEngineBridge")
		.def(nb::init<const std::string&, const std::string&>(), "ply_path"_a, "ptx_path"_a, "初始化引擎与点云缓存")
		.def("compute_paths", &OptixEngineBridge::compute_paths,
			"tx_x"_a, "tx_y"_a, "tx_z"_a,
			"rx_x"_a, "rx_y"_a, "rx_z"_a,
			"rx_r"_a, "num_rays"_a,
			"执行 SBR+IM 三阶段追踪并返回 Zero-Copy Numpy 字节流");
}
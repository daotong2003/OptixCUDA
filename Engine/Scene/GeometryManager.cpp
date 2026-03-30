#include "GeometryManager.h"
#include "../Core/CudaError.h" // 引入您引擎的 CUDA 错误检查宏，例如 CUDA_CHECK, OPTIX_CHECK
#include <cuda_runtime.h>
#include <iostream>

// [调试工具] 匿名命名空间：仅在当前 .cpp 文件内可见的辅助函数
// ==========================================================
namespace {
	// 显存数据回读对比 (Round-Trip Test)
	// 传入原始的 CPU 数组、显存指针、以及拷贝大小
	inline void verifyVRAMRoundTrip(const std::vector<uint32_t>& cpuCounts, void* d_counts_ptr, size_t countsSize) {
		// 使用静态变量确保全局只打印一次，防止刷屏
		static bool hasVerifiedVRAM = false;

		// 如果已经验证过，或者数据为空，则直接跳过
		if (hasVerifiedVRAM || cpuCounts.empty() || d_counts_ptr == nullptr) return;

		std::cout << "\n>>> [TDD Check] VRAM Round-Trip Verification Started...\n";

		// 1. 在 CPU 上创建一个全 0 的干净容器
		std::vector<uint32_t> testReadbackCounts(cpuCounts.size(), 0);

		// 2. 从显卡显存中把数据强行拽回 CPU
		CUDA_CHECK(cudaMemcpy(testReadbackCounts.data(), d_counts_ptr, countsSize, cudaMemcpyDeviceToHost));

		// 3. 打印对比前 5 个三角形的点数映射是否被正确记录在显存中
		bool isMatch = true;
		size_t checkCount = std::min<size_t>(5, cpuCounts.size());
		for (size_t k = 0; k < checkCount; ++k) {
			std::cout << "  Tri [" << k << "] point counts -> Original CPU: " << cpuCounts[k]
				<< " | Read back VRAM: " << testReadbackCounts[k] << "\n";
			if (cpuCounts[k] != testReadbackCounts[k]) isMatch = false;
		}

		std::cout << "  VRAM Transfer Status: " << (isMatch ? "[PASS]" : "[FAIL!]") << "\n\n";
		hasVerifiedVRAM = true; // 拦截后续校验
	}
}

namespace Engine {
	namespace Core {
		GeometryManager::GeometryManager(OptixDeviceContext context)
			: optixContext(context) {
		}

		GeometryManager::~GeometryManager() {
			// 严谨的显存防泄漏设计：遍历销毁所有 GAS 占用的显存
			for (auto& pair : gasRecords) {
				GasRecord& record = pair.second;

				for (void* d_v : record.d_vertices_list) {
					cudaFree(d_v);
				}
				for (void* d_i : record.d_indices_list) {
					cudaFree(d_i);
				}
				// [Step 3 新增] 严谨释放微观映射数组的显存
				for (void* d_o : record.d_pointOffsets_list) cudaFree(d_o);
				for (void* d_c : record.d_pointCounts_list) cudaFree(d_c);
				for (void* d_pi : record.d_pointIndices_list) cudaFree(d_pi);

				if (record.d_gas_output_buffer) {
					cudaFree(record.d_gas_output_buffer);
				}
			}
			gasRecords.clear();
			gasHandles.clear();
		}

		void GeometryManager::buildSceneGAS(const std::unordered_map<int32_t, std::vector<Geometry::TriangleMesh>>& meshesByInstance) {
			for (const auto& pair : meshesByInstance) {
				int32_t instance_id = pair.first;
				const std::vector<Geometry::TriangleMesh>& meshes = pair.second;

				// 构建单个部件的 GAS
				GasRecord record;
				OptixTraversableHandle handle = buildSingleGAS(meshes, record);

				// 保存句柄和显存记录
				gasHandles[instance_id] = handle;
				gasRecords[instance_id] = record;
			}
		}

		OptixTraversableHandle GeometryManager::getGasHandle(int32_t instance_id) const {
			auto it = gasHandles.find(instance_id);
			if (it != gasHandles.end()) {
				return it->second;
			}
			return 0; // 若未找到返回无效句柄
		}

		OptixTraversableHandle GeometryManager::buildSingleGAS(const std::vector<Geometry::TriangleMesh>& meshes, GasRecord& outRecord) {
			if (meshes.empty()) return 0;

			std::vector<OptixBuildInput> buildInputs;
			buildInputs.reserve(meshes.size());

			// 【核心修复】：创建一个局部数组，提前分配好固定大小。
			// 用来安全地存放所有 CUdeviceptr，确保我们在取地址 (&vertexBufferPtrs[i]) 时，
			// 地址绝对不会因为 vector 的动态扩容而失效！
			std::vector<CUdeviceptr> vertexBufferPtrs(meshes.size());

			uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

			for (size_t i = 0; i < meshes.size(); ++i) {
				const auto& mesh = meshes[i];
				if (mesh.vertices.empty() || mesh.indices.empty()) continue;

				// [新增] 将刚刚在 PointCloudConverter 中绑定的专属 Label 存入账本
				outRecord.plane_label_list.push_back(mesh.plane_label);

				// 1. 将顶点上传至显存
				size_t verticesSizeInBytes = mesh.vertices.size() * sizeof(float3);
				void* d_vertices;
				CUDA_CHECK(cudaMalloc(&d_vertices, verticesSizeInBytes));
				CUDA_CHECK(cudaMemcpy(d_vertices, mesh.vertices.data(), verticesSizeInBytes, cudaMemcpyHostToDevice));
				outRecord.d_vertices_list.push_back(d_vertices);

				// 【安全赋值】：将设备指针存入固定大小的数组
				vertexBufferPtrs[i] = (CUdeviceptr)d_vertices;

				// 2. 将索引上传至显存
				size_t indicesSizeInBytes = mesh.indices.size() * sizeof(uint3);
				void* d_indices;
				CUDA_CHECK(cudaMalloc(&d_indices, indicesSizeInBytes));
				CUDA_CHECK(cudaMemcpy(d_indices, mesh.indices.data(), indicesSizeInBytes, cudaMemcpyHostToDevice));
				outRecord.d_indices_list.push_back(d_indices);

				// ==================== [Step 3 新增] ====================
				// 3. 将微观映射数组推入显卡显存
				size_t offsetsSize = mesh.pointOffsets.size() * sizeof(uint32_t);
				if (offsetsSize > 0) {
					void* d_offsets;
					CUDA_CHECK(cudaMalloc(&d_offsets, offsetsSize));
					CUDA_CHECK(cudaMemcpy(d_offsets, mesh.pointOffsets.data(), offsetsSize, cudaMemcpyHostToDevice));
					outRecord.d_pointOffsets_list.push_back(d_offsets);
				}

				size_t countsSize = mesh.pointCounts.size() * sizeof(uint32_t);
				if (countsSize > 0) {
					void* d_counts;
					CUDA_CHECK(cudaMalloc(&d_counts, countsSize));
					CUDA_CHECK(cudaMemcpy(d_counts, mesh.pointCounts.data(), countsSize, cudaMemcpyHostToDevice));
					outRecord.d_pointCounts_list.push_back(d_counts);
				}

				size_t poolSize = mesh.pointIndices.size() * sizeof(uint32_t);
				if (poolSize > 0) {
					void* d_pt_indices;
					CUDA_CHECK(cudaMalloc(&d_pt_indices, poolSize));
					CUDA_CHECK(cudaMemcpy(d_pt_indices, mesh.pointIndices.data(), poolSize, cudaMemcpyHostToDevice));
					outRecord.d_pointIndices_list.push_back(d_pt_indices);

					// [TDD 验证] 调用独立的显存回读测试函数
					// ==========================================================
					// verifyVRAMRoundTrip(mesh.pointCounts, outRecord.d_pointCounts_list.back(), countsSize);
				}

				// 4. 配置 OptiX 的构建输入参数
				OptixBuildInput buildInput = {};
				buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

				buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				buildInput.triangleArray.numVertices = static_cast<uint32_t>(mesh.vertices.size());

				// 【核心修复】：从安全的、不会漂移的数组中取地址！
				buildInput.triangleArray.vertexBuffers = &vertexBufferPtrs[i];
				buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);

				buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				buildInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(mesh.indices.size());
				buildInput.triangleArray.indexBuffer = (CUdeviceptr)d_indices; // index 直接传值，天然安全
				buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);

				buildInput.triangleArray.flags = triangle_input_flags;
				buildInput.triangleArray.numSbtRecords = 1;

				buildInputs.push_back(buildInput);
			}

			if (buildInputs.empty()) return 0;

			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
				OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
				OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
			accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes bufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				optixContext,
				&accelOptions,
				buildInputs.data(),
				static_cast<uint32_t>(buildInputs.size()),
				&bufferSizes
			));

			void* d_temp_buffer = nullptr;
			CUDA_CHECK(cudaMalloc(&d_temp_buffer, bufferSizes.tempSizeInBytes));
			CUDA_CHECK(cudaMalloc(&outRecord.d_gas_output_buffer, bufferSizes.outputSizeInBytes));

			OptixTraversableHandle gasHandle = 0;
			OPTIX_CHECK(optixAccelBuild(
				optixContext,
				0,
				&accelOptions,
				buildInputs.data(),
				static_cast<uint32_t>(buildInputs.size()),
				(CUdeviceptr)d_temp_buffer,
				bufferSizes.tempSizeInBytes,
				(CUdeviceptr)outRecord.d_gas_output_buffer,
				bufferSizes.outputSizeInBytes,
				&gasHandle,
				nullptr,
				0
			));

			CUDA_CHECK(cudaFree(d_temp_buffer));
			CUDA_CHECK(cudaDeviceSynchronize());

			return gasHandle;
		}
	} // namespace Core
} // namespace Engine
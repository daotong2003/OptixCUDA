#include "RayTracer.h"
#include "../Core/CudaError.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "../Scene/GeometryManager.h" // [新增] 引入 GeometryManager
#include "../../NimbusCommon/SbtData.h" // [新增] 引入 SBT 数据结构
#include "../../NimbusCommon/LaunchParams.h"
#include "SbrTypes.h" // [新增] 引入 SBR 结构定义

namespace Engine {
	namespace Tracer {
		// ================== [补充构造与析构的显存分配] ==================
		RayTracer::RayTracer(OptixDeviceContext context, OptixTraversableHandle gasHandle)
			: optixContext(context), sceneHandle(gasHandle) {
			CUDA_CHECK(cudaMalloc((void**)&d_outHitStatus, sizeof(int)));
			CUDA_CHECK(cudaMalloc((void**)&d_outHitX, sizeof(float)));
			CUDA_CHECK(cudaMalloc((void**)&d_outHitY, sizeof(float)));
			CUDA_CHECK(cudaMalloc((void**)&d_outHitZ, sizeof(float)));

			// [Step 5 新增] 微观物理校准输出缓存
			CUDA_CHECK(cudaMalloc((void**)&d_outHitNormalX, sizeof(float)));
			CUDA_CHECK(cudaMalloc((void**)&d_outHitNormalY, sizeof(float)));
			CUDA_CHECK(cudaMalloc((void**)&d_outHitNormalZ, sizeof(float)));
			CUDA_CHECK(cudaMalloc((void**)&d_outHitMaterial, sizeof(uint8_t)));

			// ==================== [SBR 新增] ====================
			// 为整条射线路径分配一块连续的显存
			CUDA_CHECK(cudaMalloc((void**)&d_outSbrPath, sizeof(Engine::Tracer::RayPath)));
			// ========================================================

			CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Engine::LaunchParams)));
		}

#ifdef ENABLE_ENGINE_DEBUG
		// ================== [重构射线的装箱与解包逻辑] ==================
		void RayTracer::shootRay(float ox, float oy, float oz, float dx, float dy, float dz,
			Engine::Geometry::Point* d_globalCloud) {
			int initialStatus = 0;
			CUDA_CHECK(cudaMemcpy(d_outHitStatus, &initialStatus, sizeof(int), cudaMemcpyHostToDevice));

			Engine::LaunchParams hostParams = {};
			hostParams.handle = sceneHandle;
			hostParams.rayOrigin_x = ox; hostParams.rayOrigin_y = oy; hostParams.rayOrigin_z = oz;
			hostParams.rayDirection_x = dx; hostParams.rayDirection_y = dy; hostParams.rayDirection_z = dz;
			hostParams.tmax = 1000.0f;

			// 宏观输出绑定
			hostParams.outHitStatus = d_outHitStatus;
			hostParams.outHitPosition_x = d_outHitX;
			hostParams.outHitPosition_y = d_outHitY;
			hostParams.outHitPosition_z = d_outHitZ;

			// [Step 5 新增] 微观物理输出绑定
			hostParams.outHitNormal_x = d_outHitNormalX;
			hostParams.outHitNormal_y = d_outHitNormalY;
			hostParams.outHitNormal_z = d_outHitNormalZ;
			hostParams.outHitMaterial = d_outHitMaterial;

			// 2. 【核心绑定】：把外界传进来的全局点云显存指针，装进快递盒！
			hostParams.globalPointCloud = d_globalCloud;
			// 注意：确保此处的 hostParams.globalPointCloud 已经在外界被正确赋予了显存指针！
			// hostParams.globalPointCloud = globalCloudPtr;

			CUDA_CHECK(cudaMemcpy(d_params, &hostParams, sizeof(Engine::LaunchParams), cudaMemcpyHostToDevice));

			OPTIX_CHECK(optixLaunch(
				pipeline, 0, reinterpret_cast<CUdeviceptr>(d_params),
				sizeof(Engine::LaunchParams), &sbt, 1, 1, 1
			));
			CUDA_CHECK(cudaDeviceSynchronize());

			// 5. 拆快递，读取显卡传回来的混合结果
			int hitStatus = 0;
			CUDA_CHECK(cudaMemcpy(&hitStatus, d_outHitStatus, sizeof(int), cudaMemcpyDeviceToHost));

			if (hitStatus == 1) {
				float hx, hy, hz;
				float nx, ny, nz;
				uint8_t mat;

				CUDA_CHECK(cudaMemcpy(&hx, d_outHitX, sizeof(float), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(&hy, d_outHitY, sizeof(float), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(&hz, d_outHitZ, sizeof(float), cudaMemcpyDeviceToHost));

				// [Step 5 新增] 读取底层真实物理
				CUDA_CHECK(cudaMemcpy(&nx, d_outHitNormalX, sizeof(float), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(&ny, d_outHitNormalY, sizeof(float), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(&nz, d_outHitNormalZ, sizeof(float), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(&mat, d_outHitMaterial, sizeof(uint8_t), cudaMemcpyDeviceToHost));

				std::cout << "\n>>> [HIT] 射线命中了目标！\n"
					<< "  命中坐标: (" << hx << ", " << hy << ", " << hz << ")\n"
					<< "  真实法线: (" << nx << ", " << ny << ", " << nz << ")\n"
					<< "  材质标签: " << static_cast<int>(mat) << "\n";
			}
			else {
				std::cout << "\n>>> [MISS] 射线射向了太空，未命中任何物体。\n";
			}
		}
#endif

		RayTracer::~RayTracer() {
			auto safeFree = [](void* ptr) { if (ptr) cudaFree(ptr); };
			safeFree(d_outHitStatus); safeFree(d_outHitX);
			safeFree(d_outHitY); safeFree(d_outHitZ); safeFree(d_params);

			// ==================== [SBR 新增] ====================
			safeFree(d_outSbrPath);

			// [阶段一新增] 释放大数组显存
			safeFree(d_txRayDirections);
			safeFree(d_outCandidateTopologies);

			if (pipeline) optixPipelineDestroy(pipeline);
			if (raygenPG) optixProgramGroupDestroy(raygenPG);
			if (missPG) optixProgramGroupDestroy(missPG);
			if (hitgroupPG) optixProgramGroupDestroy(hitgroupPG);
			if (module) optixModuleDestroy(module);

			// 释放 SBT 显存
			safeFree(reinterpret_cast<void*>(sbt.raygenRecord));
			safeFree(reinterpret_cast<void*>(sbt.missRecordBase));
			safeFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase));
		}

		// ================== [SBR 专属追踪逻辑] ==================
		void RayTracer::shootRaySBR(float ox, float oy, float oz, float dx, float dy, float dz,
			Engine::Tracer::RayPath* out_hostPath,
			Engine::Geometry::Point* d_globalCloud) {
			// 1. 在发起光追前，清空显卡中的 SBR 路径缓存，确保不受上一帧的残留数据污染
			Engine::Tracer::RayPath emptyPath = {};
			emptyPath.nodeCount = 0;
			emptyPath.isEscaped = false; // 初始状态
			CUDA_CHECK(cudaMemcpy(d_outSbrPath, &emptyPath, sizeof(Engine::Tracer::RayPath), cudaMemcpyHostToDevice));

			// 2. 将参数装入快递盒 (LaunchParams)
			Engine::LaunchParams hostParams = {};
			hostParams.handle = sceneHandle;
			hostParams.rayOrigin_x = ox; hostParams.rayOrigin_y = oy; hostParams.rayOrigin_z = oz;
			hostParams.rayDirection_x = dx; hostParams.rayDirection_y = dy; hostParams.rayDirection_z = dz;
			hostParams.tmax = 1000.0f;

			// [SBR 专属绑定]
			hostParams.outSbrPath = d_outSbrPath;
			hostParams.maxBounceDepth = Engine::Tracer::MAX_BOUNCE_DEPTH;
			hostParams.globalPointCloud = d_globalCloud;

			CUDA_CHECK(cudaMemcpy(d_params, &hostParams, sizeof(Engine::LaunchParams), cudaMemcpyHostToDevice));

			// 3. 扣动扳机，启动 OptiX 射线追踪
			OPTIX_CHECK(optixLaunch(
				pipeline, 0, reinterpret_cast<CUdeviceptr>(d_params),
				sizeof(Engine::LaunchParams), &sbt, 1, 1, 1
			));
			CUDA_CHECK(cudaDeviceSynchronize()); // 必须等显卡算完

			// 4. 拆快递，把显卡算好的完整路径拷贝回 CPU 容器
			if (out_hostPath) {
				CUDA_CHECK(cudaMemcpy(out_hostPath, d_outSbrPath, sizeof(Engine::Tracer::RayPath), cudaMemcpyDeviceToHost));
			}
		}

		void RayTracer::initPipelineAndSBT(const std::string& ptxPath, const Engine::Core::GeometryManager& geometryManager)
		{
			// std::cout << "[Tracer] 正在读取 PTX 汇编文件: " << ptxPath << "\n";
			std::ifstream ptxFile(ptxPath, std::ios::ate | std::ios::binary);
			if (!ptxFile.is_open()) {
				throw std::runtime_error("找不到 PTX 文件！请检查路径。");
			}
			size_t fileSize = ptxFile.tellg();
			std::string ptxCode(fileSize, '\0');
			ptxFile.seekg(0);
			ptxFile.read(&ptxCode[0], fileSize);

			// ==========================================
			// 1. 创建 OptiX 模块 (Module)
			// ==========================================
			OptixModuleCompileOptions moduleCompileOptions = {};
			OptixPipelineCompileOptions pipelineCompileOptions = {};
			pipelineCompileOptions.usesMotionBlur = false;
			pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
			pipelineCompileOptions.numPayloadValues = 2; // 我们的“小书包”里只有 1 个 uint32_t
			pipelineCompileOptions.numAttributeValues = 2; // 三角形求交必备
			pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
			pipelineCompileOptions.pipelineLaunchParamsVariableName = "params"; // 必须与 .cu 中的变量名完全一致！

			OPTIX_CHECK(optixModuleCreate(
				optixContext, &moduleCompileOptions, &pipelineCompileOptions,
				ptxCode.c_str(), ptxCode.size(), nullptr, nullptr, &module
			));

			// ==========================================
			// 2. 创建程序组 (Program Groups)
			// ==========================================
			OptixProgramGroupOptions pgOptions = {};

			OptixProgramGroupDesc raygenDesc = {};
			raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygenDesc.raygen.module = module;
			raygenDesc.raygen.entryFunctionName = "__raygen__los"; // .cu 中的函数名
			OPTIX_CHECK(optixProgramGroupCreate(optixContext, &raygenDesc, 1, &pgOptions, nullptr, nullptr, &raygenPG));

			OptixProgramGroupDesc missDesc = {};
			missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			missDesc.miss.module = module;
			missDesc.miss.entryFunctionName = "__miss__los";
			OPTIX_CHECK(optixProgramGroupCreate(optixContext, &missDesc, 1, &pgOptions, nullptr, nullptr, &missPG));

			OptixProgramGroupDesc hitgroupDesc = {};
			hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroupDesc.hitgroup.moduleCH = module;
			hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__los";
			OPTIX_CHECK(optixProgramGroupCreate(optixContext, &hitgroupDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPG));

			// ==========================================
			// 3. 链接管线 (Pipeline)
			// ==========================================
			OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };
			OptixPipelineLinkOptions pipelineLinkOptions = {};
			pipelineLinkOptions.maxTraceDepth = 1; // LOS 只算 1 次，不考虑多次反射
			OPTIX_CHECK(optixPipelineCreate(
				optixContext, &pipelineCompileOptions, &pipelineLinkOptions,
				programGroups, 3, nullptr, nullptr, &pipeline
			));

			// ==========================================
			// 4. 构建着色器绑定表 (SBT)
			// ==========================================
			// --- 4.1 组装 Record (无专属数据) ---
			Engine::RaygenRecord rgRecord;
			OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rgRecord));
			CUdeviceptr d_rgRecord;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgRecord), sizeof(Engine::RaygenRecord)));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rgRecord), &rgRecord, sizeof(Engine::RaygenRecord), cudaMemcpyHostToDevice));
			sbt.raygenRecord = d_rgRecord;

			// --- 4.2 组装 Miss Record (无专属数据) ---
			Engine::MissRecord msRecord;
			OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &msRecord));
			CUdeviceptr d_msRecord;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_msRecord), sizeof(Engine::MissRecord)));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_msRecord), &msRecord, sizeof(Engine::MissRecord), cudaMemcpyHostToDevice));
			sbt.missRecordBase = d_msRecord;
			sbt.missRecordStrideInBytes = sizeof(Engine::MissRecord);
			sbt.missRecordCount = 1;

			// --- 4.3 [Step 4 核心] 动态组装 HitGroup Records ---
			std::vector<Engine::HitGroupRecord> hgRecords;

			// 从 GeometryManager 拿到我们在 Step 3 存好的所有显存大管家记录
			const auto& allGasRecords = geometryManager.getGasRecords();

			for (const auto& pair : allGasRecords) {
				int32_t instance_id = pair.first;
				const auto& record = pair.second;

				// 遍历该部件下的每一个三角形网格 (Mesh)
				for (size_t i = 0; i < record.d_vertices_list.size(); ++i) {
					Engine::HitGroupRecord hg_rec = {};

					// 打上 OptiX 内部的函数入口标记
					OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hg_rec));

					// 1. 塞入宏观网格指针
					hg_rec.data.vertices = reinterpret_cast<float3*>(record.d_vertices_list[i]);
					hg_rec.data.indices = reinterpret_cast<uint3*>(record.d_indices_list[i]);
					hg_rec.data.instance_id = instance_id;
					hg_rec.data.material_id = 0; // 默认材质，可后续拓展

					// 2. 塞入微观物理映射指针 (从 Step 3 的显卡地址中直接取)
					if (!record.d_pointOffsets_list.empty() && i < record.d_pointOffsets_list.size()) {
						hg_rec.data.pointOffsets = reinterpret_cast<uint32_t*>(record.d_pointOffsets_list[i]);
						hg_rec.data.pointCounts = reinterpret_cast<uint32_t*>(record.d_pointCounts_list[i]);
						hg_rec.data.pointIndices = reinterpret_cast<uint32_t*>(record.d_pointIndices_list[i]);
					}
					else {
						hg_rec.data.pointOffsets = nullptr;
						hg_rec.data.pointCounts = nullptr;
						hg_rec.data.pointIndices = nullptr;
					}

					hgRecords.push_back(hg_rec);
				}
			}

			// --- 4.4 将 HitGroup 数组一口气推入显存 ---
			CUdeviceptr d_hitgroup_records;
			size_t hgSize = sizeof(Engine::HitGroupRecord) * hgRecords.size();

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hgSize));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_hitgroup_records),
				hgRecords.data(),
				hgSize,
				cudaMemcpyHostToDevice
			));

			// 交接给管线的大使馆 (sbt)
			sbt.hitgroupRecordBase = d_hitgroup_records;
			sbt.hitgroupRecordStrideInBytes = sizeof(Engine::HitGroupRecord);
			sbt.hitgroupRecordCount = static_cast<uint32_t>(hgRecords.size());

			// std::cout << "[Tracer] 管线与 SBT 配置完成！共成功打包并上传了 "<< hgRecords.size() << " 个 HitGroup Record 携带独立数据。\n";
		}

		// ================== [阶段一：SBR 批量发射与拓扑粗搜] ==================
		void RayTracer::shootRaysBatchSBR(
			const std::vector<float3>& host_txRays,
			float tx_ox, float tx_oy, float tx_oz,
			float rx_ox, float rx_oy, float rx_oz, float rx_radius,
			std::vector<Engine::Tracer::PathTopology>& out_topologies,
			Engine::Geometry::Point* d_globalCloud)
		{
			size_t numRays = host_txRays.size();
			if (numRays == 0) return;

			// 1. 动态显存管理：如果当前射线数量超过了显存容量，就重新分配更大的空间
			if (numRays > allocatedRayCapacity) {
				if (d_txRayDirections) cudaFree(d_txRayDirections);
				if (d_outCandidateTopologies) cudaFree(d_outCandidateTopologies);

				CUDA_CHECK(cudaMalloc((void**)&d_txRayDirections, numRays * sizeof(float3)));
				CUDA_CHECK(cudaMalloc((void**)&d_outCandidateTopologies, numRays * sizeof(Engine::Tracer::PathTopology)));

				allocatedRayCapacity = numRays;
			}

			// 2. 把 CPU 算好的斐波那契射线束拷贝到显卡
			CUDA_CHECK(cudaMemcpy(d_txRayDirections, host_txRays.data(), numRays * sizeof(float3), cudaMemcpyHostToDevice));

			// 初始化输出数组（清空上一帧的残留拓扑）
			CUDA_CHECK(cudaMemset(d_outCandidateTopologies, 0, numRays * sizeof(Engine::Tracer::PathTopology)));

			// 3. 装箱：配置新的 LaunchParams
			Engine::LaunchParams hostParams = {};
			hostParams.handle = sceneHandle;
			hostParams.tmax = 1000.0f;
			hostParams.maxBounceDepth = Engine::Tracer::MAX_BOUNCE_DEPTH;
			hostParams.globalPointCloud = d_globalCloud;

			// Tx 配置
			hostParams.rayOrigin_x = tx_ox; hostParams.rayOrigin_y = tx_oy; hostParams.rayOrigin_z = tx_oz;
			hostParams.txRayDirections = d_txRayDirections;
			hostParams.numRays = static_cast<unsigned int>(numRays);

			// Rx 配置
			hostParams.rxPosition_x = rx_ox; hostParams.rxPosition_y = rx_oy; hostParams.rxPosition_z = rx_oz;
			hostParams.rxRadius = rx_radius;

			// 输出缓存配置
			hostParams.outCandidateTopologies = d_outCandidateTopologies;

			CUDA_CHECK(cudaMemcpy(d_params, &hostParams, sizeof(Engine::LaunchParams), cudaMemcpyHostToDevice));

			// 4. 万箭齐发！注意这里的并行宽度从 1 变成了 numRays！
			// OptiX 会在 GPU 上同时启动 numRays 个线程，每根射线独占一个线程！
			OPTIX_CHECK(optixLaunch(
				pipeline, 0, reinterpret_cast<CUdeviceptr>(d_params),
				sizeof(Engine::LaunchParams), &sbt,
				numRays, 1, 1  // <--- 这里的宽度变了！
			));
			CUDA_CHECK(cudaDeviceSynchronize());

			// 5. 拆箱：把所有射线的拓扑账本拉回 CPU
			out_topologies.resize(numRays);
			CUDA_CHECK(cudaMemcpy(out_topologies.data(), d_outCandidateTopologies, numRays * sizeof(Engine::Tracer::PathTopology), cudaMemcpyDeviceToHost));
		}

		void RayTracer::validatePathsOptiX(std::vector<Engine::Tracer::ExactPath>& paths) {
			if (paths.empty()) return;

			// 1. 将精确路径推入 GPU
			Engine::Tracer::ExactPath* d_paths = nullptr;
			size_t bytes = paths.size() * sizeof(Engine::Tracer::ExactPath);
			CUDA_CHECK(cudaMalloc((void**)&d_paths, bytes));
			CUDA_CHECK(cudaMemcpy(d_paths, paths.data(), bytes, cudaMemcpyHostToDevice));

			// 2. 配置 LaunchParams (完全参照 shootRaysBatchSBR 的变量命名)
			Engine::LaunchParams hostParams = {};
			hostParams.handle = sceneHandle;
			hostParams.tmax = 1000.0f;

			// 核心：绑定验证路径数组
			hostParams.validationPaths = d_paths;
			hostParams.numValidationPaths = static_cast<int>(paths.size());

			// 必须确保其他模式的指针为空，防止 Device 端误触发 SBR 逻辑
			hostParams.outCandidateTopologies = nullptr;
			hostParams.outSbrPath = nullptr;
			hostParams.txRayDirections = nullptr;

			// 拷贝参数到显卡 (d_params 是类成员)
			CUDA_CHECK(cudaMemcpy(d_params, &hostParams, sizeof(Engine::LaunchParams), cudaMemcpyHostToDevice));

			// 3. 呼叫 OptiX (线程数等于路径数，使用 0 号流)
			OPTIX_CHECK(optixLaunch(
				pipeline, 0,
				reinterpret_cast<CUdeviceptr>(d_params),
				sizeof(Engine::LaunchParams),
				&sbt,
				static_cast<unsigned int>(paths.size()), 1, 1
			));

			// 4. 同步并取回结果
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaMemcpy(paths.data(), d_paths, bytes, cudaMemcpyDeviceToHost));

			// 5. 打扫战场
			CUDA_CHECK(cudaFree(d_paths));
		}
	} // namespace Tracer
} // namespace Engine
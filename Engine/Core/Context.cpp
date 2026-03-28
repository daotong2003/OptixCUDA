// 通过 OptixContextManager 类来管理底层硬件环境
// 函数功能简介在.h文件中已经注释过了，这里就不赘述了
#include "Context.h"
#include "CudaError.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h> // 核心：实例化全局函数表
#include <iostream>
#include <iomanip>

namespace Engine {
	namespace Core {
		// 日志回调函数
		static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
			std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
		}

		OptixContextManager::OptixContextManager() {
			std::cout << "[System] 正在初始化底层硬件环境...\n";

			// 1. 唤醒 CUDA
			CUDA_CHECK(cudaFree(0));

			int numDevices = 0;
			CUDA_CHECK(cudaGetDeviceCount(&numDevices));
			if (numDevices == 0) {
				throw std::runtime_error("未找到支持 CUDA 的显卡！");
			}
			std::cout << "[System] 成功检测到 " << numDevices << " 张 CUDA 显卡。\n";

			// 2. 初始化 OptiX 函数表
			OPTIX_CHECK(optixInit());
			std::cout << "[System] OptiX API 初始化成功！\n";

			// 3. 创建 OptiX 设备上下文
			OptixDeviceContextOptions options = {};
			options.logCallbackFunction = &context_log_cb;
			options.logCallbackLevel = 2;

			OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));
			std::cout << "[System] OptiX 上下文创建成功，引擎已就绪！\n";
		}

		OptixContextManager::~OptixContextManager() {
			if (context != nullptr) {
				optixDeviceContextDestroy(context);
				std::cout << "[System] OptiX 上下文已安全销毁，显存资源释放完毕。\n";
			}
		}
	} // namespace Core
} // namespace Engine
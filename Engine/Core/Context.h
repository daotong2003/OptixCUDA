// (上下文管理类声明)

#include <optix.h>

namespace Engine {
    namespace Core {

        class OptixContextManager {
        public:
            // 构造函数：负责检测显卡并初始化 OptiX API
            OptixContextManager();

            // 析构函数：负责安全销毁上下文
            ~OptixContextManager();

            // 提供一个接口供外部获取上下文句柄
            OptixDeviceContext getContext() const { return context; }

        private:
            OptixDeviceContext context = nullptr;
        };

    } // namespace Core
} // namespace Engine
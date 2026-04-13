import sys
import os
import gc
import numpy as np

# =====================================================================
# [核心修复] Python 3.8+ 必须手动注入底层依赖库 (CUDA) 的路径
# =====================================================================
# 1. 注入 CUDA bin 目录 (cudart64_118.dll 所在位置)
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
if os.path.exists(cuda_bin_path) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(cuda_bin_path)
    print(f">>> [DLL 注入] 成功添加 CUDA 路径: {cuda_bin_path}")
else:
    print(f"[警告] 找不到 CUDA 路径: {cuda_bin_path}，可能导致 DLL 加载失败！")

# 2. 将编译输出目录添加到 sys.path (用于寻找 .pyd)
target_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\RelWithDebInfo"
sys.path.append(target_path)

# 同时也将目标目录加入 DLL 搜索路径 (防患于未然)
if os.path.exists(target_path) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(target_path)

print(">>> [Python 端] 正在尝试导入 optix_backend...")

try:
    import optix_backend

    print(">>> [Python 端] 成功导入 optix_backend 模块！")
except ImportError as e:
    print(f"\n[致命错误] 导入失败: {e}")
    print(
        "如果依然失败，请检查 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin 目录下是否存在 cudart64_xxx.dll")
    sys.exit(1)
# ... 前面的 DLL 注入和 import optix_backend 保持完全不变 ...

# =====================================================================
# 严格对齐 80 bytes 的内存映射结构 (绝对不能变)
# =====================================================================
exact_path_dtype = np.dtype([
    ('vertices', np.float32, (5, 3)),
    ('hit_objects', np.int32, (3,)),
    ('vertexCount', np.int32),
    ('isValid', np.bool_),
    ('padding', np.uint8, (3,))
])


def run_real_engine_test():
    # 1. 设置模型和 PTX 的绝对路径 (请确保这两个文件真实存在！)
    ply_path = r"E:\RT_software\Clanguage\sy.ply"

    # PTX 文件现在应该在你的 RelWithDebInfo 编译目录的 Ptx 文件夹下
    ptx_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\Ptx\device_programs.ptx"

    print("\n>>> [Python 端] 正在初始化 C++ OptiX 真实引擎 (这会加载点云并构建 GAS/IAS)...")
    try:
        # 实例化我们在 C++ 中写的真实桥接类
        engine = optix_backend.OptixEngineBridge(ply_path, ptx_path)
    except Exception as e:
        print(f"[引擎初始化失败] 请检查 PLY 或 PTX 路径是否正确: {e}")
        return

    # 2. 设置收发机参数 (跟你在 main.cpp 里测试的一样)
    tx_pos = (26.5, 4.0, 2.0)
    rx_pos = (25.5, 12.0, 2.0)
    rx_radius = 0.5
    num_rays = 2000000  # 直接上两百万条射线！

    print(f"\n>>> [Python 端] 引擎就绪！向 C++ 发起 {num_rays} 条并发射线追踪请求...")

    # 3. 呼叫 C++ 计算！返回的是底层的 uint8 字节流
    raw_buffer = engine.compute_paths(*tx_pos, *rx_pos, rx_radius, num_rays)

    # 4. Zero-Copy 瞬间重塑为 NumPy 结构体数组
    paths = raw_buffer.view(exact_path_dtype)

    print("\n========================================================")
    print(">>> [Python 端] 🚀 Zero-Copy 内存映射解析成功！")
    print(f"  - 提取到的绝对有效物理多径数量: {len(paths)} 条")
    print("========================================================")

    if len(paths) > 0:
        first_path = paths[0]
        print(f"\n>>> [路径 0 数据核查]")
        print(f"  - 顶点数量: {first_path['vertexCount']}")
        print(
            f"  - Tx 起点: ({first_path['vertices'][0][0]:.2f}, {first_path['vertices'][0][1]:.2f}, {first_path['vertices'][0][2]:.2f})")
        print(f"  - 第一次弹跳击中 Label: {first_path['hit_objects'][0]}")


if __name__ == "__main__":
    run_real_engine_test()
    print("\n>>> [Python 端] 退出作用域，强制执行垃圾回收...")
    gc.collect()
    print(">>> [Python 端] 测试完毕，引擎内存已安全释放。")
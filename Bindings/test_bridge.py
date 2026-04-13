import numpy as np
import os
import sys

# =========================================================
# 1. 严格对齐更新后的 C++ ExactPath 结构体 (216 Bytes)
# =========================================================
exact_path_dtype = np.dtype([
    ('vertices', np.float32, (5, 3)),
    ('normals', np.float32, (3, 3)),
    ('k_i', np.float32, (3, 3)),
    ('k_r', np.float32, (3, 3)),
    ('k_tx', np.float32, (3,)),
    ('k_rx', np.float32, (3,)),
    ('total_distance', np.float32),
    ('hit_objects', np.int32, (3,)),  # 记录平面 Label
    ('vertexCount', np.int32),
    ('isValid', np.uint8),
    ('hit_materials', np.uint8, (3,))  # [新增] 完美替换 padding[3]，提取材质 ID
], align=False)


def verify_materials(raw_buffer):
    print("\n>>> [验证阶段] 开始解析 C++ 零拷贝内存...")

    # 强制将 C++ 内存指针映射为结构化 NumPy 数组
    paths_np = np.frombuffer(raw_buffer, dtype=exact_path_dtype)
    num_paths = len(paths_np)
    print(f"成功映射总路径数: {num_paths}")

    if num_paths == 0:
        print("未收到任何路径数据！")
        return

    # 过滤出 isValid 为 True (1) 的有效物理路径
    valid_paths = paths_np[paths_np['isValid'] == 1]
    print(f"其中有效路径数: {len(valid_paths)}\n")

    # 打印前 5 条路径进行人眼核对
    for i in range(min(5, len(valid_paths))):
        path = valid_paths[i]
        bounces = path['vertexCount'] - 2  # 实际弹跳次数 = 顶点数 - Tx - Rx

        print(f"[Path {i}] 弹跳次数: {bounces}")
        print(f"  - 总距离 : {path['total_distance']:.4f} m")

        # 只打印实际发生弹跳的数组切片
        if bounces > 0:
            print(f"  - 击中平面 (hit_objects)  : {path['hit_objects'][:bounces]}")
            print(f"  - 击中材质 (hit_materials): {path['hit_materials'][:bounces]}")
        else:
            print("  - 直射径 (LOS)，无反射平面与材质。")
        print("-" * 50)


if __name__ == "__main__":
    # --- 您的环境注入代码 ---
    cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    target_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\RelWithDebInfo"
    if os.path.exists(cuda_bin_path) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_bin_path)
    if os.path.exists(target_path) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(target_path)
    sys.path.append(target_path)

    try:
        import optix_backend
    except ImportError as e:
        print(f"[致命错误] 导入失败: {e}")
        sys.exit(1)

    # --- 启动引擎并获取 raw_buffer ---
    ply_path = r"E:\RT_software\Clanguage\sy.ply"
    ptx_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\Ptx\device_programs.ptx"

    engine = optix_backend.OptixEngineBridge(ply_path, ptx_path)
    tx_pos = [26.5, 4.0, 2.0]
    rx_pos = [25.5, 12.0, 2.0]

    # 模拟寻径并获取内存指针
    raw_buffer = engine.compute_paths(*tx_pos, *rx_pos, 0.5, 2000000)

    # 运行验证
    verify_materials(raw_buffer)
import os
import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Paths

# ---------------------------------------------------------
# 1. 定义与 C++ 严格对齐的内存结构 (216 bytes)
# ---------------------------------------------------------
exact_path_dtype = np.dtype([
    ('vertices', np.float32, (5, 3)),     # 60 bytes: 5 个顶点 (Tx + 3反射点 + Rx)
    ('normals', np.float32, (3, 3)),      # 36 bytes: 3 次弹跳的表面法线
    ('k_i', np.float32, (3, 3)),          # 36 bytes: 3 次弹跳的入射方向
    ('k_r', np.float32, (3, 3)),          # 36 bytes: 3 次弹跳的反射方向
    ('k_tx', np.float32, (3,)),           # 12 bytes: 发射端出发方向
    ('k_rx', np.float32, (3,)),           # 12 bytes: 接收端到达方向
    ('total_distance', np.float32, (1,)), # 4 bytes: 总飞行距离
    ('hit_objects', np.int32, (3,)),      # 12 bytes: 3 次弹跳的平面 Label
    ('vertexCount', np.int32),            # 4 bytes: 实际顶点数
    ('isValid', np.bool_),                # 1 byte: 有效性标志
    ('padding', np.uint8, (3,))           # 3 bytes: 对齐填充
], align=False)


def create_phantom_scene(label_dict):
    """
    根据 C++ 端的平面 Label，动态生成一个只有 1 个极小三角形的幽灵场景。
    完美骗过 Sionna 的 TypeError('Only triangle meshes are supported') 检查。
    """
    # 1. 凭空捏造一个极小的单三角形 OBJ 文件 (边长 1 纳米)
    dummy_obj_path = "dummy_phantom.obj"
    dummy_obj_content = """v 0 0 0
v 1e-9 0 0
v 0 1e-9 0
f 1 2 3
"""
    with open(dummy_obj_path, "w") as f:
        f.write(dummy_obj_content)

    # 2. 组装 Mitsuba XML，所有 Label 物体都指向这个 dummy OBJ
    xml_content = ['<scene version="2.0.0">']

    for label in label_dict.keys():
        xml_content.append(f'''
        <shape type="obj" id="label_{label}">
            <string name="filename" value="{dummy_obj_path}"/>
        </shape>
        ''')
    xml_content.append('</scene>')

    # 写入临时 XML
    temp_xml = "phantom_scene.xml"
    with open(temp_xml, "w") as f:
        f.write("\n".join(xml_content))

    return temp_xml


# ---------------------------------------------------------
# 2. 定义与 C++ 严格对齐的内存结构 (216 bytes)
# ---------------------------------------------------------
exact_path_dtype = np.dtype([
    ('vertices', np.float32, (5, 3)),     # 60 bytes: 5 个顶点 (Tx + 3反射点 + Rx)
    ('normals', np.float32, (3, 3)),      # 36 bytes: 3 次弹跳的表面法线
    ('k_i', np.float32, (3, 3)),          # 36 bytes: 3 次弹跳的入射方向
    ('k_r', np.float32, (3, 3)),          # 36 bytes: 3 次弹跳的反射方向
    ('k_tx', np.float32, (3,)),           # 12 bytes: 发射端出发方向
    ('k_rx', np.float32, (3,)),           # 12 bytes: 接收端到达方向
    ('total_distance', np.float32, (1,)), # 4 bytes: 总飞行距离
    ('hit_objects', np.int32, (3,)),      # 12 bytes: 3 次弹跳的平面 Label
    ('vertexCount', np.int32),            # 4 bytes: 实际顶点数
    ('isValid', np.bool_),                # 1 byte: 有效性标志
    ('padding', np.uint8, (3,))           # 3 bytes: 对齐填充
], align=False)


# ---------------------------------------------------------
# 3. 核心电磁计算接入函数
# ---------------------------------------------------------
def run_sionna_pure_em(raw_buffer):
    # 1. 映射您的 C++ Label 到具体的电磁材质
    # 假设您的场景里 Label 6 是混凝土墙，Label 14 是金属机器
    cpp_material_map = {
        6: "itu_concrete",
        14: "itu_metal",
        2: "itu_glass"
    }

    print(">>> [Sionna EM] 1. 正在初始化幽灵材质场景...")
    phantom_xml = create_phantom_scene(cpp_material_map)
    sionna_scene = load_scene(phantom_xml)
    sionna_scene.frequency = 3.5e9  # 设置 5G NR 载波频率

    # 将 Sionna 内置的电磁属性赋予这些幽灵物体，并建立 ID 映射表
    label_to_sionna_id = {}
    for label, mat_name in cpp_material_map.items():
        obj_name = f"label_{label}"
        sionna_scene.objects[obj_name].radio_material = mat_name
        label_to_sionna_id[label] = sionna_scene.objects[obj_name].object_id

    # 2. 使用 Sionna 设置天线与收发机 (这正是你想要的特性)
    print(">>> [Sionna EM] 2. 正在配置 MIMO 天线阵列...")
    # 发射端配置 4x4 平面天线阵列
    tx_array = PlanarArray(num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole")
    # 接收端配置单天线
    rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="isotropic")

    tx = Transmitter("tx1", position=[26.5, 4.0, 2.0], orientation=[0, 0, 0])
    tx.array = tx_array
    sionna_scene.add(tx)

    rx = Receiver("rx1", position=[25.5, 12.0, 2.0], orientation=[0, 0, 0])
    rx.array = rx_array
    sionna_scene.add(rx)

    # 3. Zero-Copy 提取 C++ 的几何折线
    print(">>> [Sionna EM] 3. 正在重组 C++ 几何拓扑...")
    paths_np = raw_buffer.view(exact_path_dtype)
    num_paths = len(paths_np)
    max_bounces = 3

    if num_paths == 0:
        return

    expand_dims = [1, 1, 1, 1, 1, num_paths]

    # [核心：切片与提取]
    reflector_pts = paths_np['vertices'][:, 1:max_bounces + 1, :]
    vertices_tf = tf.reshape(tf.convert_to_tensor(reflector_pts), expand_dims + [max_bounces, 3])

    # [核心：转换材质 ID] 把 C++ 的 6, 14 转换成 Sionna 内部的 object_id
    raw_labels = paths_np['hit_objects']
    sionna_ids = np.zeros_like(raw_labels)
    for i in range(num_paths):
        for b in range(max_bounces):
            lbl = raw_labels[i, b]
            sionna_ids[i, b] = label_to_sionna_id.get(lbl, 0)  # 兜底材质 ID 0

    objects_tf = tf.reshape(tf.convert_to_tensor(sionna_ids), expand_dims + [max_bounces])

    bounce_counts = paths_np['vertexCount'] - 2
    mask_tf = tf.reshape(tf.sequence_mask(bounce_counts, maxlen=max_bounces), expand_dims + [max_bounces])

    # 4. 召唤 Sionna EM 求解器！
    print(">>> [Sionna EM] 4. 注入 Sionna Paths，开始电磁解算...")
    paths = Paths(
        sources=sionna_scene.transmitters,
        targets=sionna_scene.receivers,
        scene=sionna_scene,
        vertices=vertices_tf,
        objects=objects_tf,
        mask=mask_tf
    )

    # 取消内置的延迟归一化（因为我们的坐标已经是绝对精确的了）
    paths.normalize_delays = False

    # 解算复数衰减(a)和传播时延(tau)
    a, tau = paths.apply_em_solver()

    print("\n========================================================")
    print(">>> [成功] C++ 几何与 Sionna 电磁/天线 完美合体！")
    print(f"  - 包含天线方向图增益的衰减系数 (a) shape: {a.shape}")
    print(f"  - 传播时延 (tau) shape: {tau.shape}")

    # 甚至可以直接生成带 MIMO 效应的信道冲激响应 CIR！
    # cir = paths.cir()
    print("========================================================")


if __name__ == "__main__":
    import sys
    import gc

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

    # 记得把你的 out/build/x64-RelWithDebInfo/RelWithDebInfo 目录加入 sys.path
    # 并且 os.add_dll_directory CUDA bin 目录
    import optix_backend

    ply_path = r"E:\RT_software\Clanguage\sy.ply"
    ptx_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\Ptx\device_programs.ptx"

    engine = optix_backend.OptixEngineBridge(ply_path, ptx_path)
    # 调用 C++，拿到纯物理几何路径
    raw_buffer = engine.compute_paths(26.5, 4.0, 2.0, 25.5, 12.0, 2.0, 0.5, 2000000)

    # 将几何路径喂给 Sionna 算电磁和 MIMO 天线
    run_sionna_pure_em(raw_buffer)
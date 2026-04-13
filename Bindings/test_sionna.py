import os
import sys
import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Paths
from sionna.rt.solver_paths import PathsTmpData
# ---------------------------------------------------------
# 1. 结构对齐：严格匹配 SbrTypes.h 中的 216 字节布局
# ---------------------------------------------------------
# 根据 SbrTypes.h，MAX_BOUNCE_DEPTH = 3
# ExactPath 包含 vertices[5], normals[3], k_i[3], k_r[3], k_tx, k_rx, total_distance,
# hit_objects[3], vertexCount, isValid, hit_materials[3]
exact_path_dtype = np.dtype([
    ('vertices', np.float32, (5, 3)),
    ('normals', np.float32, (3, 3)),
    ('k_i', np.float32, (3, 3)),
    ('k_r', np.float32, (3, 3)),
    ('k_tx', np.float32, (3,)),
    ('k_rx', np.float32, (3,)),
    ('total_distance', np.float32),
    ('hit_objects', np.int32, (3,)),
    ('vertexCount', np.int32),
    ('isValid', np.uint8),
    ('hit_materials', np.uint8, (3,))
], align=False)


def create_nimbus_phantom_scene(unique_materials):
    """模仿 NimbusRT，根据材质 ID 动态生成幽灵对象"""
    dummy_obj_path = "dummy_phantom.obj"
    with open(dummy_obj_path, "w") as f:
        f.write("v 0 0 0\nv 1e-9 0 0\nv 0 1e-9 0\nf 1 2 3\n")

    xml_content = ['<scene version="2.0.0">']
    for mat_id in unique_materials:
        if mat_id != 255:
            xml_content.append(
                f'<shape type="obj" id="mat_obj_{mat_id}"><string name="filename" value="{dummy_obj_path}"/></shape>')
    xml_content.append('</scene>')

    temp_xml = "nimbus_scene.xml"
    with open(temp_xml, "w") as f:
        f.write("\n".join(xml_content))
    return temp_xml


def print_tensor_info(name, tensor):
    if isinstance(tensor, np.ndarray):
        print(f"  {name}: np.ndarray, shape={tensor.shape}, dtype={tensor.dtype}")
    elif isinstance(tensor, tf.Tensor):
        print(f"  {name}: tf.Tensor, shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        print(f"  {name}: {type(tensor).__name__}, shape={getattr(tensor, 'shape', 'N/A')}")


# ---------------------------------------------------------
# 2. 核心桥接逻辑
# ---------------------------------------------------------
def run_sionna_nimbus_final(raw_buffer, tx_pos, rx_pos):
    paths_np = np.frombuffer(raw_buffer, dtype=exact_path_dtype)
    valid_paths = paths_np[paths_np['isValid'] == 1]
    num_paths = len(valid_paths)

    if num_paths == 0:
        print(">>> [警告] 未发现有效路径。")
        return

    print(f">>> [调试] 有效路径数量: {num_paths}")

    # A. 初始化场景与材质映射
    unique_mats = np.unique(valid_paths['hit_materials'])
    sionna_scene = load_scene(create_nimbus_phantom_scene(unique_mats))
    sionna_scene.frequency = 3.5e9
    sionna_scene.synthetic_array = True

    for mat_id in unique_mats:
        if mat_id == 255:
            continue
        obj = sionna_scene.objects[f"mat_obj_{mat_id}"]
        obj.radio_material = "itu_concrete"
        obj.object_id = int(mat_id)

    # B. 天线配置 (4x4 MIMO)
    sionna_scene.tx_array = PlanarArray(4, 4, 0.5, 0.5, "dipole", "V")
    sionna_scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")

    tx = Transmitter("tx1", position=tx_pos)
    tx.array = sionna_scene.tx_array
    sionna_scene.add(tx)
    rx = Receiver("rx1", position=rx_pos)
    rx.array = sionna_scene.rx_array
    sionna_scene.add(rx)

    # C. 张量维度对齐
    # Sionna 0.19 shape 约定（参考 solver_paths.py 源码）：
    #   Paths.vertices:       [max_depth, num_rx, num_tx, num_paths, 3]
    #   Paths.objects:        [max_depth, num_rx, num_tx, num_paths]
    #   Paths.mask:           [num_rx, num_tx, num_paths]
    #   Paths.theta_t/phi_t:  [num_rx, num_tx, num_paths]
    #   PathsTmpData.normals: [max_depth, num_rx, num_tx, num_paths, 3]
    #   PathsTmpData.k_i:     [max_depth+1, num_rx, num_tx, num_paths, 3]
    #   PathsTmpData.k_r:     [max_depth, num_rx, num_tx, num_paths, 3]
    #   PathsTmpData.k_tx:    [num_rx, num_tx, num_paths, 3]
    #   PathsTmpData.k_rx:    [num_rx, num_tx, num_paths, 3]
    #   PathsTmpData.total_distance: [num_rx, num_tx, num_paths]
    # 注意：depth 维度在最前面！
    sources = tf.convert_to_tensor([tx_pos], dtype=tf.float32)
    targets = tf.convert_to_tensor([rx_pos], dtype=tf.float32)

    max_depth = 3
    num_rx, num_tx = 1, 1

    def to_tf(np_arr, shape, dtype=tf.float32):
        return tf.reshape(tf.convert_to_tensor(np.ascontiguousarray(np_arr), dtype=dtype), shape)

    k_tx_np = valid_paths['k_tx']
    k_rx_np = valid_paths['k_rx']

    print("\n>>> [调试] 原始 C++ 数据 shape:")
    print_tensor_info("  valid_paths['vertices']", valid_paths['vertices'])
    print_tensor_info("  valid_paths['normals']", valid_paths['normals'])
    print_tensor_info("  valid_paths['k_i']", valid_paths['k_i'])
    print_tensor_info("  valid_paths['k_r']", valid_paths['k_r'])
    print_tensor_info("  valid_paths['k_tx']", k_tx_np)
    print_tensor_info("  valid_paths['k_rx']", k_rx_np)
    print_tensor_info("  valid_paths['total_distance']", valid_paths['total_distance'])
    print_tensor_info("  valid_paths['hit_objects']", valid_paths['hit_objects'])

    theta_t = to_tf(np.arccos(np.clip(k_tx_np[:, 2], -1.0, 1.0)), [num_rx, num_tx, num_paths])
    phi_t = to_tf(np.arctan2(k_tx_np[:, 1], k_tx_np[:, 0]), [num_rx, num_tx, num_paths])
    theta_r = to_tf(np.arccos(np.clip(k_rx_np[:, 2], -1.0, 1.0)), [num_rx, num_tx, num_paths])
    phi_r = to_tf(np.arctan2(k_rx_np[:, 1], k_rx_np[:, 0]), [num_rx, num_tx, num_paths])

    ref_paths = Paths(sources, targets, sionna_scene)
    ref_paths.types = Paths.SPECULAR
    dif_paths = Paths(sources, targets, sionna_scene)
    dif_paths.types = Paths.DIFFRACTED
    sct_paths = Paths(sources, targets, sionna_scene)
    sct_paths.types = Paths.SCATTERED
    ris_paths = Paths(sources, targets, sionna_scene)
    ris_paths.types = Paths.RIS

    tmp_ref = PathsTmpData(sources, targets, dtype=tf.complex64)
    tmp_dif = PathsTmpData(sources, targets, dtype=tf.complex64)
    tmp_sct = PathsTmpData(sources, targets, dtype=tf.complex64)
    tmp_ris = PathsTmpData(sources, targets, dtype=tf.complex64)

    # vertices: C++ (num_paths, 5, 3) -> 取交互点 [:, 1:4, :] -> (num_paths, 3, 3)
    #         -> transpose -> (3, num_paths, 3) -> reshape -> (3, 1, 1, num_paths, 3)
    vertices_np = valid_paths['vertices'][:, 1:max_depth + 1, :]
    ref_paths.vertices = to_tf(vertices_np.transpose(1, 0, 2),
                               [max_depth, num_rx, num_tx, num_paths, 3])

    # objects: C++ (num_paths, 3) -> transpose -> (3, num_paths) -> reshape -> (3, 1, 1, num_paths)
    mat_ids = np.where(valid_paths['hit_materials'] == 255, -1, valid_paths['hit_materials'])
    ref_paths.objects = to_tf(mat_ids.transpose(1, 0),
                              [max_depth, num_rx, num_tx, num_paths], tf.int32)

    ref_paths.mask = tf.ones([num_rx, num_tx, num_paths], dtype=tf.bool)
    ref_paths.theta_t, ref_paths.phi_t = theta_t, phi_t
    ref_paths.theta_r, ref_paths.phi_r = theta_r, phi_r
    ref_paths.tau = to_tf(valid_paths['total_distance'] / 299792458.0,
                          [num_rx, num_tx, num_paths])

    # normals: C++ (num_paths, 3, 3) -> transpose -> (3, num_paths, 3) -> reshape -> (3, 1, 1, num_paths, 3)
    tmp_ref.normals = to_tf(valid_paths['normals'].transpose(1, 0, 2),
                            [max_depth, num_rx, num_tx, num_paths, 3])

    # k_i: C++ (num_paths, 3, 3) -> 需要追加第4项(到达接收端方向=-k_rx)
    #     -> (num_paths, 4, 3) -> transpose -> (4, num_paths, 3) -> reshape -> (4, 1, 1, num_paths, 3)
    k_i_np = valid_paths['k_i']
    k_i_full = np.concatenate([k_i_np, -k_rx_np[:, np.newaxis, :]], axis=1)
    tmp_ref.k_i = to_tf(k_i_full.transpose(1, 0, 2),
                        [max_depth + 1, num_rx, num_tx, num_paths, 3])

    # k_r: C++ (num_paths, 3, 3) -> transpose -> (3, num_paths, 3) -> reshape -> (3, 1, 1, num_paths, 3)
    tmp_ref.k_r = to_tf(valid_paths['k_r'].transpose(1, 0, 2),
                        [max_depth, num_rx, num_tx, num_paths, 3])

    # k_tx / k_rx: [num_rx, num_tx, num_paths, 3]
    tmp_ref.k_tx = to_tf(k_tx_np, [num_rx, num_tx, num_paths, 3])
    tmp_ref.k_rx = to_tf(k_rx_np, [num_rx, num_tx, num_paths, 3])

    # total_distance: [num_rx, num_tx, num_paths]
    tmp_ref.total_distance = to_tf(valid_paths['total_distance'],
                                   [num_rx, num_tx, num_paths])

    print("\n>>> [调试] 转换为 TensorFlow 后的 shape:")
    print_tensor_info("  ref_paths.vertices", ref_paths.vertices)
    print_tensor_info("  ref_paths.objects", ref_paths.objects)
    print_tensor_info("  ref_paths.mask", ref_paths.mask)
    print_tensor_info("  ref_paths.theta_t", ref_paths.theta_t)
    print_tensor_info("  ref_paths.phi_t", ref_paths.phi_t)
    print_tensor_info("  ref_paths.theta_r", ref_paths.theta_r)
    print_tensor_info("  ref_paths.phi_r", ref_paths.phi_r)
    print_tensor_info("  ref_paths.tau", ref_paths.tau)
    print_tensor_info("  tmp_ref.normals", tmp_ref.normals)
    print_tensor_info("  tmp_ref.k_i", tmp_ref.k_i)
    print_tensor_info("  tmp_ref.k_r", tmp_ref.k_r)
    print_tensor_info("  tmp_ref.k_tx", tmp_ref.k_tx)
    print_tensor_info("  tmp_ref.k_rx", tmp_ref.k_rx)
    print_tensor_info("  tmp_ref.total_distance", tmp_ref.total_distance)

    path_tuple = (ref_paths, dif_paths, sct_paths, ris_paths,
                  tmp_ref, tmp_dif, tmp_sct, tmp_ris)

    print("\n>>> [Sionna EM] 正在执行 Native 注入...")
    computed_paths = sionna_scene.compute_fields(*path_tuple, check_scene=False)

    print("\n========================================================")
    print(">>> [成功] C++ -> Sionna 链路完美贯通！")
    print(f"  - 衰减矩阵 (a) shape: {computed_paths.a.shape}")
    print(f"  - 时延矩阵 (tau) shape: {computed_paths.tau.shape}")
    print("========================================================")


if __name__ == "__main__":
    cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    target_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\RelWithDebInfo"
    if os.path.exists(cuda_bin_path) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_bin_path)
    if os.path.exists(target_path) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(target_path)
    sys.path.append(target_path)

    import optix_backend

    ply_path = r"E:\RT_software\Clanguage\sy.ply"
    ptx_path = r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\Ptx\device_programs.ptx"

    engine = optix_backend.OptixEngineBridge(ply_path, ptx_path)
    tx_pos = [26.5, 4.0, 2.0]
    rx_pos = [25.5, 12.0, 2.0]

    raw_buffer = engine.compute_paths(*tx_pos, *rx_pos, 0.5, 2000000)
    run_sionna_nimbus_final(raw_buffer, tx_pos, rx_pos)

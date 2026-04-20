import os
import sys
import time
import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Paths
from sionna.rt.solver_paths import PathsTmpData
from sionna.channel import cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel.utils import subcarrier_frequencies
import scipy.io

EXACT_PATH_DTYPE = np.dtype([
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

RT_PATHS_KEYS = [
    'a', 'tau', 'theta_t', 'phi_t', 'theta_r', 'phi_r',
    'source_coordinates', 'target_coordinates',
    'Transmitter_coordinates', 'Receiver_coordinates',
    'mask', 'objects', 'doppler', 'types',
    'targets_sources_mask', 'vertices'
]


def _create_phantom_scene(unique_materials):
    dummy_obj_path = "dummy_phantom.obj"
    with open(dummy_obj_path, "w") as f:
        f.write("v 0 0 0\nv 1e-9 0 0\nv 0 1e-9 0\nf 1 2 3\n")

    xml_content = ['<scene version="2.0.0">']
    for mat_id in unique_materials:
        if mat_id != 255:
            xml_content.append(
                f'<shape type="obj" id="mat_obj_{mat_id}">'
                f'<string name="filename" value="{dummy_obj_path}"/></shape>')
    xml_content.append('</scene>')

    temp_xml = "nimbus_scene.xml"
    with open(temp_xml, "w") as f:
        f.write("\n".join(xml_content))
    return temp_xml


def _build_sionna_scene(valid_paths, tx_pos, rx_pos, frequency, power_dbm):
    unique_mats = np.unique(valid_paths['hit_materials'])
    sionna_scene = load_scene(_create_phantom_scene(unique_mats))
    sionna_scene.frequency = frequency
    sionna_scene.synthetic_array = True

    for mat_id in unique_mats:
        if mat_id == 255:
            continue
        obj = sionna_scene.objects[f"mat_obj_{mat_id}"]
        obj.radio_material = "itu_concrete"
        obj.object_id = int(mat_id)

    sionna_scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
    sionna_scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")

    tx = Transmitter("tx1", position=tx_pos, power_dbm=power_dbm)
    tx.array = sionna_scene.tx_array
    sionna_scene.add(tx)
    rx = Receiver("rx1", position=rx_pos)
    rx.array = sionna_scene.rx_array
    sionna_scene.add(rx)

    return sionna_scene


def _convert_to_sionna_paths(valid_paths, sionna_scene, tx_pos, rx_pos, max_depth=3):
    num_paths = len(valid_paths)
    num_rx, num_tx = 1, 1

    sources = tf.convert_to_tensor([tx_pos], dtype=tf.float32)
    targets = tf.convert_to_tensor([rx_pos], dtype=tf.float32)

    def to_tf(np_arr, shape, dtype=tf.float32):
        return tf.reshape(
            tf.convert_to_tensor(np.ascontiguousarray(np_arr), dtype=dtype), shape)

    k_tx_np = valid_paths['k_tx']
    k_rx_np = valid_paths['k_rx']

    theta_t = to_tf(np.arccos(np.clip(k_tx_np[:, 2], -1.0, 1.0)),
                    [num_rx, num_tx, num_paths])
    phi_t = to_tf(np.arctan2(k_tx_np[:, 1], k_tx_np[:, 0]),
                  [num_rx, num_tx, num_paths])
    theta_r = to_tf(np.arccos(np.clip(k_rx_np[:, 2], -1.0, 1.0)),
                    [num_rx, num_tx, num_paths])
    phi_r = to_tf(np.arctan2(k_rx_np[:, 1], k_rx_np[:, 0]),
                  [num_rx, num_tx, num_paths])

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

    vertices_np = valid_paths['vertices'][:, 1:max_depth + 1, :]
    ref_paths.vertices = to_tf(vertices_np.transpose(1, 0, 2),
                               [max_depth, num_rx, num_tx, num_paths, 3])

    mat_ids = np.where(valid_paths['hit_materials'] == 255, -1,
                       valid_paths['hit_materials'])
    ref_paths.objects = to_tf(mat_ids.transpose(1, 0),
                              [max_depth, num_rx, num_tx, num_paths], tf.int32)

    ref_paths.mask = tf.ones([num_rx, num_tx, num_paths], dtype=tf.bool)
    ref_paths.theta_t, ref_paths.phi_t = theta_t, phi_t
    ref_paths.theta_r, ref_paths.phi_r = theta_r, phi_r
    ref_paths.tau = to_tf(valid_paths['total_distance'] / 299792458.0,
                          [num_rx, num_tx, num_paths])

    tmp_ref.normals = to_tf(valid_paths['normals'].transpose(1, 0, 2),
                            [max_depth, num_rx, num_tx, num_paths, 3])

    k_i_np = valid_paths['k_i']
    k_i_full = np.concatenate([k_i_np, -k_rx_np[:, np.newaxis, :]], axis=1)
    tmp_ref.k_i = to_tf(k_i_full.transpose(1, 0, 2),
                        [max_depth + 1, num_rx, num_tx, num_paths, 3])

    tmp_ref.k_r = to_tf(valid_paths['k_r'].transpose(1, 0, 2),
                        [max_depth, num_rx, num_tx, num_paths, 3])

    tmp_ref.k_tx = to_tf(k_tx_np, [num_rx, num_tx, num_paths, 3])
    tmp_ref.k_rx = to_tf(k_rx_np, [num_rx, num_tx, num_paths, 3])

    tmp_ref.total_distance = to_tf(valid_paths['total_distance'],
                                   [num_rx, num_tx, num_paths])

    return (ref_paths, dif_paths, sct_paths, ris_paths,
            tmp_ref, tmp_dif, tmp_sct, tmp_ris)


class OptixSionnaBinding:
    def __init__(self, ply_path, ptx_path, tx_pos,
                 cuda_bin_path=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
                 dll_path=r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\RelWithDebInfo",
                 frequency=3.5e9, power_dbm=30,
                 bandwidth=200e6, num_subcarriers=1024,
                 max_depth=3, point_radius=0.5, max_paths=2000000):
        self.tx_pos = np.asarray(tx_pos, dtype=np.float64)
        self.frequency = frequency
        self.power_dbm = power_dbm
        self.bandwidth = bandwidth
        self.num_subcarriers = num_subcarriers
        self.max_depth = max_depth
        self.point_radius = point_radius
        self.max_paths = max_paths

        self.frequencies = subcarrier_frequencies(
            num_subcarriers, bandwidth / num_subcarriers)

        if os.path.exists(cuda_bin_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(cuda_bin_path)
        if os.path.exists(dll_path) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_path)
        sys.path.append(dll_path)

        import optix_backend
        self.engine = optix_backend.OptixEngineBridge(ply_path, ptx_path)

    def compute_snapshot(self, rx_pos):
        rx_pos = np.asarray(rx_pos, dtype=np.float64)

        raw_buffer = self.engine.compute_paths(
            *self.tx_pos, *rx_pos, self.point_radius, self.max_paths)

        paths_np = np.frombuffer(raw_buffer, dtype=EXACT_PATH_DTYPE)
        valid_paths = paths_np[paths_np['isValid'] == 1]
        num_paths = len(valid_paths)

        if num_paths == 0:
            return None

        sionna_scene = _build_sionna_scene(
            valid_paths, self.tx_pos, rx_pos, self.frequency, self.power_dbm)
        path_tuple = _convert_to_sionna_paths(
            valid_paths, sionna_scene, self.tx_pos, rx_pos, self.max_depth)

        computed_paths = sionna_scene.compute_fields(
            *path_tuple, check_scene=False)

        a, tau = computed_paths.cir()

        H_rt = np.squeeze(
            cir_to_ofdm_channel(self.frequencies, a, tau, normalize=False))
        h_rt = np.squeeze(
            cir_to_time_channel(self.bandwidth, a, tau, 0, 1023,
                                normalize=False))

        snapshot_paths = {}
        snapshot_paths['a'] = tf.squeeze(computed_paths.a).numpy()
        snapshot_paths['tau'] = tf.squeeze(computed_paths.tau).numpy() * 1e9
        snapshot_paths['theta_t'] = tf.squeeze(computed_paths.theta_t).numpy()
        snapshot_paths['phi_t'] = tf.squeeze(computed_paths.phi_t).numpy()
        snapshot_paths['theta_r'] = tf.squeeze(computed_paths.theta_r).numpy()
        snapshot_paths['phi_r'] = tf.squeeze(computed_paths.phi_r).numpy()
        snapshot_paths['source_coordinates'] = tf.squeeze(
            computed_paths.sources).numpy()
        snapshot_paths['target_coordinates'] = tf.squeeze(
            computed_paths.targets).numpy()
        snapshot_paths['Transmitter_coordinates'] = self.tx_pos
        snapshot_paths['Receiver_coordinates'] = rx_pos
        snapshot_paths['mask'] = tf.squeeze(computed_paths.mask).numpy()
        snapshot_paths['objects'] = tf.squeeze(computed_paths.objects).numpy()
        snapshot_paths['types'] = tf.squeeze(computed_paths.types).numpy()
        snapshot_paths['vertices'] = tf.squeeze(
            computed_paths.vertices).numpy()
        snapshot_paths['doppler'] = tf.squeeze(
            computed_paths.doppler).numpy()
        snapshot_paths['targets_sources_mask'] = tf.squeeze(
            computed_paths.targets_sources_mask).numpy()

        return H_rt, h_rt, snapshot_paths

    def run_simulation(self, rx_positions, save_dir=None, filename='RT_CIR'):
        rx_positions = np.asarray(rx_positions, dtype=np.float64)
        num_snaps = len(rx_positions)

        H_rt_list = []
        h_rt_list = []
        rt_paths = {k: {} for k in RT_PATHS_KEYS}

        start_time = time.time()

        for snapid in range(num_snaps):
            rx_pos = rx_positions[snapid]
            print(f"\n>>> 快照 {snapid + 1}/{num_snaps}: RX = {rx_pos}")

            result = self.compute_snapshot(rx_pos)

            if result is None:
                print(">>> [警告] 未发现有效路径。")
                continue

            H_rt, h_rt, snapshot_paths = result
            H_rt_list.append(H_rt)
            h_rt_list.append(h_rt)

            for key in RT_PATHS_KEYS:
                rt_paths[key][snapid] = snapshot_paths[key]

            print(f">>> 有效路径数量: {snapshot_paths['a'].shape[-1]}")

        end_time = time.time()

        H_rt_final = np.stack(H_rt_list, axis=0)
        h_rt_final = np.stack(h_rt_list, axis=0)

        H_rt_revise = np.transpose(H_rt_final, (1, 0))
        h_rt_revise = np.transpose(h_rt_final, (1, 0))

        tau_rt = np.arange(h_rt_revise.shape[0]) / self.bandwidth * 1e9

        data_to_save = {
            'H_rt': H_rt_revise,
            'h_rt': h_rt_revise,
            'run_time': end_time - start_time,
            'rx_pos': rx_positions,
            'tx_pos': self.tx_pos,
            'delay_resolution': tau_rt,
            'paths': rt_paths,
            'scene_info': {},
            'fc': self.frequency,
            'bandwidth': self.bandwidth
        }

        print(f"\n=== 仿真完成 ===")
        print(f"H_rt shape: {H_rt_revise.shape}, dtype: {H_rt_revise.dtype}")
        print(f"h_rt shape: {h_rt_revise.shape}, dtype: {h_rt_revise.dtype}")
        print(f"tau_rt shape: {tau_rt.shape}")
        print(f"总耗时: {end_time - start_time:.2f} 秒")

        if save_dir is not None:
            self.save_results(data_to_save, save_dir, filename)

        return data_to_save

    @staticmethod
    def save_results(data_to_save, directory, filename='RT_CIR'):
        os.makedirs(directory, exist_ok=True)
        mat_path = os.path.join(directory, f'{filename}.mat')
        npy_path = os.path.join(directory, f'{filename}.npy')

        scipy.io.savemat(mat_path, data_to_save)
        print(f">>> 结果保存到 {mat_path}")
        np.save(npy_path, data_to_save)
        print(f">>> 结果保存到 {npy_path}")


if __name__ == "__main__":
    binding = OptixSionnaBinding(
        ply_path=r"E:\RT_software\Clanguage\sy.ply",
        ptx_path=r"E:\RT_software\Clanguage\PointRT\out\build\x64-RelWithDebInfo\Ptx\device_programs.ptx",
        tx_pos=np.array([26.5, 4.0, 2.0]),
    )

    rx_start = np.array([25.5, 12.0, 2.0])
    rx_end = np.array([45.5, 12.0, 2.0])
    num_snaps = 50
    rx_positions = np.linspace(rx_start, rx_end, num_snaps)

    data = binding.run_simulation(
        rx_positions,
        save_dir=r'.\Result',
        filename='RT_CIR_my'
    )

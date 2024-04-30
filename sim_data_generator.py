import os
import time
import argparse
import queue
import concurrent

import numpy as np

from entity import CW_Func_Handler, Snapshot_Generator, CW_Source, Three_Elements_Array
from utils import deg_pol2cart

# TODO: 改为生成为多个flac文件，文件中写入采样率


def sig_gen(fc: float, c: float, r: float, speed: float, angle: float, d: float, K: float, fs_factor: int, sample_interval: int):
    """生成指定参数组合的信号片段

    Parameters
    ----------
    fc : float
        声源频率
    r : float
        声源距离
    angle : float
        声源方位角（度）
    sample_interval : int
        采样时长

    Returns
    -------
    snapshots_slices : np.ndarray
        shape of (3, fc * fs_factor, sample_interval)
    fs : int
    rs : np.ndarray
        shape of (sample_interval,), 阵列坐标下每秒的声源距离
    angles : np.ndarray
        shape of (sample_interval,), 阵列坐标下每秒的声源方位角（度）
    """
    cw_func_handler = CW_Func_Handler(
        f=fc,  # 声源频率
        T=1,  # Cw信号周期
        T_on=10e-2,  # Cw信号脉宽
    )
    snapshot_generator = Snapshot_Generator(
        CW_Source(
            signal_func_callback=cw_func_handler,
            r=r,  # 声源距离
            angle=angle  # 声源角度
        ),
        Three_Elements_Array(d, K),
        c=c  # 声速
    )

    fs = fs_factor * cw_func_handler.f
    vel_angle = 90
    velocity = deg_pol2cart(speed, vel_angle)

    # sig.array.set_noise_params(0.01, 0.01, 0.01)
    snapshot_generator.set_ideal()

    t = np.arange(0, 1, 1 / fs)
    snapshots, r_real, angle_real = snapshot_generator(t, velocity)
    snapshots_slices = np.zeros((3, len(t), sample_interval))  # shape: (3, t_len, sample_interval)
    rs = np.zeros((sample_interval, 1))
    angles = np.zeros((sample_interval, 1))
    for i in range(sample_interval):
        t = t + 1
        snapshots, r_real, angle_real = snapshot_generator(t, velocity)
        snapshots_slices[:, :, i] = snapshots
        rs[i] = r_real
        angles[i] = angle_real
    return snapshots_slices.astype(np.float32), int(fs), rs, angles


def worker(angle):
    return sig_gen(args.fc, args.c, args.r, args.speed, angle, args.d, args.K, args.fs_factor, args.sample_interval)


if __name__ == '__main__':
    # python sim_data_generator.py --path dataset/train dataset/val --N 1500 500 --sample_interval 2 --d 0.08 --K 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc', type=float, default=37500)
    parser.add_argument('--c', type=float, default=1500)
    parser.add_argument('--r', type=float, default=100)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--d', type=float, default=1)
    parser.add_argument('--K', type=float, default=1)
    parser.add_argument('--fs_factor', type=int, default=4)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--path', nargs='*', type=str, default=['dataset/train', 'dataset/val'])
    parser.add_argument('--N', nargs='*', type=int, default=[1600, 400])
    parser.add_argument('--left_limit', type=int, default=15)
    parser.add_argument('--right_limit', type=int, default=165)
    args = parser.parse_args()
    assert len(args.path) == len(args.N), 'path和N的数量不一致'

    for path, N in zip(args.path, args.N):
        width = len(str(N))

        if not os.path.exists(path):
            os.makedirs(path)

        print(f'Generating {N} samples to {path}')

        label_filename = np.array([])
        source_init_angles = np.random.randint(args.left_limit, args.right_limit + 1, N)
        # 创建线程池和队列
        compute_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        q = queue.Queue()

        # 提交计算任务
        for angle in source_init_angles:
            future = compute_executor.submit(worker, angle)
            future.add_done_callback(lambda future: q.put(future.result()))

        # 提交I/O任务
        def save_result():
            count = 0
            while True:
                result = q.get()
                if result is None:
                    break
                count += 1
                filename = str(time.time()).replace('.', '')
                np.savez_compressed(f'{path}/{filename}.npz', snapshots=result[0], fs=result[1], angles=result[2])
                print(f'{count:{width}}: {path}/{filename}.npz')

        io_executor.submit(save_result)

        # 等待所有任务完成
        compute_executor.shutdown()
        q.put(None)
        io_executor.shutdown()

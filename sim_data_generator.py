import os
import time
import argparse

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from entity import CW_Func_Handler, Snapshot_Generator, CW_Source, Three_Elements_Array
from utils import deg_pol2cart

# TODO: 改为生成为多个flac文件，文件中写入采样率


def sig_gen(fc: float, c: float, r: float, angle: float | np.float32, d: float, K: float, fs_factor: int, sample_interval: int):
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
    snapshots : np.ndarray
        shape of (3, sample_interval * fc * 4)
    angle : float | np.float32
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
    speed = 0  # 先静止
    velocity = deg_pol2cart(speed, vel_angle)

    # sig.array.set_noise_params(0.01, 0.01, 0.01)
    snapshot_generator.set_ideal()

    t = np.arange(0, sample_interval, 1 / fs)
    snapshots = snapshot_generator(t, velocity)
    return snapshots.astype(np.float32), angle


def worker(angle):
    return sig_gen(args.fc, args.c, args.r, angle, args.d, args.K, args.fs_factor, args.sample_interval)


if __name__ == '__main__':
    # python sim_data_generator.py --path dataset/train dataset/val --N 1500 500 --sample_interval 2 --d 0.08 --K 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc', type=float, default=37500)
    parser.add_argument('--c', type=float, default=1500)
    parser.add_argument('--r', type=float, default=100)
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
        labels = np.random.randint(args.left_limit, args.right_limit + 1, N).astype(np.float32)
        with ThreadPoolExecutor(max_workers=8) as executor:
            count = 0
            futures = {executor.submit(worker, label) for label in labels}
            for future in as_completed(futures):
                snapshots, label = future.result()
                count += 1
                filename = str(time.time()).replace('.', '')

                np.save(f'{path}/{filename}.npy', snapshots)
                print(f'{count:{width}}: {filename}.npy with label {label}')

                label = str(label)
                if label_filename.size == 0:
                    label_filename = np.array([[label, filename]])
                else:
                    label_filename = np.concatenate((label_filename, np.array([[label, filename]])))

        # 如果labels.csv文件已存在，读取旧数据并与新数据拼接
        if os.path.isfile(f'{path}/labels.csv'):
            old_data = np.genfromtxt(f'{path}/labels.csv', delimiter=',', dtype='<U')
            label_filename = np.concatenate((old_data, label_filename))

        np.savetxt(f'{path}/labels.csv', label_filename, delimiter=',', fmt='%s')

import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    parser.add_argument('--fc', type=int, default=37500)
    parser.add_argument('--c', type=float, default=1500)
    parser.add_argument('--r', type=float, default=100)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--d', type=float, default=1)
    parser.add_argument('--K', type=float, default=1)
    parser.add_argument('--fs_factor', type=int, default=4)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--path', type=str, default='dataset')
    parser.add_argument('--N', nargs='*', type=int, default=[1600, 400])
    parser.add_argument('--left_limit', type=int, default=15)
    parser.add_argument('--right_limit', type=int, default=165)
    args = parser.parse_args()

    paths = [f'{args.path}/fc_{args.fc}-fs_factor_{args.fs_factor}-d_{args.d}-K_{args.K}/{item}' for item in ['train', 'val']]

    for path, N in zip(paths, args.N):
        width = len(str(N))

        if not os.path.exists(path):
            os.makedirs(path)

        print(f'Generating {N} samples to {path}')

        label_filename = np.array([])
        source_init_angles = np.random.randint(args.left_limit, args.right_limit + 1, N)
        with ThreadPoolExecutor(max_workers=12) as executor:
            count = 0
            jobs = {}
            jobs_left = len(source_init_angles)
            jobs_iter = iter(source_init_angles)
            while jobs_left:
                for angle in jobs_iter:
                    job = executor.submit(worker, angle)
                    jobs[job] = angle
                    if len(jobs) > 300:  # 限制同时进行的任务数量, 否则会消耗过量内存
                        break
                for job in as_completed(jobs):
                    jobs_left -= 1
                    snapshots, fs, _, angles = job.result()
                    del jobs[job]  # 强制回收内存
                    count += 1
                    filename = str(time.time()).replace('.', '')
                    np.savez(f'{path}/{filename}.npz', snapshots=snapshots, fs=fs, angles=angles)
                    print(f'{count:{width}}: {path}/{filename}.npz')

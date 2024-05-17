import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from entity import Array_Data_Sampler, CW_Func_Handler, CW_Source, Three_Elements_Array
from utils import deg_pol2cart


def sig_gen(fc: float, c: float, r: float, speed: float, angle: float, d: float, K: float, SNR_dB: float, fs_factor: int, sample_interval: int):
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
        prf=1,  # Cw脉冲重复频率
        pulse_width=10e-3,  # Cw信号脉宽
    )
    array_data_sampler = Array_Data_Sampler(
        CW_Source(
            signal_func_callback=cw_func_handler,
            r=r,  # 声源距离
            angle=angle  # 声源角度
        ),
        Three_Elements_Array(d, K),
        c=c  # 声速
    )
    if SNR_dB == 'ideal':
        array_data_sampler.set_ideal()
    else:
        array_data_sampler.set_SNR(float(SNR_dB))

    fs = fs_factor * cw_func_handler.f
    vel_angle = 90
    velocity = deg_pol2cart(speed, vel_angle)

    t = np.arange(0, 1, 1 / fs)
    data, r_real, angle_real = array_data_sampler(t, velocity)
    data_segments = np.zeros((sample_interval, 3, len(t)))  # shape: (sample_intervals, 3, t_len)
    r_n = np.zeros((sample_interval, 1))
    angle_n = np.zeros((sample_interval, 1))
    for i in range(sample_interval):
        t = t + 1
        data, r_real, angle_real = array_data_sampler(t, velocity)
        data_segments[i, :, :] = data
        r_n[i] = r_real
        angle_n[i] = angle_real
    return data_segments.astype(np.float32), int(fs), r_n, angle_n


def worker(distance, angle):
    return sig_gen(args.fc, args.c, distance, args.speed, angle, args.d, args.K, args.SNR, args.fs_factor, args.sample_interval)


if __name__ == '__main__':
    # python sim_data_generator.py --path dataset/train dataset/val --N 1500 500 --sample_interval 2 --d 0.08 --K 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc', type=int, default=37500)
    parser.add_argument('--c', type=float, default=1500)
    parser.add_argument('--r', nargs=2, type=int, default=[100, 1000])
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--d', type=float, default=0.5)
    parser.add_argument('--K', type=float, default=1.0)
    parser.add_argument('--SNR', type=str, default='10', help='value could be "ideal" or a float number to set SNR in dB.')
    parser.add_argument('--fs_factor', type=int, default=4)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--path', type=str, default='dataset')
    parser.add_argument('--N', nargs='*', type=int, default=[1600, 400])
    parser.add_argument('--left_limit', type=int, default=15)
    parser.add_argument('--right_limit', type=int, default=165)
    args = parser.parse_args()

    paths = [f'{args.path}/fc_{args.fc}-fs_factor_{args.fs_factor}-d_{args.d}-K_{args.K}-SNR_{args.SNR}-r_{args.r[0]}_{args.r[1]}/{item}' for item in ['train', 'val']]

    for path, N in zip(paths, args.N):
        width = len(str(N))

        if not os.path.exists(path):
            os.makedirs(path)

        print(f'Generating {N} samples to {path}')

        label_filename = np.array([])
        source_init_angles = np.random.randint(args.left_limit, args.right_limit + 1, N)
        source_init_distances = np.random.uniform(args.r[0], args.r[1], N)
        with ThreadPoolExecutor(max_workers=32) as executor:
            count = 0
            jobs = {}
            jobs_left = len(source_init_angles)
            jobs_iter = iter(zip(source_init_distances, source_init_angles))
            while jobs_left:
                for distance, angle in jobs_iter:
                    job = executor.submit(worker, distance, angle)
                    jobs[job] = angle
                    if len(jobs) > 3000:  # 限制同时进行的任务数量, 否则会消耗过量内存
                        break
                for job in as_completed(jobs):
                    jobs_left -= 1
                    data_segments, fs, r_n, angle_n = job.result()
                    del jobs[job]  # 强制回收内存
                    count += 1
                    filename = str(time.time()).replace('.', '')
                    np.savez(f'{path}/{filename}.npz', data_segments=data_segments, fs=fs, r_n=r_n, angle_n=angle_n)
                    print(f'{count:{width}}: {path}/{filename}.npz')

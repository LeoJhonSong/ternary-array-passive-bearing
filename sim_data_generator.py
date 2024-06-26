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
        device='cuda',
    )
    array_data_sampler = Array_Data_Sampler(
        CW_Source(
            signal_func_callback=cw_func_handler,
            r=r,  # 声源距离
            angle=angle,  # 声源角度
            device='cuda',
        ),
        Three_Elements_Array(d, K, device='cuda'),
        c=c,  # 声速
        device='cuda',
    )
    if SNR_dB == 'ideal':
        array_data_sampler.set_ideal()
    else:
        array_data_sampler.set_SNR(float(SNR_dB))

    fs = fs_factor * cw_func_handler.f
    vel_angle = 90
    velocity = deg_pol2cart(speed, vel_angle)

    t = np.arange(0, 1, 1 / fs)
    data, r_real, angle_real, t_bound = array_data_sampler(t, velocity)
    data_segments = np.zeros((sample_interval, 3, len(t)))  # shape: (sample_intervals, 3, t_len)
    r_n = np.zeros((sample_interval, 1))
    angle_n = np.zeros((sample_interval, 1))
    t_bound_n = np.zeros((sample_interval, 2))
    for i in range(sample_interval):
        t = t + 1
        data, r_real, angle_real, t_bound = array_data_sampler(t, velocity)
        t_bound = np.array((t_bound[0] - t[0], t_bound[1] - t[0]))
        data_segments[i, :, :] = data
        r_n[i] = r_real
        angle_n[i] = angle_real
        t_bound_n[i] = t_bound
    return data_segments.astype(np.float32), int(fs), r_n.astype(np.float32), angle_n.astype(np.float32), t_bound_n.astype(np.float32)


def worker(distance, angle, snr):
    return sig_gen(args.fc, args.c, distance, args.speed, angle, args.d, args.K, snr, args.fs_factor, args.sample_interval)


if __name__ == '__main__':
    # python sim_data_generator.py --path dataset/train dataset/val --N 1500 500 --sample_interval 2 --d 0.08 --K 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc', type=int, default=37500)
    parser.add_argument('--c', type=float, default=1500)
    parser.add_argument('--r', nargs=2, type=int, default=[100, 1000])
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--d', type=float, default=0.5)
    parser.add_argument('--K', type=float, default=1.0)
    # parser.add_argument('--SNR', nargs=2, type=int, default=[0, 30], help='SNR in dB.')
    # parser.add_argument('--ideal', action='store_true', help='when this set, SNR would be ignored and ideal signals are generated.')
    parser.add_argument('--fs_factor', type=int, default=4)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--path', type=str, default='dataset')
    parser.add_argument('--N', nargs='*', type=int, default=[1000, 500, 500, 500, 500])
    parser.add_argument('--left_limit', type=int, default=15)
    parser.add_argument('--right_limit', type=int, default=165)
    args = parser.parse_args()

    paths = [f'{args.path}/fc-{args.fc}_fs_factor-{args.fs_factor}_d-{args.d}_K-{args.K}_r-{args.r[0]}-{args.r[1]}/{item}' for item in ['train', 'val']]
    snr_set = ['ideal', '30', '20', '10', '5']

    for i, path in enumerate(paths):
        for SNR, N in zip(snr_set, args.N):
            if i == 1:
                N = N // 4
            width = len(str(N))

            if not os.path.exists(f'{path}/{SNR}'):
                os.makedirs(f'{path}/{SNR}')

            print(f'Generating {N} samples to {path}')

            label_filename = np.array([])
            source_init_angles = np.random.uniform(args.left_limit, args.right_limit, N)
            source_init_distances = np.random.uniform(args.r[0], args.r[1], N)
            snr_batch = [SNR] * N

            with ThreadPoolExecutor(max_workers=32) as executor:
                count = 0
                jobs = {}
                jobs_left = len(source_init_angles)
                jobs_iter = iter(zip(source_init_distances, source_init_angles, snr_batch))
                while jobs_left:
                    for distance, angle, snr in jobs_iter:
                        job = executor.submit(worker, distance, angle, snr)
                        jobs[job] = angle
                        if len(jobs) > 3000:  # 限制同时进行的任务数量, 否则会消耗过量内存
                            break
                    for job in as_completed(jobs):
                        jobs_left -= 1
                        data_segments, fs, r_n, angle_n, t_bound_n = job.result()
                        del jobs[job]  # 强制回收内存
                        count += 1
                        filename = str(time.time()).replace('.', '')
                        np.savez(f'{path}/{SNR}/{filename}.npz', data_segments=data_segments, fs=fs, r_n=r_n, angle_n=angle_n, t_bound_n=t_bound_n)
                        print(f'{count:{width}}: {path}/{SNR}/{filename}.npz')

import numpy as np
from multiprocessing import Pool

from entity import CW_Signal, Array_Signals, CW_Source, Three_Elements_Array
from utils import deg_pol2cart


def sig_gen(fc: float, r: float, angle: float, sample_interval: int):
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
    (3, sample_interval * fc * 4)
        _description_
    """
    target_sig = CW_Signal(
        f=fc,  # 声源频率
        T=1,  # Cw信号周期
        T_on=10e-3,  # Cw信号脉宽
    )
    sig = Array_Signals(
        CW_Source(
            signal=target_sig,
            r=r,  # 声源距离
            angle=angle  # 声源角度
        ),
        Three_Elements_Array(d=1, K=1),
        c=1500  # 声速
    )

    sample_interval = 5  # 采样时长
    fs = int(target_sig.f * 4)  # 4倍频采样
    vel_angle = 90
    speed = 0  # 先静止
    velocity = deg_pol2cart(speed, vel_angle)

    # sig.array.set_noise_params(0.01, 0.01, 0.01)
    sig.set_ideal()
    sig.init_rng(0)

    t = np.arange(0, sample_interval, 1 / fs)
    x = sig.next(t, velocity)
    return x


def worker(angle):
    return sig_gen(fc, r, angle, sample_interval)


if __name__ == '__main__':
    fc = 37500
    r = 100
    sample_interval = 5

    start_angle = 15
    end_angle = 165

    x_arr = []
    labels = np.array(np.arange(start_angle, end_angle + 1))
    with Pool() as p:
        x_arr = p.map(worker, labels)

    array_signal = np.stack(x_arr, axis=2)
    np.savez('sim_data.npz', array_signal=array_signal, labels=labels)

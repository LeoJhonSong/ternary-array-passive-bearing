import sys
from os.path import abspath, dirname

import numpy as np
import plotly.graph_objects as go
import utils
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

sys.path.append(dirname(dirname(abspath(__file__))))
from algorithm import cw_delay_estimation, far_locate
from simulation.signal import Array_Signals

# 设置参数 ######################################################################

f = 37500  # 声源频率
sample_interval = 1  # 采样时长
r = 100  # 距离
theta = np.deg2rad(60)  # 角度
vel_angle = 90

d = 0.5
K = 0.5 / d
T = 1  # Cw信号周期
T_on = 10e-3  # Cw信号脉宽
c = 1500  # 声速
fs = 500e3  # 采样频率500k

# 几何模型 ######################################################################

d1, d2, d3 = -(K + 1) * d / 2, -(K - 1) * d / 2, (K + 1) * d / 2
S = np.array([r * np.cos(theta), r * np.sin(theta)])

r1 = float(np.linalg.norm(S - [d1, 0]))
r2 = float(np.linalg.norm(S - [d2, 0]))
r3 = float(np.linalg.norm(S - [d3, 0]))

# 真实时延
t12 = (r1 - r2) / c
t23 = (r2 - r3) / c

# 仿真数据源 ####################################################################

interval_ = sample_interval * 1000
down_sample_rate = 50
r_far, angle_far = 0, 0
position = np.array([[0], [0]])

t = np.arange(0, sample_interval, 1 / fs)
latest_samples = np.zeros((3, int(T * fs)))  # 3个通道最后一周期的数据
plot_samples = np.zeros((4, int(6 * fs / down_sample_rate)))
sig = Array_Signals(c, f, T_on, T, S, (d1, d2, d3))
sig.set_params(1, 1, 1, 1, 1e-6)
sig.init_rng(1)

# Dash #########################################################################

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
app = Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    update_title=None  # type: ignore
)
app.title = '三元阵目标定位系统'
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
application = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [html.H3("三元阵目标定位系统", className="app__header__title")],
                    className="app__header__desc",
                )
            ],
            className="app__header"
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id='signal',
                            animate=True,
                            # animation_options={'transition': {'duration': 800}}
                        ),
                        dcc.Interval(id='signal-update', interval=1000, n_intervals=0)
                    ],
                    className="signal__container first"
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id='direction', animate=True)],
                            className="one-third column graph__container second"
                        ),
                        html.Div(
                            [dcc.Graph(id='trajectory', animate=True)],
                            className="two-thirds column graph__container second"
                        ),
                        dcc.Interval(id='data-update', interval=interval_, n_intervals=0)  # FIXME: 这个应该不用了
                    ],
                    className='direction__trajectory'
                ),
            ],
            className="app__content"
        )
    ],
    className="app__container"
)


@app.callback(Output('signal', 'figure'), [Input('signal-update', 'n_intervals')])
def get_signal_data(interval):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=plot_samples[0], y=plot_samples[1], mode='lines', name='阵元1信号'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_samples[0], y=plot_samples[2], mode='lines', name='阵元2信号'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_samples[0], y=plot_samples[3], mode='lines', name='阵元3信号'), row=3, col=1)
    fig.update_xaxes(range=[np.min(plot_samples[0]), np.max(plot_samples[0])])
    fig.update_layout(
        paper_bgcolor=app_color["graph_bg"],
        plot_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        title_text='信号实时数据 (V)',
        title_x=0.03
    )
    return fig


@app.callback(Output('direction', 'figure'), [Input('data-update', 'n_intervals')])
def estimate_direction(interval):
    global vel_angle, position, t, sig, plot_samples, latest_samples, r_far, angle_far

    vel = utils.deg2vec(vel_angle, 1)
    position = np.hstack((position, position[:, -1:] + sample_interval * np.array([vel]).T))
    # 应当使用当前时刻的位置计算时延与角度, 这是期望得到的, 但一定估计不准, 除非加上预测
    t12, t23, r_real, angle_real = utils.calc_real(c, (d1, d2, d3), vel_angle, position[:, -1], S)

    x = sig.next(t, vel)
    latest_samples = np.hstack((latest_samples[:, x.shape[1]:], x))
    plot_samples = np.hstack((plot_samples[:, x.shape[1]:], np.vstack((t, x))[:, ::down_sample_rate]))

    tau12_hat = cw_delay_estimation(latest_samples[0], latest_samples[1], fs, f, T, T_on, 1)
    tau23_hat = cw_delay_estimation(latest_samples[1], latest_samples[2], fs, f, T, T_on, 1)
    print('─────────────────────────────────────────────────────────────')
    print(f"""\t\ttau12\t\t\ttau23
真实时延差\t{t12}\t{t23}
相关时延差\t{tau12_hat}\t{tau23_hat}
误差\t\t{np.abs(t12 - tau12_hat) / t12 * 100:.2f}%\t\t\t{np.abs(t23 - tau23_hat) / t23 * 100:.2f}%""")

    print(f'\n速度方向: {vel_angle}')
    if abs(tau12_hat) >= abs(d) / c or abs(tau23_hat) >= abs(K * d) / c:
        print('时延估计错误')
    elif np.sign(tau12_hat) == np.sign(tau23_hat):
        # 远场条件下tau12, tau23应当同号
        r_far, angle_far = far_locate(tau12_hat, tau23_hat, c, K, d)
        vel_angle = vel_angle - 90 + angle_far
    else:
        r_far, angle_far = np.nan, np.pi / 2
    print(f'真实| theta: {angle_real:.3f}, r: {r_real:.3f}')
    print(f'远场| theta: {angle_far:.3f}, r: {r_far:.3f}')
    t = t + sample_interval

    fig = go.Figure(go.Barpolar(r=[r_real, r_real], theta=[angle_far, angle_real], width=[10, 10], opacity=0.8, marker_color=['rgb(99, 110, 250)', 'red']))
    fig.update_layout(
        paper_bgcolor=app_color["graph_bg"],
        plot_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        title_text='目标估计方位角',
        title_x=0.03,
    )
    fig.update_polars(
        bgcolor=app_color["graph_line"],
        angularaxis_gridcolor='#999',
        radialaxis_angle=angle_far,
        radialaxis_gridcolor='rgba(0, 0, 0, 0)'
    )
    return fig


@app.callback(Output('trajectory', 'figure'), [Input('data-update', 'n_intervals')])
def plot_trajectory(interval):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=position[0], y=position[1], mode='lines', name='潜器轨迹'))
    fig.add_trace(go.Scatter(x=[S[0]], y=[S[1]], mode='markers', name='目标位置'))
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        # range=(min(S[0], 0) - 1, max(S[0], 0) + 1),
        # constrain='domain'
    )
    fig.update_layout(
        paper_bgcolor=app_color["graph_bg"],
        plot_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        title_text='潜器运动轨迹 (仿真)',
        title_x=0.03
    )
    # TODO: 画个速度方向箭头
    return fig


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, processes=6, threaded=False)

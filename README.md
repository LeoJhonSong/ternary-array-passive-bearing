# ternary-array-passive-bearing
passive bearing based on ternary sonar array

## 数据流

1. A/D采样后为数字信号, 即采样时间离散, 数值为量化数值
2. 在python程序中收集采样后数据, 每1s数据解算一次
3. TODO: 降采样后的数据以及每秒解算出的角度存至数据库

## MARS

### 树莓派作为网关组网

- 电脑侧设置与`10.30.4.77`通信时通过**eth0**接口收发, 网关为`10.30.4.77`

	```sh
	sudo ip route add 10.30.4.77 via [树莓派eth0 ip] dev eth0
	```

- 树莓派侧**eth1**设置静态ip为`10.30.4.1`, 子网掩码`255.255.255.0`, 网关`10.30.4.1`, 开启ip转发

	```sh
	# 临时 (重启后失效)
	sysctl -w net.ipv4.ip_forward=1
	# 永久
	sudo vi /etc/sysctl.conf # 取消net.ipv4.ip_forward=1的注释
	sudo sysctl -p /etc/sysctl.conf
	```

#### 路由配置命令备忘

```bash
# 查看当前路由
ip route
# 删除指定路由
sudo ip route del default via <gateway_ip> dev <interface>
# 添加路由并设置优先级 (数值越小优先级越高)
sudo ip route add default via <gateway_ip> dev <interface> metric <new_metric>
```

## 参考资料

- [dash.dcc.Graph属性](https://dash.plotly.com/dash-core-components/graph#graph-properties)
- [通过dcc.Interval进行实时刷新](https://dash.plotly.com/live-updates)
- [plotly Python接口文档](https://plotly.com/python/)
- [参考实现: dash-wind-streaming](https://dash.gallery/dash-wind-streaming/), [代码](https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-wind-streaming/app.py)
- [在Dash回调间共享数据](https://dash.plotly.com/sharing-data-between-callbacks)
- [如何正确使用numba给Python加速？](https://www.zhihu.com/question/406931055)
- [Parallel Programming with numpy and scipy](https://scipy.github.io/old-wiki/pages/ParallelProgramming)
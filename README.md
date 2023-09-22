# ternary-array-passive-bearing
passive bearing based on ternary sonar array

## 数据流

1. A/D采样后为数字信号, 即采样时间离散, 数值为量化数值
2. 在python程序中收集采样后数据, 每1s数据解算一次
3. TODO: 降采样后的数据以及每秒解算出的角度存至数据库

## 参考资料

- [dash.dcc.Graph属性](https://dash.plotly.com/dash-core-components/graph#graph-properties)
- [通过dcc.Interval进行实时刷新](https://dash.plotly.com/live-updates)
- [plotly Python接口文档](https://plotly.com/python/)
- [参考实现: dash-wind-streaming](https://dash.gallery/dash-wind-streaming/), [代码](https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-wind-streaming/app.py)
- [在Dash回调间共享数据](https://dash.plotly.com/sharing-data-between-callbacks)
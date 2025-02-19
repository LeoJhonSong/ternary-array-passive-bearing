import logging
import time

from yacs.config import CfgNode as CN
import sensors.mars as mars

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')

logger = logging.getLogger('detector')

cfg = CN.load_cfg(open('./config.yaml'))

mars = mars.Mars(cfg)
mars.connect()
mars.record_switch(True)

while True:
    is_connected, frame_length = mars.read_parse_head()
    if not is_connected:
        break
    if frame_length is None:
        # 不是帧头
        continue
    is_connected, array_data = mars.read_data_frame(frame_length)
    if not is_connected:
        break
    if array_data is None:
        # 不是实时预览数据帧
        continue
    latest_array_data = mars.stack_data(array_data)
    if latest_array_data is None:
        continue
    # TODO: 估计角度
    # TODO: 打印/绘图
        break
mars.record_switch(False)
mars.disconnect()

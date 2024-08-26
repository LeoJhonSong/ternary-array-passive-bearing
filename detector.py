import logging

import mars

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')

logger = logging.getLogger('detector')

mars = mars.Mars('10.30.4.77', 7777, 7778)
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
    print()
    break
mars.record_switch(False)
mars.disconnect()

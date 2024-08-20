import socket
import struct
import numpy as np

# TODO: 封装为模块

# 参数
mars_ip = '10.30.4.77'
data_port = 7778
header_len = 6

# 创建套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((mars_ip, data_port))
print('已连接')  # TODO: 改用log

# TODO: 开始采样

# 持续读取实时预览数据帧
while True:
    # 读取数据帧头
    header = sock.recv(header_len)
    # header = bytearray(f.read(header_len))
    if not header:
        break  # 连接已关闭

    preamble, frame_length, protocol_version = struct.unpack('<HHH', header)

    # 检查开始字节及协议版本是否正确 (用于判读是否是数据帧头)
    if preamble != 0xFEFE or protocol_version != 0x0001:
        continue

    # 读取数据帧剩余部分
    data = b''
    while len(data) < frame_length - header_len:
        chunk = sock.recv(frame_length - header_len - len(data))
        if not chunk:
            break  # 连接已关闭, 数据不完整
        data += chunk

    # 解析数据帧内容
    meta_data, data_content = data[:6], data[6:]
    transaction_id, source_addr, dest_addr, frame_type, crc = struct.unpack('<BBBBH', meta_data)

    # 检查帧类型是否为实时预览数据帧
    if frame_type != 0x82:
        continue

    (
        _,  # 4
        data_byte_num,  # 2. 给出数据字节数, 每个数据点为3字节大端序有符整数, 对应-2.5V-2.5V范围, 按通道排序
        _,  # 2
        sample_offset,  # 8
        channel_mask,  # 12
    ) = struct.unpack('<IHHQ12s', data_content[:28])
    channel_mask = int.from_bytes(channel_mask, 'little')
    sample_num = data_byte_num // 3 // bin(channel_mask).count('1')
    preview_data = data_content[28:]
    samples = []
    for i in range(sample_num):
        start = 9 * i
        sample_vector = np.array([
            int.from_bytes(preview_data[start + 3 * j: start + 3 * (j + 1)], byteorder='big', signed=True)
            for j in range(3)
        ])
        samples.append(sample_vector / 2 ** 23 * 2.5)

        # TODO: 保存为.npy


sock.close()

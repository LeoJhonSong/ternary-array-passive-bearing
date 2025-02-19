import logging
import socket
import struct

import numpy as np
import torch
import torchaudio
from yacs.config import CfgNode as CN

# 创建日志记录器
logger = logging.getLogger('MARS')


class Mars:
    def __init__(self, cfg: CN):
        self.cfg = cfg
        self.ip = cfg.mars.ip
        self.command_port = cfg.mars.command_port
        self.data_port = cfg.mars.data_port
        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.header_len = 6
        self.preamble = 0xFEFE
        self.protocol_version = 0x0001
        self.msg_scrambler_number = 0x5A5C
        self.channels = cfg.mars.channels
        self.sample_bytes = 3
        self.downsampler = torchaudio.transforms.Resample(cfg.mars.fs, cfg.signal.fs)
        self.window = np.hanning
        self.overlap = 110  # 正常情况mars每次发送110组数据
        self.downsampled_overlap = int(np.ceil(self.overlap * cfg.signal.fs / cfg.mars.fs))  # 下采样后的重叠长度
        self.overlap_win = np.zeros((self.channels, self.overlap))
        # 目前作为*降采样*后最后cfg.signal.fs个阵列数据样本的缓冲区
        self.array_data_buf = np.zeros((self.channels, 0))

    def connect(self):
        self.command_sock.connect((self.ip, self.command_port))
        self.data_sock.connect((self.ip, self.data_port))
        logger.info('命令端口, 数据端口已连接')

    def disconnect(self):
        self.command_sock.close()
        self.data_sock.close()
        logger.info('命令端口, 数据端口已断开')

    def calc_crc(self, data: bytes):
        words = struct.unpack(f'<{len(data) // 2}H', data)
        crc = 0
        for word in words:
            crc ^= word
        crc = crc ^ self.msg_scrambler_number
        return struct.pack('<H', crc)

    def record_switch(self, enable: bool):
        frame_length = 16 + 8 * 1
        transaction_id = 0x01  # 帧事务号
        source_addr = 0x00  # 源地址
        dest_addr = 0x00  # 目的地址
        frame_type = 0x01  # 帧类型为配置帧
        config_params_count = 1  # 配置参数个数为 1
        config_param_type = 8  # 用于控制设备执行对应命令
        # 0: 停止采样（无论出于何种模式的采样过程）
        # 1: 开始采样（设备处于手动采样模式下有效）
        config_param_value = int(enable)
        crc = 0
        head = struct.pack('<HHH', self.preamble, frame_length, self.protocol_version)
        head += struct.pack('<BBBB', transaction_id, source_addr, dest_addr, frame_type)
        data_content = struct.pack('<BBHHHI', config_params_count, 0, 0, config_param_type, 0, config_param_value)
        # data_content += struct.pack('<HHI', 8, 0, 4)
        crc = self.calc_crc(head + data_content)
        self.command_sock.send(head + crc + data_content)
        logger.info('开始采样' if enable else '停止采样')

    def read_parse_head(self):
        header = self.data_sock.recv(self.header_len)
        if not header:
            # 第一个参数是is_connected
            logger.info('连接断开')
            return False, None
        self.temp = header
        preamble, frame_length, protocol_version = struct.unpack('<HHH', header)
        # 检查开始字节及协议版本是否正确 (用于判读是否是数据帧头)
        if preamble != self.preamble or protocol_version != self.protocol_version:
            return True, None
        return True, frame_length

    def read_data_frame(self, frame_length: int):
        data = b''
        while len(data) < frame_length - self.header_len:
            chunk = self.data_sock.recv(frame_length - self.header_len - len(data))
            if not chunk:
                # 第一个参数是is_connected
                logger.info('连接断开')
                return False, None
            data += chunk
        # 解析数据帧内容
        meta_data, data_content = data[:6], data[6:]
        transaction_id, source_addr, dest_addr, frame_type, crc = struct.unpack('<BBBBH', meta_data)

        # 检查帧类型是否为实时预览数据帧
        if frame_type != 0x82:
            return True, None

        (
            _,  # 4
            data_byte_num,  # 2. 给出数据字节数, 每个数据点为3字节大端序有符整数, 对应-2.5V-2.5V范围, 按通道排序
            _,  # 2
            sample_offset,  # 8
            channel_mask,  # 12
        ) = struct.unpack('<IHHQ12s', data_content[:28])
        # channel_mask = int.from_bytes(channel_mask, 'little')
        # channels = bin(channel_mask).count('1')
        sample_num = data_byte_num // 3 // self.channels
        preview_data = data_content[28:]
        array_data = []
        for i in range(sample_num):
            start = self.sample_bytes * self.channels * i
            sample_vector = np.array([
                int.from_bytes(preview_data[start + self.sample_bytes * j: start + self.sample_bytes * (j + 1)], byteorder='big', signed=True)
                for j in range(self.sample_bytes)
            ])
            array_data.append(sample_vector / 2 ** 23 * 2.5)

        return True, np.array(array_data).astype(np.float32).T  # shape: (channels, sample_num)

    def stack_data(self, data):
        # TODO: 考虑一个线程存一个线程加窗降采样
        self.overlap_win = self.overlap_win[:, :self.overlap]
        self.overlap_win = np.concatenate((self.overlap_win, data), axis=1)
        windowed_data = self.downsampler(torch.tensor(self.overlap_win).float()).numpy()
        if self.array_data_buf.shape[1] == 0:
            self.array_data_buf = windowed_data[:, self.downsampled_overlap - 1:]  # 初始化时overlap_win中为全零
            return None
        self.array_data_buf[:, -self.downsampled_overlap:] += windowed_data[:, :self.downsampled_overlap]
        self.array_data_buf = np.concatenate((self.array_data_buf, windowed_data[:, self.downsampled_overlap:]), axis=1)
        if self.array_data_buf.shape[1] > self.cfg.signal.fs:
            self.array_data_buf = self.array_data_buf[:, -self.cfg.signal.fs:]
            return self.array_data_buf
        else:
            return None


if __name__ == '__main__':
    config = CN({
        'mars': {
            'ip': '10.30.4.77',
            'command_port': 7777,
            'data_port': 7778,
            'fs': 512000,
            'channels': 3,
        },
        'signal': {
            'fs': 340000,
        }
    })
    mars = Mars(config)
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
        print(array_data)
        break
    mars.record_switch(False)
    mars.disconnect()

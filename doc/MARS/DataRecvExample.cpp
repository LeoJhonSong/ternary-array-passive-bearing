#include <iostream>
#include <WinSock2.h>
#include <stdint.h>

#pragma comment(lib,"ws2_32.lib") 
#pragma warning(disable:4996)
#pragma pack(4)
/* Frame Structure */
struct FRAME_STRU
{
    uint16_t preamble;    //固定为0xfefe
    uint16_t msgLen;      //帧长度
    uint16_t version;     //当前版本为0x0901
    uint8_t sequence;     //事务号
    uint8_t srcAddr;      //源设备地址,保留
    uint8_t destAddr;     //目的设备地址,保留
    uint8_t msgType;      //帧类型
    uint16_t crc;          //crc校验
    uint8_t content[0];
};

struct PREVIEW_DATA_STRU
{
    uint8_t reserved[20];    //保留
    uint64_t dataOffset;     //本帧采样点偏移
    uint32_t dataCount;      //本帧包含的样点数
    uint8_t sampleData[1184];//采样数据
};
#pragma pack()

typedef enum
{
    MT_HEARTBEAT = 0x00,      //心跳帧
    MT_CFG_REQ = 0x01,        //岸基配置请求
    MT_CFG_REPLY = 0x81,      //配置回复
    MT_CFG_ERR_REPLY = 0xC1,  //配置错误回复

    MT_PREVIEW_DATA = 0x12
}MSG_TYPE;

/* CRC校验 */
#define MSG_SCRAMBLER_NUMBER 0x5A5C
bool Msg_CheckCrc(uint8_t* pMsg, int len)
{
    uint16_t* pData = (uint16_t*)pMsg;
    int num = len / (sizeof(uint16_t));
    uint16_t crc = 0;

    for (int i = 0; i < num; i++)
    {
        crc ^= pData[i];
    }

    return crc == MSG_SCRAMBLER_NUMBER;
}

#define MAX_BUF_SIZE 1024*10      /* 接收缓冲区大小 10K */
#define MAX_MSG_FRAME_SIZE 1500U  //每个数据帧最大长度为1500
#define MAX_DATA_LEN_PER_MSG (MAX_MSG_FRAME_SIZE - sizeof(MSG_HEADER_STRU))

typedef struct
{
    /* 接收缓冲区 */
    uint8_t recvBuf[MAX_BUF_SIZE];
    uint8_t* readPtr;
    uint8_t* writePtr;
} TCP_CHANNEL_STRU;


void Msg_HandleFrame(FRAME_STRU*frame)
{
    if (frame->msgType == MT_PREVIEW_DATA)
    {
        PREVIEW_DATA_STRU* previewData = (PREVIEW_DATA_STRU*)frame->content;
        printf("sample data recvd, offset:%llu len:%d\n", previewData->dataOffset, previewData->dataCount);
    }
}

void dataRecv(TCP_CHANNEL_STRU *channel)
{
    /* 组帧处理 */
    bool frameHandled = false;

    while (channel->readPtr <= (channel->writePtr - sizeof(FRAME_STRU)))
    {
        uint8_t* ptr = channel->readPtr;

        if ((ptr[0] == 0xfe) &&  /* find header */
            (ptr[1] == 0xfe) &&
            (ptr[2] != 0xfe))
        {
            FRAME_STRU* frame = (FRAME_STRU*)ptr;

            if (frame->msgLen < sizeof(FRAME_STRU) &&
                frame->msgLen > MAX_MSG_FRAME_SIZE)
            {
                /* 错误的帧长度,清空整个缓冲区 */
                channel->writePtr = channel->readPtr = channel->recvBuf;
                break;
            }

            if ((channel->writePtr - channel->readPtr) >= frame->msgLen)
            {
                /* a complete frame has been received */
                /* verify checksum */
                if (Msg_CheckCrc(ptr, frame->msgLen))
                {
                    /* handle frame */
                    Msg_HandleFrame(frame);
                }

                channel->readPtr += frame->msgLen;
                frameHandled = true;
            }
            else
            {
                break;
            }
        }
        else
        {
            channel->readPtr++;
        }
    }

    if (frameHandled)
    {
        const int byteLeft = channel->writePtr - channel->readPtr;
        memcpy(channel->recvBuf, channel->readPtr, byteLeft);

        channel->writePtr = channel->recvBuf + byteLeft;
        channel->readPtr = channel->recvBuf;
    }
}

#define PORT 7777  //UDP监听端口
#define RECV_BUF_SIZE 1500 //每一帧最大为1200,适当大一点

int main()
{
    TCP_CHANNEL_STRU channel;
    channel.readPtr = channel.writePtr = channel.recvBuf;

    /* 1.初始化winsock */
    WSADATA wsaData;
    WSAStartup(WINSOCK_VERSION, &wsaData);

    /* 2.创建和绑定socket */
    SOCKET sock = INVALID_SOCKET;
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET)
    {
        std::cout << "create socket error" << std::endl;
        return -1;
    }

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    //inet_pton(AF_INET, "10.30.4.31", &addr.sin_addr);
    addr.sin_addr.S_un.S_addr = inet_addr("10.30.4.31");

    int value = 1024*1024;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&value, sizeof(int));
    
    connect(sock, (SOCKADDR*)&addr, sizeof(addr));

    std::cout << "connected" << std::endl;

    /* 3.处理 */
    char buf[RECV_BUF_SIZE];

    while (true)
    {
        int len;

        while ((len = recv(sock, (char *)channel.writePtr, MAX_BUF_SIZE - (channel.writePtr - channel.readPtr), 0)) > 0)
        {
            channel.writePtr += len;
        }

        if (len < 0 && errno != EAGAIN)
        {
            printf("recv error:%s\n", strerror(errno));
        }

        //handle the data
        dataRecv(&channel);
    }

    closesocket(sock);

    WSACleanup();
}

# 2025新地图
# 两层仿射变换 0和300
# 盲区预测
# 卡尔曼滤波
# config
import datetime
import threading
import time
from collections import deque
import serial
from information_ui import draw_information_ui
from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, \
    get_Value, image_control
from MvImport.MvCameraControl_class import *
import cv2
import numpy as np
from detect_function import YOLOv5Detector
from RM_serial_py.ser_api import build_send_packet, receive_packet, Radar_decision, \
    build_data_decision, build_data_radar_all
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:  # 指定 UTF-8 编码
    config = yaml.safe_load(f)

state = config['global']['state']  # R:红方/B:蓝方
assert config['filter']['type'] in ['kalman', 'sliding_window'], "滤波器类型必须为kalman或sliding_window"
if config['filter']['type'] == 'sliding_window':
    assert 1 <= config['filter']['sliding_window']['window_size'] <= 20, "滑动窗口大小应在1-20之间"
# 图像测试模式（获取图像根据自己的设备，在）
camera_mode = config['global']['camera_mode']  # 'test':测试模式,'hik':海康相机,'video':USB相机（videocapture）
if config['global']['use_serial']:
    try:
        ser1 = serial.Serial(
            config['serial']['port'],
            config['serial']['baudrate'],
            timeout=config['serial']['timeout']
        )
        print(f"串口已连接：{config['serial']['port']}")
    except Exception as e:
        print(f"串口连接失败：{str(e)}")
        ser1 = None
        config['global']['use_serial'] = False  # 自动禁用串口功能
else:
    ser1 = None
    print("串口功能已禁用")
save_img = config['global']['save_img']

# 文件路径配置
if state == 'R':
    loaded_arrays = np.load(config['paths']['calibration']['red'])
    map_image = cv2.imread(config['paths']['map_images']['red'])
else:
    loaded_arrays = np.load(config['paths']['calibration']['blue'])
    map_image = cv2.imread(config['paths']['map_images']['blue'])

mask_image = cv2.imread(config['paths']['map_images']['mask'])
map_backup = cv2.imread(config['paths']['map_images']['backup'])

# 导入战场每个高度的不同仿射变化矩阵
M_ground = loaded_arrays[0]  # 地面层、公路层
M_height_r = loaded_arrays[1]  # 中央高地

# 确定地图画面像素，保证不会溢出
height, width = mask_image.shape[:2]
height -= 1
width -= 1

# 初始化战场信息UI（易伤情况、双倍易伤次数、双倍易伤触发状态）
information_ui = np.zeros((config['ui']['info_panel_size'][1],
                           config['ui']['info_panel_size'][0], 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()
double_vulnerability_chance = -1  # 双倍易伤机会数
opponent_double_vulnerability = -1  # 是否正在触发双倍易伤
target = -1  # 飞镖当前瞄准目标（用于触发双倍易伤）
chances_flag = 1  # 双倍易伤触发标志位，需要从1递增，每小局比赛会重置，所以每局比赛要重启程序
vulnerability = [-1, -1, -1, -1, -1, -1]  # 易伤情况

# 加载战场地图
map = map_backup.copy()

# 初始化盲区预测列表
guess_list = {
    "B1": True,
    "B2": True,
    "B3": True,
    "B4": True,
    "B5": True,
    "B6": True,
    "B7": True,
    "R1": True,
    "R2": True,
    "R3": True,
    "R4": True,
    "R5": True,
    "R6": True,
    "R7": True
}
# 上次盲区预测时的标记进度
guess_value = {
    "B1": 0,
    "B2": 0,
    "B7": 0,
    "R1": 0,
    "R2": 0,
    "R7": 0
}
# 当前标记进度（用于判断是否预测正确正确）
guess_value_now = {
    "B1": 0,
    "B2": 0,
    "B7": 0,
    "R1": 0,
    "R2": 0,
    "R7": 0
}

# 机器人名字对应ID
mapping_table = {
    "R1": 1,
    "R2": 2,
    "R3": 3,
    "R4": 4,
    "R5": 5,
    "R6": 6,
    "R7": 7,
    "B1": 101,
    "B2": 102,
    "B3": 103,
    "B4": 104,
    "B5": 105,
    "B6": 106,
    "B7": 107
}

# 盲区预测点位
guess_table = {}
for robot, points in config['blind_zone']['points'].items():
    guess_table[robot] = [tuple(point) for point in points]


class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵 (假设匀速模型)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)

        # 测量矩阵
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

        # 过程噪声协方差
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 测量噪声协方差
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 后验误差协方差
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.last_measurement = None
        self.last_update_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 更新状态转移矩阵的时间参数
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        # 预测
        prediction = self.kf.predict()

        # 更新
        measurement = np.array([[np.float32(measurement[0])],
                                [np.float32(measurement[1])]])
        self.kf.correct(measurement)

        self.last_measurement = measurement

    def get_estimate(self):
        if self.last_measurement is None:
            return None
        state = self.kf.statePost
        return (state[0, 0], state[1, 0])


class EnhancedKalmanFilter(KalmanFilter):
    def __init__(self,
                 process_noise=float(config['kalman']['process_noise']),
                 measurement_noise=float(config['kalman']['measurement_noise'])):
        super().__init__(process_noise, measurement_noise)
        # 初始速度估计为0
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)

    def update(self, measurement):
        # 首帧特殊处理
        if self.last_measurement is None:
            dx = dy = 0
        else:
            dx = measurement[0] - self.last_measurement[0]
            dy = measurement[1] - self.last_measurement[1]
            dt = time.time() - self.last_update_time
            # 动态调整过程噪声
            dx_val = dx.item() if isinstance(dx, np.ndarray) else dx
            dy_val = dy.item() if isinstance(dy, np.ndarray) else dy
            self.kf.processNoiseCov[2, 2] = min(0.5, 0.1 + abs(dx_val) / 10)
            self.kf.processNoiseCov[3, 3] = min(0.5, 0.1 + abs(dy_val) / 10)

        super().update(measurement)


class KalmanFilterWrapper:
    def __init__(self, max_inactive_time=2.0):
        self.max_inactive_time = max_inactive_time
        self.filters = {}  # 存储不同机器人的卡尔曼滤波器
        self.last_update = {}  # 存储每个机器人的最后更新时间

    def add_data(self, name, x, y, threshold=100000.0):
        if name not in self.filters:
            self.filters[name] = EnhancedKalmanFilter()
            self.last_update[name] = time.time()

        current_filter = self.filters[name]

        # 异常值检测
        if current_filter.last_measurement is not None:
            dx = x - current_filter.last_measurement[0, 0]
            dy = y - current_filter.last_measurement[1, 0]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance > threshold:
                return

        current_filter.update((x, y))
        self.last_update[name] = time.time()
        guess_list[name] = False

    def filter_data(self, name):
        if name not in self.filters:
            return None

        return self.filters[name].get_estimate()

    def get_all_data(self):
        filtered_d = {}
        current_time = time.time()

        to_remove = []
        for name in list(self.filters.keys()):
            if current_time - self.last_update[name] > self.max_inactive_time:
                to_remove.append(name)
                guess_list[name] = True
            else:
                estimate = self.filter_data(name)
                if estimate is not None:
                    filtered_d[name] = estimate
                guess_list[name] = False

        # 清理超时的滤波器
        for name in to_remove:
            del self.filters[name]
            del self.last_update[name]

        return filtered_d


# 添加滑动窗口滤波器
class SlidingWindowFilter:
    def __init__(self, window_size=5, max_inactive_time=2.0, threshold=100000.0):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.threshold = threshold
        self.windows = {}
        self.last_update = {}

    def add_data(self, name, x, y):
        if name not in self.windows:
            self.windows[name] = deque(maxlen=self.window_size)

        # 异常值检测
        if len(self.windows[name]) > 0:
            last_x, last_y = self.windows[name][-1]
            if (x - last_x) ** 2 + (y - last_y) ** 2 > self.threshold:
                return

        self.windows[name].append((x, y))
        self.last_update[name] = time.time()

    def get_all_data(self):
        current_time = time.time()
        filtered = {}

        # 清理过期数据
        to_remove = []
        for name in self.windows:
            if current_time - self.last_update.get(name, 0) > self.max_inactive_time:
                to_remove.append(name)
                guess_list[name] = True

        for name in to_remove:
            del self.windows[name]
            del self.last_update[name]

        # 计算窗口均值
        for name, window in self.windows.items():
            if len(window) >= self.window_size:
                x_avg = sum(p[0] for p in window) / len(window)
                y_avg = sum(p[1] for p in window) / len(window)
                filtered[name] = (x_avg, y_avg)
                guess_list[name] = False

        return filtered


# 滤波器选择
def create_filter(config):
    filter_type = config['filter']['type']

    if filter_type == "kalman":
        return KalmanFilterWrapper(
            max_inactive_time=float(config['filter']['kalman']['max_inactive_time'])
        )
    elif filter_type == "sliding_window":
        return SlidingWindowFilter(
            window_size=int(config['filter']['sliding_window']['window_size']),
            max_inactive_time=float(config['filter']['sliding_window']['max_inactive_time']),
            threshold=float(config['filter']['sliding_window']['threshold'])
        )
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")


# 读取视频帧
def video_loop():
    global camera_image, cap
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束，重置到开头重新播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        camera_image = frame.copy()
        time.sleep(0.03)  # 控制帧率


# 海康相机图像获取线程
def hik_camera_get():
    # 获得设备信息
    global camera_image
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    # nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    while 1:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            # sys.exit()
        else:
            print("Find %d devices!" % deviceList.nDeviceNum)
            break

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    # 手动选择设备
    # nConnectionNum = input("please input the number of the device to connect:")
    # 自动选择设备
    nConnectionNum = '0'
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    print(get_Value(cam, param_type="float_value", node_name="ExposureTime"),
          get_Value(cam, param_type="float_value", node_name="Gain"),
          get_Value(cam, param_type="enum_value", node_name="TriggerMode"),
          get_Value(cam, param_type="float_value", node_name="AcquisitionFrameRate"))

    # 海康相机线程中的硬编码参数
    set_Value(cam, param_type="float_value", node_name="ExposureTime",
              node_value=config['camera_params']['exposure_time'])
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=config['camera_params']['gain'])
    # 开启设备取流
    start_grab_and_get_data_size(cam)
    # 主动取流方式抓取图像
    stParam = MVCC_INTVALUE_EX()

    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(pData)
            # 处理海康相机的图像格式为OPENCV处理的格式
            camera_image = image_control(data=image, stFrameInfo=stFrameInfo)
        else:
            print("no data[0x%x]" % ret)


def video_capture_get():
    global camera_image
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if ret:
            camera_image = img
            time.sleep(0.016)  # 60fps


# 串口发送线程
def ser_send():
    if not ser1:  # 检查串口是否可用
        print("串口未启用，发送线程退出")
        return
    seq = 0
    global chances_flag
    global guess_value
    # 单点预测时间
    guess_time = {
        'B1': 0,
        'B2': 0,
        'B7': 0,
        'R1': 0,
        'R2': 0,
        'R7': 0,
    }
    # 预测点索引
    guess_index = {
        'B1': 0,
        'B2': 0,
        'B7': 0,
        'R1': 0,
        'R2': 0,
        'R7': 0,
    }

    # 发送蓝方机器人坐标
    def send_point_B(send_name, all_filter_data):
        # front_time = time.time()
        # 转换为地图坐标系
        filtered_xyz = (2800 - all_filter_data[send_name][1], all_filter_data[send_name][0])
        # 转换为裁判系统单位M
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
        # 打包坐标数据包
        # data = build_data_radar(mapping_table.get(send_name), ser_x, ser_y)
        # packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        # ser1.write(packet)
        # back_time = time.time()
        # # 计算发送延时，动态调整
        # waste_time = back_time - front_time
        # # print("发送：",send_name, seq_s)
        # time.sleep(0.1 - waste_time)
        return ser_x, ser_y

    # 发送红发机器人坐标
    def send_point_R(send_name, all_filter_data):
        # front_time = time.time()
        # 转换为地图坐标系
        filtered_xyz = (all_filter_data[send_name][1], 1500 - all_filter_data[send_name][0])
        # 转换为裁判系统单位M
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
        # 打包坐标数据包
        # data = build_data_radar(mapping_table.get(send_name), ser_x, ser_y)
        # packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        # ser1.write(packet)
        # back_time = time.time()
        # # 计算发送延时，动态调整
        # waste_time = back_time - front_time
        # # print('发送：',send_name, seq_s)
        # time.sleep(0.1 - waste_time)
        return ser_x, ser_y

    # 发送盲区预测点坐标
    def send_point_guess(send_name, guess_time_limit):
        # front_time = time.time()
        # print(guess_value_now.get(send_name),guess_value.get(send_name) ,guess_index[send_name])
        # 进度未满 and 预测进度没有涨 and 超过单点预测时间上限，同时满足则切换另一个点预测
        if guess_value_now.get(send_name) < 120 and guess_value_now.get(send_name) - guess_value.get(
                send_name) <= 0 and time.time() - guess_time.get(send_name) >= guess_time_limit:
            guess_index[send_name] = 1 - guess_index[send_name]  # 每个ID不一样
            guess_time[send_name] = time.time()
        if guess_value_now.get(send_name) - guess_value.get(send_name) > 0:
            guess_time[send_name] = time.time()
        # 打包坐标数据包
        # data = build_data_radar(mapping_table.get(send_name), guess_table.get(send_name)[guess_index.get(send_name)][0],
        #                         guess_table.get(send_name)[guess_index.get(send_name)][1])
        # packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        # ser1.write(packet)
        # back_time = time.time()
        # 计算发送延时，动态调整
        # waste_time = back_time - front_time
        # print('发送：',send_name, seq_s)
        # time.sleep(0.1 - waste_time)
        return guess_table.get(send_name)[guess_index.get(send_name)][0], \
            guess_table.get(send_name)[guess_index.get(send_name)][1]

    time_s = time.time()
    target_last = 0  # 上一帧的飞镖目标
    update_time = 0  # 上次预测点更新时间
    send_count = 0  # 信道占用数，上限为4
    send_map = {
        "R1": (0, 0),
        "R2": (0, 0),
        "R3": (0, 0),
        "R4": (0, 0),
        "R5": (0, 0),
        "R6": (0, 0),
        "R7": (0, 0),
        "B1": (0, 0),
        "B2": (0, 0),
        "B3": (0, 0),
        "B4": (0, 0),
        "B5": (0, 0),
        "B6": (0, 0),
        "B7": (0, 0)
    }
    while True:

        guess_time_limit = config['blind_zone']['base_time'] + config['blind_zone'][
            'offset_time']  # 单位：秒，根据上一帧的信道占用数动态调整单点预测时间
        # print(guess_time_limit)
        send_count = 0  # 重置信道占用数
        try:
            all_filter_data = filter.get_all_data()
            if state == 'R':
                if not guess_list.get('B1'):
                    if all_filter_data.get('B1', False):
                        send_map['B1'] = send_point_B('B1', all_filter_data)
                else:
                    send_map['B1'] = (0, 0)

                if not guess_list.get('B2'):
                    if all_filter_data.get('B2', False):
                        send_map['B2'] = send_point_B('B2', all_filter_data)
                else:
                    send_map['B2'] = (0, 0)

                # 步兵3号
                if not guess_list.get('B3'):
                    if all_filter_data.get('B3', False):
                        send_map['B3'] = send_point_B('B3', all_filter_data)
                else:
                    send_map['B3'] = (0, 0)

                # 步兵4号
                if not guess_list.get('B4'):
                    if all_filter_data.get('B4', False):
                        send_map['B4'] = send_point_B('B4', all_filter_data)
                else:
                    send_map['B4'] = (0, 0)

                if not guess_list.get('B5'):
                    if all_filter_data.get('B5', False):
                        send_map['B5'] = send_point_B('B5', all_filter_data)
                else:
                    send_map['B5'] = (0, 0)

                # 哨兵
                if guess_list.get('B7'):
                    send_map['B7'] = send_point_guess('B7', guess_time_limit)
                # 未识别到哨兵，进行盲区预测
                else:
                    if all_filter_data.get('B7', False):
                        send_map['B7'] = send_point_B('B7', all_filter_data)

            if state == 'B':
                if not guess_list.get('R1'):
                    if all_filter_data.get('R1', False):
                        send_map['R1'] = send_point_R('R1', all_filter_data)
                else:
                    send_map['R1'] = (0, 0)

                if not guess_list.get('R2'):
                    if all_filter_data.get('R2', False):
                        send_map['R2'] = send_point_R('R2', all_filter_data)
                else:
                    send_map['R2'] = (0, 0)

                # 步兵3号
                if not guess_list.get('R3'):
                    if all_filter_data.get('R3', False):
                        send_map['R3'] = send_point_R('R3', all_filter_data)
                else:
                    send_map['R3'] = (0, 0)

                # 步兵4号
                if not guess_list.get('R4'):
                    if all_filter_data.get('R4', False):
                        send_map['R4'] = send_point_R('R4', all_filter_data)
                else:
                    send_map['R4'] = (0, 0)

                if not guess_list.get('R5'):
                    if all_filter_data.get('R5', False):
                        send_map['R5'] = send_point_R('R5', all_filter_data)
                else:
                    send_map['R5'] = (0, 0)

                # 哨兵
                if guess_list.get('R7'):
                    send_map['R7'] = send_point_guess('R7', guess_time_limit)
                # 未识别到哨兵，进行盲区预测
                else:
                    if all_filter_data.get('R7', False):
                        send_map['R7'] = send_point_R('R7', all_filter_data)

            ser_data = build_data_radar_all(send_map, state)
            packet, seq = build_send_packet(ser_data, seq, [0x03, 0x05])
            ser1.write(packet)
            time.sleep(0.2)
            print(send_map, seq)
            # 超过单点预测时间上限，更新上次预测的进度
            if time.time() - update_time > guess_time_limit:
                update_time = time.time()
                if state == 'R':
                    guess_value['B1'] = guess_value_now.get('B1')
                    guess_value['B2'] = guess_value_now.get('B2')
                    guess_value['B7'] = guess_value_now.get('B7')
                else:
                    guess_value['R1'] = guess_value_now.get('R1')
                    guess_value['R2'] = guess_value_now.get('R2')
                    guess_value['R7'] = guess_value_now.get('R7')

            # 判断飞镖的目标是否切换，切换则尝试发动双倍易伤
            if target != target_last and target != 0:
                target_last = target
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 发送时间间隔为10秒
                    if time_e - time_s > 10:
                        print("请求双倍触发")
                        data = build_data_decision(chances_flag, state)
                        packet, seq = build_send_packet(data, seq, [0x03, 0x01])
                        # print(packet.hex(),chances_flag,state)
                        ser1.write(packet)
                        print("请求成功", chances_flag)
                        # 更新标志位
                        chances_flag += 1
                        if chances_flag >= 3:
                            chances_flag = 1

                        time_s = time.time()
        except Exception as r:
            print('未知错误 %s' % (r))


# 裁判系统串口接收线程
def ser_receive():
    if not ser1:  # 检查串口是否可用
        print("串口未启用，接收线程退出")
        return
    global vulnerability  # 标记进度列表
    global double_vulnerability_chance  # 拥有双倍易伤次数
    global opponent_double_vulnerability  # 双倍易伤触发状态
    global target  # 飞镖当前目标
    progress_cmd_id = [0x02, 0x0C]  # 任意想要接收数据的命令码，这里是雷达标记进度的命令码0x020E
    vulnerability_cmd_id = [0x02, 0x0E]  # 双倍易伤次数和触发状态
    target_cmd_id = [0x01, 0x05]  # 飞镖目标
    buffer = b''  # 初始化缓冲区
    while True:
        # 从串口读取数据
        received_data = ser1.read_all()  # 读取一秒内收到的所有串口数据
        # 将读取到的数据添加到缓冲区中
        buffer += received_data

        # 查找帧头（SOF）的位置
        sof_index = buffer.find(b'\xA5')

        while sof_index != -1:
            # 如果找到帧头，尝试解析数据包
            if len(buffer) >= sof_index + 5:  # 至少需要5字节才能解析帧头
                # 从帧头开始解析数据包
                packet_data = buffer[sof_index:]

                # 查找下一个帧头的位置
                next_sof_index = packet_data.find(b'\xA5', 1)

                if next_sof_index != -1:
                    # 如果找到下一个帧头，说明当前帧头到下一个帧头之间是一个完整的数据包
                    packet_data = packet_data[:next_sof_index]
                    # print(packet_data)
                else:
                    # 如果没找到下一个帧头，说明当前帧头到末尾不是一个完整的数据包
                    break

                # 解析数据包
                progress_result = receive_packet(packet_data, progress_cmd_id,
                                                 info=False)  # 解析单个数据包，cmd_id为0x020E,不输出日志
                vulnerability_result = receive_packet(packet_data, vulnerability_cmd_id, info=False)
                target_result = receive_packet(packet_data, target_cmd_id, info=False)
                # 更新裁判系统数据，标记进度、易伤、飞镖目标
                if progress_result is not None:
                    received_cmd_id1, received_data1, received_seq1 = progress_result
                    # vulnerability = received_data1[0]
                    vulnerability = [((received_data1[0] >> i) & 0x01) * 120 for i in range(5)]
                    if state == 'R':
                        guess_value_now['B1'] = vulnerability[0]
                        guess_value_now['B2'] = vulnerability[1]
                        guess_value_now['B7'] = vulnerability[4]
                    else:
                        guess_value_now['R1'] = vulnerability[0]
                        guess_value_now['R2'] = vulnerability[1]
                        guess_value_now['R7'] = vulnerability[4]
                if vulnerability_result is not None:
                    received_cmd_id2, received_data2, received_seq2 = vulnerability_result
                    received_data2 = list(received_data2)[0]
                    double_vulnerability_chance, opponent_double_vulnerability = Radar_decision(received_data2)
                if target_result is not None:
                    received_cmd_id3, received_data3, received_seq3 = target_result
                    target = (list(received_data3)[1] & 0b1100000) >> 5

                # 从缓冲区中移除已解析的数据包
                buffer = buffer[sof_index + len(packet_data):]

                # 继续寻找下一个帧头的位置
                sof_index = buffer.find(b'\xA5')

            else:
                # 缓冲区中的数据不足以解析帧头，继续读取串口数据
                break
        time.sleep(0.5)


# 创建机器人坐标滤波器
filter = create_filter(config)
print(f"已启用滤波器类型: {config['filter']['type']}")

# 加载模型，实例化机器人检测器和装甲板检测器 yolov5
weights_path = config['paths']['models']['car']
weights_path_next = config['paths']['models']['armor']
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14, ui=True)
detector_next = YOLOv5Detector(weights_path_next, data='yaml/armor.yaml', conf_thres=0.50, iou_thres=0.2,
                               max_det=10,
                               ui=True)

# 串口接收线程
if config['global']['use_serial']:
    thread_receive = threading.Thread(target=ser_receive, daemon=True)
    thread_receive.start()
else:
    print("跳过串口接收线程初始化")

# 串口发送线程
if config['global']['use_serial']:
    thread_list = threading.Thread(target=ser_send, daemon=True)
    thread_list.start()
else:
    print("跳过串口发送线程初始化")

camera_image = None
cap = None

if camera_mode == 'test':
    camera_image = cv2.imread('images/test_image.jpg')
    # cap = cv2.VideoCapture("images/screen_20250524_093555.mp4")
    # if not cap.isOpened():
    #     print("Error: Could not open video.")
    #     exit()
    # thread_camera = threading.Thread(target=video_loop, daemon=True)
    # thread_camera.start()
elif camera_mode == 'hik':
    # 海康相机图像获取线程
    thread_camera = threading.Thread(target=hik_camera_get, daemon=True)
    thread_camera.start()
elif camera_mode == 'video':
    # USB相机图像获取线程
    thread_camera = threading.Thread(target=video_capture_get, daemon=True)
    thread_camera.start()

while camera_image is None:
    print("等待图像。。。")
    time.sleep(0.5)

# 获取相机图像的画幅，限制点不超限
img0 = camera_image.copy()
img_y = img0.shape[0]
img_x = img0.shape[1]
print(img0.shape)

while True:
    # 刷新裁判系统信息UI图像
    information_ui_show = information_ui.copy()
    map = map_backup.copy()
    det_time = 0
    img0 = camera_image.copy()
    ts = time.time()
    # 第一层神经网络识别
    result0 = detector.predict(img0)
    det_time += 1
    for detection in result0:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            # 存储第一次检测结果和区域
            # ROI出机器人区域
            cropped = camera_image[top:top + h, left:left + w]
            cropped_img = np.ascontiguousarray(cropped)
            # --- 新增：收集所有ROI ---
            if 'roi_list' not in locals():
                roi_list = []
                roi_pos = []
            roi_list.append(cropped_img)
            roi_pos.append((left, top, w, h))


    # --- 新增：拼接ROI并批量识别 ---
    def pack_rois_to_canvas(roi_list, roi_pos, canvas_size=(160, 160), padding=10):
        """
        将roi_list拼接到若干张canvas_size的画布上，返回画布列表和每个roi在画布中的位置
        每个ROI四周加padding，避免目标贴边
        """
        canvases = []
        canvas_rois = []  # 每个画布上的roi信息: [(roi_idx, x_with_pad, y_with_pad, w, h, x_raw, y_raw)]
        cur_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        cur_rois = []
        x, y, row_h = 0, 0, 0
        for idx, roi in enumerate(roi_list):
            h, w = roi.shape[:2]
            ph, pw = h + 2 * padding, w + 2 * padding
            if pw > canvas_size[0] or ph > canvas_size[1]:
                continue  # 跳过过大的roi
            if x + pw > canvas_size[0]:
                x = 0
                y += row_h
                row_h = 0
            if y + ph > canvas_size[1]:
                canvases.append(cur_canvas)
                canvas_rois.append(cur_rois)
                cur_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
                cur_rois = []
                x, y, row_h = 0, 0, 0
            # 先填充黑色，再放ROI
            cur_canvas[y:y + ph, x:x + pw] = 0
            cur_canvas[y + padding:y + padding + h, x + padding:x + padding + w] = roi
            cur_rois.append((idx, x + padding, y + padding, w, h, x, y))  # 记录带padding和原始canvas坐标
            x += pw
            row_h = max(row_h, ph)
        if cur_rois:
            canvases.append(cur_canvas)
            canvas_rois.append(cur_rois)
        return canvases, canvas_rois


    if 'roi_list' in locals() and roi_list:
        canvases, canvas_rois = pack_rois_to_canvas(roi_list, roi_pos, padding=10)
        # 显示所有拼接画布
        # for i, canvas in enumerate(canvases):
        #     cv2.imshow(f'canvas_{i}', canvas)
        all_results = []
        for i, (canvas, rois) in enumerate(zip(canvases, canvas_rois)):
            result_n = detector_next.predict(canvas)
            canvas_show = canvas.copy()
            # 画出所有ROI内容区域
            for idx, rx, ry, rw, rh, ox, oy in rois:
                cv2.rectangle(canvas_show, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)  # 红色框表示ROI内容
            # 画出所有检测框
            # for detection1 in result_n:
            #     cls, xywh, conf = detection1
            #     x, y, w, h = xywh
            #     cv2.rectangle(canvas_show, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            #     cv2.putText(canvas_show, f'{cls} {conf:.2f}', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f'canvas_result_{i}', canvas_show)

            # 复原检测框并画到img0上（每个ROI只保留置信度最高的）
            for idx, rx, ry, rw, rh, ox, oy in rois:
                best_det = None
                best_conf = -1
                best_xywh = None
                best_cls = None
                # 遍历所有检测框，找属于该ROI的
                for detection1 in result_n:
                    cls, xywh, conf = detection1
                    x, y, w, h = xywh
                    cx, cy = x + w // 2, y + h // 2
                    if rx <= cx < rx + rw and ry <= cy < ry + rh:
                        if conf > best_conf:
                            best_conf = conf
                            best_det = detection1
                            best_xywh = xywh
                            best_cls = cls
                # 只处理置信度最高的
                if best_det is not None:
                    x, y, w, h = best_xywh
                    left, top, _, _ = roi_pos[idx]
                    x0 = x - rx + left
                    y0 = y - ry + top
                    # 画框
                    cv2.rectangle(img0, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (255, 0, 255), 2)
                    cv2.putText(img0, f'{best_cls} {best_conf:.2f}', (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 255), 2)
                    # 原图中装甲板的中心下沿作为待仿射变化的点
                    t1 = time.time()
                    camera_point = np.array([[[min(x0 + 0.5 * w, img_x), min(y0 + 1.5 * h, img_y)]]], dtype=np.float32)
                    # 先套用地面层仿射变化矩阵
                    mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
                    x_c = max(int(mapped_point[0][0][0]), 0)
                    y_c = max(int(mapped_point[0][0][1]), 0)
                    x_c = min(x_c, width)
                    y_c = min(y_c, height)
                    color = mask_image[y_c, x_c]  # 通过掩码图像，获取地面层的颜色：黑（0，0，0）
                    if color[0] == color[1] == color[2] == 0:
                        X_M = x_c
                        Y_M = y_c
                        # Z_M = 0
                    else:
                        # 不满足则继续套用R型高地层仿射变换矩阵
                        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_height_r)
                        x_c = max(int(mapped_point[0][0][0]), 0)
                        y_c = max(int(mapped_point[0][0][1]), 0)
                        x_c = min(x_c, width)
                        y_c = min(y_c, height)
                        color = mask_image[y_c, x_c]  # 通过掩码图像，获取R型高地层的颜色：绿（0，255，0）
                        X_M = x_c
                        Y_M = y_c
                        # Z_M = 400
                    if isinstance(filter, SlidingWindowFilter):
                        # 滑动窗口需要完整坐标序列
                        filter.add_data(best_cls, X_M, Y_M)
                    else:
                        # 卡尔曼滤波直接更新
                        filter.add_data(best_cls, X_M, Y_M)
        del roi_list
        del roi_pos

    # 获取所有识别到的机器人坐标
    all_filter_data = filter.get_all_data()
    # print(all_filter_data_name)
    if all_filter_data != {}:
        for name, xyxy in all_filter_data.items():
            # print(name, xyxy)
            if xyxy is not None:
                if name[0] == "R":
                    color_m = (0, 0, 255)
                else:
                    color_m = (255, 0, 0)
                if state == 'R':
                    filtered_xyz = (2800 - xyxy[1], xyxy[0])  # 缩放坐标到地图图像
                else:
                    filtered_xyz = (xyxy[1], 1500 - xyxy[0])  # 缩放坐标到地图图像
                # 只绘制敌方阵营的机器人（这里不会绘制盲区预测的机器人）
                if name[0] != state:
                    cv2.circle(map, (int(filtered_xyz[0]), int(filtered_xyz[1])), 15, color_m, -1)  # 绘制圆
                    cv2.putText(map, str(name),
                                (int(filtered_xyz[0]) - 5, int(filtered_xyz[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                    ser_x = int(filtered_xyz[0]) * 10 / 10
                    ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
                    cv2.putText(map, "(" + str(ser_x) + "," + str(ser_y) + ")",
                                (int(filtered_xyz[0]) - 100, int(filtered_xyz[1]) + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    te = time.time()
    t_p = te - ts
    print("fps:", 1 / t_p)  # 打印帧率
    # 绘制UI
    _ = draw_information_ui(vulnerability, state, information_ui_show)
    cv2.putText(information_ui_show, "vulnerability_chances: " + str(double_vulnerability_chance),
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(information_ui_show, "vulnerability_Triggering: " + str(opponent_double_vulnerability),
                (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('information_ui', information_ui_show)
    map_show = cv2.resize(map, tuple(config['ui']['map_display_size']))
    cv2.imshow('map', map_show)
    img0 = cv2.resize(img0, tuple(config['ui']['img_display_size']))
    cv2.imshow('img', img0)
    if save_img:
        img_name1 = "save_img/game1/1/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        img_name2 = "save_img/game1/2/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        cv2.imwrite(img_name1, map_show)
        cv2.imwrite(img_name2, img0)
    key = cv2.waitKey(1)

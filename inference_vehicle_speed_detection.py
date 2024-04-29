from ultralytics import YOLO
from ultralytics.solutions import object_counter
from IPython.display import display, Image
from pytube import YouTube
from PIL import Image
import os
import cv2
import sys
import torch
import numpy as np  # 導入numpy庫

HOME = os.getcwd()
print(HOME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if sys.platform == 'darwin': device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

width = 0
height = 0
region_points_1 = []
region_points_2 = []
region_points_3 = []
region_points_4 = []
region_points_5 = []
region_points_6 = []

half_width = 0
half_height = 0

'''half_region_points_1 = [(100, half_height - 100),
                        (width - 100, half_height)]
half_region_points_2 = [(100, half_height - 100),
                        (width - 100, half_height)]'''

# Load a model
# model_name = os.path.join(HOME, 'yolo', 'PT', 'yolov8m.pt')
model_name = os.path.join(HOME, 'yolo', 'yolov9', 'yolov9c.pt')
# model_name = os.path.join(HOME, 'yolo', 'yolov8 world', 'yolov8m-worldv2.pt')

model = YOLO(model=model_name)  # load an official model
model.to(device, non_blocking=True)

# file_test = ['test/images/4K Video of Highway Traffic!.mp4']
file_test = ['test/images/54 4K Camera Road in Thailand.mp4']

fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
fontScale = 1  # 字體大小
bigger_fontScale = 2  # 字體大小
red_color = (40, 40, 255)
blue_color = (255, 40, 40)
thickness = 2  # 線條的厚度
lineType = cv2.LINE_AA  # 抗鋸齒

count_num = 0
last_speed = 0
'''id_record_in_region1 = set()
id_record_in_region2 = set()
id_record_in_region3 = set()'''
id_record_in_region1 = {}
id_record_in_region2 = {}
id_record_in_region3 = {}


def is_point_in_quadrilateral(px, py, region_points):
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

    ax, ay = region_points[0]
    bx, by = region_points[1]
    cx, cy = region_points[2]
    dx, dy = region_points[3]
    b1 = sign(px, py, ax, ay, bx, by) < 0.0
    b2 = sign(px, py, bx, by, cx, cy) < 0.0
    b3 = sign(px, py, cx, cy, dx, dy) < 0.0
    b4 = sign(px, py, dx, dy, ax, ay) < 0.0

    return ((b1 == b2) and (b2 == b3) and (b3 == b4))


def is_point_in_region(x, y, region_points):
    """
    檢測點 (x, y) 是否在由 (x1, y1) 和 (x2, y2) 定義的矩形內部。

    參數:
    x, y: 點的座標。
    x1, y1: 矩形左下角的座標。
    x2, y2: 矩形右上角的座標。

    返回:
    True 如果點在矩形內部，否則 False。
    """
    x1 = region_points[0][0]
    x2 = region_points[1][0]
    y1 = region_points[0][1]
    y2 = region_points[1][1]
    return x1 < x < x2 and y1 < y < y2


def update_dict(id_record_in_region):
    for key in id_record_in_region.keys():
        id_record_in_region[key] += 1


def draw_boxes(frame, boxes_xyxy, conf, cls, count=True):
    global width, height, half_width, half_height, count_num, region_points_1, region_points_2, \
        region_points_3, region_points_4, region_points_5, region_points_6, \
        id_record_in_region1, id_record_in_region2, id_record_in_region3, fps, last_speed

    joint_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    """
    繪製關鍵點
    :param frame: 視頻幀
    :param keypoint_xy: 關鍵點的(x, y)座標列表
    """
    xy_center = []
    if boxes_xyxy.shape == (0, 4):
        return

    # 確保關鍵點座標是整數
    boxes_xyxy = (boxes_xyxy + np.array([0, half_height, 0, half_height])).astype(int)
    # boxes_xyxy = (boxes_xyxy).astype(int)
    cls = (cls).astype(int)

    cv2.putText(frame, f'count:{count_num} {last_speed}Km/h', (half_width, 50), font, bigger_fontScale, blue_color,
                thickness,
                lineType)
    cv2.polylines(frame, [region_points_1], True, blue_color, thickness)
    cv2.polylines(frame, [region_points_2], True, blue_color, thickness)
    cv2.polylines(frame, [region_points_3], True, blue_color, thickness)
    cv2.polylines(frame, [region_points_4], True, blue_color, thickness)
    cv2.polylines(frame, [region_points_5], True, blue_color, thickness)
    cv2.polylines(frame, [region_points_6], True, blue_color, thickness)

    # for i, (xy, score, label) in enumerate(zip(boxes_xyxy, conf, cls)):
    #     # if label != 0: continue  # 只標注人 #classes中已指定了
    #     # cv2.putText(frame, f'id:{id_count} person {score:.4f}', xy[0], font, fontScale, blue_color, thickness, lineType)
    #     cv2.putText(frame, f'id:{i} person', xy[0], font, fontScale, blue_color, thickness, lineType)
    #
    #     for joint_pair in joint_pairs:
    #         start_point = xy[joint_pair[0]]
    #         end_point = xy[joint_pair[1]]
    #
    #         cv2.line(frame, start_point, end_point, red_color, 2)  # 繪製連線
    update_dict(id_record_in_region1)
    update_dict(id_record_in_region2)
    update_dict(id_record_in_region3)

    for i, (xy, score, label) in enumerate(zip(boxes_xyxy, conf, cls)):
        # if label != 0: continue  # 只標注人 #classes中已指定了
        # cv2.putText(frame, f'id:{id_count} person {score:.4f}', xy[0], font, fontScale, blue_color, thickness, lineType)
        x = int((xy[0] + xy[2]) / 2)
        y = int(xy[3])  # 方框正下方
        xy_center.append((x, y))
        if is_point_in_quadrilateral(x, y, region_points_2):  # 檢測是否在region2
            id_record_in_region1[i] = 0

        if is_point_in_quadrilateral(x, y, region_points_1):  # 檢測是否在region1
            if i in id_record_in_region1:
                last_speed = vehicle_speed(id_record_in_region1[i])
                '''cv2.putText(frame, f'{last_speed}', (xy[2], xy[3]), font, fontScale,
                            blue_color, thickness,
                            lineType)'''
                del id_record_in_region1[i]
                count_num += 1
        if is_point_in_quadrilateral(x, y, region_points_4):  # 檢測是否在region4
            id_record_in_region2[i] = 0

        if is_point_in_quadrilateral(x, y, region_points_3):  # 檢測是否在region3
            if i in id_record_in_region2:
                last_speed = vehicle_speed(id_record_in_region2[i])
                '''cv2.putText(frame, f'{last_speed}', (xy[2], xy[3]), font, fontScale,
                            blue_color, thickness,
                            lineType)'''
                del id_record_in_region2[i]
                count_num += 1
        if is_point_in_quadrilateral(x, y, region_points_6):  # 檢測是否在region6
            id_record_in_region3[i] = 0

        if is_point_in_quadrilateral(x, y, region_points_5):  # 檢測是否在region5
            if i in id_record_in_region3:
                last_speed = vehicle_speed(id_record_in_region3[i])
                '''cv2.putText(frame, f'{last_speed}', (xy[2], xy[3]), font, fontScale,
                            blue_color, thickness,
                            lineType)'''
                del id_record_in_region3[i]
                count_num += 1

        cv2.putText(frame, f'id:{i}', (xy[2], xy[3]), font, fontScale, blue_color, thickness, lineType)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def vehicle_speed(frame_detection):
    global fps
    speed_second_meter = 10 / (frame_detection / fps)
    speed_hour_kilometer = round(speed_second_meter * 3.6, 2)
    print(f"asdf：{speed_hour_kilometer}Km/h")

    return speed_hour_kilometer


def inference_vehicle_speed_detection(file_list, count=True):
    global width, height, half_width, half_height, count_num, region_points_1, region_points_2, \
        region_points_3, region_points_4, region_points_5, region_points_6, \
        id_record_in_region1, id_record_in_region2, id_record_in_region3, fps, last_speed

    for file in file_list:
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            print(f"無法打開視頻檔: {file}")
            continue

        # 獲取視頻的原始幀率
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        half_width = int(width / 2)
        half_width = 1090  # 特定
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        half_height = int(height / 2)
        half_height = 360  # 特定

        x_offset_1 = 240
        x_offset_2 = 275
        x_offset_3 = 220
        x_offset_4 = 250
        x_offset_5 = 210
        x_offset_6 = 225
        y_offset_1 = 100
        y_offset_2 = 85
        y_offset_3 = 100
        y_offset = 12

        region_points_1 = np.array([[100, 490],
                                    [100 + x_offset_1, 475],
                                    [100 + x_offset_1, 475 + y_offset],
                                    [100, 490 + y_offset]], np.int32)  # 1在比較上面

        region_points_2 = np.array([[50, 500 + y_offset_1],
                                    [50 + x_offset_2, 485 + y_offset_1],
                                    [50 + x_offset_2, 485 + y_offset_1 + y_offset],
                                    [50, 500 + y_offset_1 + y_offset]], np.int32)  # 2在比較下麵

        region_points_3 = np.array([[350, 475],
                                    [350 + x_offset_3, 470],
                                    [350 + x_offset_3, 470 + y_offset],
                                    [350, 475 + y_offset]], np.int32)  # 3在比較上面

        region_points_4 = np.array([[335, 500 + y_offset_2],
                                    [335 + x_offset_4, 490 + y_offset_2],
                                    [335 + x_offset_4, 490 + y_offset_2 + y_offset],
                                    [335, 500 + y_offset_2 + y_offset]], np.int32)  # 4在比較下麵

        region_points_5 = np.array([[560, 465],
                                    [560 + x_offset_5, 460],
                                    [560 + x_offset_5, 460 + y_offset],
                                    [560, 465 + y_offset]], np.int32)  # 5在比較上面

        region_points_6 = np.array([[590, 475 + y_offset_3],
                                    [590 + x_offset_6, 470 + y_offset_3],
                                    [590 + x_offset_6, 470 + y_offset_3 + y_offset],
                                    [590, 475 + y_offset_3 + y_offset]], np.int32)  # 6在比較下麵

        while True:
            ret, frame = cap.read()

            if not ret:
                break  # 視頻結束

            # frame_lower_half = frame[half_height:, :]
            frame_sp = frame[half_height:, :half_width]  # 切取左半邊

            with torch.no_grad():
                # 這裡調用模型預測方法，獲取當前幀的關鍵點
                results = model.track(source=frame_sp,
                                      vid_stride=1,
                                      imgsz=1120,
                                      conf=0.4,
                                      device=device,
                                      show_boxes=False,
                                      show_conf=False,
                                      show_labels=False,
                                      show=False,
                                      half=True,
                                      visualize=False,
                                      int8=False,
                                      classes=[2, 3, 5, 7],  # 如果只有一個result.boxes.id永遠都是none
                                      persist=True,  # persist=True 時，這通常意味著追蹤器會在連續的幀之間保持或「記住」追蹤的目標
                                      tracker="bytetrack.yaml")

            result = results[0]
            # frame = result.plot()

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()

            # if boxes_xyxy.shape == (0, 4):
            #     updated_boxes = boxes_xyxy
            # else:
            #     boxes_xyxy = (boxes_xyxy + np.array([0, half_height, 0, half_height]))
            #     # 初始化一個新的列表來保存更新後的座標
            #     updated_boxes = np.empty((len(boxes_xyxy), 4, 2), dtype=np.float32)
            #     for i, xy in enumerate(boxes_xyxy):
            #         # 計算右上和左下的座標
            #         right_top = np.array([xy[2], xy[1]])  # [右下x, 左上y]
            #         left_bottom = np.array([xy[0], xy[3]])  # [左上x, 右下y]
            #
            #         # 組合成完整的座標順序：左上, 右上, 右下, 左下
            #         xy = np.concatenate((xy, right_top, left_bottom))
            #
            #         # 重新排序以符合指定的順序：左上, 右上, 右下, 左下
            #         xy = xy[[0, 1, 4, 5, 2, 3, 6, 7]]
            #         xy = xy.reshape(4, 2)  # 將平坦的座標列表轉換為四個點的座標
            #         # 更新到新的數據結構中
            #         updated_boxes[i] = xy

            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            # 繪製關鍵點

            draw_boxes(frame, boxes_xyxy, conf, cls, True)

            resized_image = cv2.resize(frame, (1600, 900), interpolation=cv2.INTER_LINEAR)

            # 顯示幀
            cv2.imshow('Frame with Keypoints', resized_image)

            # 檢測按鍵事件
            key = cv2.waitKeyEx(1)
            if key == ord('q'):
                break
            elif key == 2555904:  # 右方向鍵
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 100)
            elif key == 2424832:  # 左方向鍵
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 100))

        cap.release()

    cv2.destroyAllWindows()


# keypoints”: [“nose”,“left_eye”,“right_eye”,“left_ear”,“right_ear”,
# “left_shoulder”,“right_shoulder”,“left_elbow”,“right_elbow”,“left_wrist”,
# “right_wrist”,“left_hip”,“right_hip”,“left_knee”,“right_knee”,“left_ankle”,“right_ankle”]

# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
#         10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
#         20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
#         30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
#         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
#         50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#         60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
#         70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# obb_names: {0: 'plane', 1: 'ship', 2: 'storage tank', 3: 'baseball diamond', 4: 'tennis court',
#           5: 'basketball court', 6: 'ground track field', 7: 'harbor', 8: 'bridge', 9: 'large vehicle',
#           10: 'small vehicle', 11: 'helicopter', 12: 'roundabout', 13: 'soccer ball field', 14: 'swimming pool'}
if __name__ == "__main__":
    inference_vehicle_speed_detection(file_test)

import os
import time
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from  ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap, QPainter
from main_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QFileDialog,QDialog,QPushButton
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from webCam import WebCam
from utils import Video

def name_to_dic(names,dic):
    dic["hat"]=0
    dic["person"]=0
    print("name_to_dict")
    for i in names:
        if i==0:
            dic["hat"]+=1
        elif i==1:
            dic["person"]+=1
    #dict["warning"]=dict["person"]+"人未佩戴安全帽"

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    send_warning=pyqtSignal(str)

    def __init__(self,mode="pic"):
        super(DetThread, self).__init__()
        self.weights = "./yolov8n.pt"
        self.mode=mode
        self.source="0"
        self.conf_thres = 0.25
        self.img=None
        self.addr=""
    def run(self):
        print("run----")
        if self.mode=="pic":
            print(self.source)
            print("pic  mode")
            model = YOLO(self.weights)
            #print(model)
            ##类别名称列表
            print("模型已经加载")
            result=model.predict(self.source)[0]  ##返回的是一个yolo.v8.detect.DetectionPredictor类对象#self.img
            ##获取这一帧中所有的类别信息，但是是一个列表
            #print(result.boxes.cls.numpy())
            #result=model.track(source=0, tracker="track.yaml")
            result=result.cuda()
            print(result.boxes.cls.tolist())
            ##类别列表
            cls=result.boxes.cls.tolist()
            names=[int(x) for x in cls]

            raw = result.orig_img
            img=result.plot()


            print("图片image发送完成")
            ##检测到图片，通过信号发送出去
            self.send_img.emit(img)
            self.send_raw.emit(raw)

            statistic_dic ={"hat":0,"person":0}#{name: 0 for name in names}
            ###处理逻辑，，我为佩戴要进行告警
            name_to_dic(names,statistic_dic)
            #statistic_dic["Warning"]=statistic_dic["person"]+"人，未佩戴安全帽"

            # str=str(statistic_dic["person"]) + "人，未佩戴安全帽！"
            # print("sig",str)
            #self.send_warning.emit(str)
            self.send_statistic.emit(statistic_dic)
        elif self.mode=="video":
            model = YOLO(self.weights)
            print("video mode")
            results = model.predict(source="0", stream=True)
            #results=model.track(source=0, tracker="track.yaml",show=True)
            for result in results:
                print("打印results")
                ##每一帧的boxes
                # boxes=result.boxes
                # raw=boxes.data
                raw=result.orig_img
                img=result.plot()

                #result = result.cuda()
                print(result.boxes.cls.tolist())
                ##类别列表
                cls = result.boxes.cls.tolist()
                names = [int(x) for x in cls]
                statistic_dic = {"hat": 0,"person":0}
                name_to_dic(names, statistic_dic)
                print(img)
                self.send_img.emit(img)
                self.send_raw.emit(raw)
                # 将类别信息发送出去
                print(statistic_dic)
                #self.send_warning.emit("Warning" + statistic_dic["person"] + "人，未佩戴安全帽！")
                self.send_statistic.emit(statistic_dic)
        elif self.mode=="webcam":
            model = YOLO(self.weights)
            print("webcam mode",self.source)
            results = model.predict(source=self.source, stream=True)##self.source
            for result in results:
                print("打印results")
                ##每一帧的boxes
                # boxes=result.boxes
                # raw=boxes.data
                raw = result.orig_img

                img = result.plot()
                print(img)
                self.send_img.emit(img)
                self.send_raw.emit(raw)

                cls = result.boxes.cls.tolist()
                names = [int(x) for x in cls]
                statistic_dic = {"hat": 0, "person": 0}  # {name: 0 for name in names}
                ###处理逻辑，，我为佩戴要进行告警
                name_to_dic(names, statistic_dic)
                print(statistic_dic)

                #self.send_warning.emit("Warning"+statistic_dic["person"]+"人，未佩戴安全帽！")

                # 将类别信息发送出去
                self.send_statistic.emit(statistic_dic)
            print("run直接退出")

class  window(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.setupUi(self)
        self.addr1=""
        #self.addr2=""

        self.model = './yolov8n.pt'
        self.det_thread=DetThread()
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_result))
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_raw))
        self.det_thread.send_statistic.connect(self.show_statistic)
        # self.det_thread.send_warning.connect(self.show_warning)
        self.RunProgram.triggered.connect(self.term_or_con)  #执行或者关闭
        self.SelFile.triggered.connect(self.open_file)   #选择文件
        self.SelModel.triggered.connect(self.open_model)  #选择模型
        self.status_bar_init()
        self.cam_switch.triggered.connect(self.camera)   #选择摄像机

        #设置rtsp摄像头
        self.setWebCam.triggered.connect(self.openWebcam)
        self.webcam=WebCam()
        # ##当对话框点击提交之后，会得到webcam的地址
        #self.addr1=self.webcam.addr1
        #self.addr2=self.webcam.addr2
        ##上述操作得到rtsp视频流地址

        ##将地址传递给Video类

        self.img=None
        # ##rtsp视频流
        # self.video=Video(self.addr1)
        # ##设置触发信号的规则
        #
        # ##一但有网络视频流
        # self.video.res.connect(self.recvImg)

        ##需要一个触发的按键进行检测
        self.detectDouble.triggered.connect(self.detectDC)

        self.horizontalSlider.valueChanged.connect(lambda: self.conf_change(self.horizontalSlider))
        self.spinBox.valueChanged.connect(lambda: self.conf_change(self.spinBox))

    ##recvimg
    def recvImg(self,img):
        self.img=img

    ##双摄像头检测
    def detectDC(self):

        if self.detectDouble.isChecked():
            self.det_thread.mode="webcam"
            ##此处传给模型图像数据

            self.det_thread.source=self.webcam.addr1
            print("self.dee_source:",self.det_thread.source)

            self.det_thread.start()
            self.statusbar.showMessage('正在检测 >> 模型：{}，文件：{}'.
                                       format(os.path.basename(self.det_thread.weights),
                                              os.path.basename(self.det_thread.source)
                                                               if os.path.basename(self.det_thread.source) != '0'
                                                               else '摄像头设备'))
        else:
            self.det_thread.terminate()
            ###+++++++++++++++++++++++++++++++后期需要修改关闭摄像头+++++++++++++++++++++++++++++++++++++
            # if hasattr(self.det_thread, 'vid_cap'):
            #     if self.det_thread.vid_cap:
            #         self.det_thread.vid_cap.release()
            self.statusbar.showMessage('结束检测')




    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("icon/bgg.jpg")
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    # 更改置信度
    def conf_change(self, method):
        if method == self.horizontalSlider:
            self.spinBox.setValue(self.horizontalSlider.value())
        if method == self.spinBox:
            self.horizontalSlider.setValue(self.spinBox.value())
        self.det_thread.conf_thres = self.horizontalSlider.value()/100
        self.statusbar.showMessage("置信度已更改为："+str(self.det_thread.conf_thres))

    def openWebcam(self):
        # addr1=self.webcam.edit.text()
        # print(addr1)
        # addr2=self.webcam.edit2.text()
        # print(addr2)
        self.webcam.exec()

    def status_bar_init(self):
        self.statusbar.showMessage('界面已准备')

    def open_file(self):
        source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                           "*.jpg *.png)")
        print(source)
        if source[0]:
            print(source[0])
            self.det_thread.source = source[0]
        self.statusbar.showMessage('加载文件：{}'.format(os.path.basename(self.det_thread.source)
                                                    if os.path.basename(self.det_thread.source) != '0'
                                                    else '摄像头设备'))

    def term_or_con(self):
        if self.RunProgram.isChecked():
            self.det_thread.start()
            self.statusbar.showMessage('正在检测 >> 模型：{}，文件：{}'.
                                       format(os.path.basename(self.det_thread.weights),
                                              os.path.basename(self.det_thread.source)
                                                               if os.path.basename(self.det_thread.source) != '0'
                                                               else '摄像头设备'))

        else:
            self.det_thread.terminate()
            ###+++++++++++++++++++++++++++++++后期需要修改关闭摄像头+++++++++++++++++++++++++++++++++++++
            # if hasattr(self.det_thread, 'vid_cap'):
            #     if self.det_thread.vid_cap:
            #         self.det_thread.vid_cap.release()
            self.statusbar.showMessage('结束检测')

    def open_model(self):
        self.model = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.pt)")[0]
        if self.model:
            self.det_thread.weights = self.model
        self.statusbar.showMessage('加载模型：' + os.path.basename(self.det_thread.weights))

    def camera(self):
        print("choose camera")
        if self.cam_switch.isChecked():
            print("camera choosed")
            self.det_thread.source = '0'
            self.det_thread.mode="video"
            self.statusbar.showMessage('摄像头已打开')
        else:
            self.det_threadm.terminate()
            if hasattr(self.det_thread, 'vid_cap'):
                self.det_thread.vid_cap.release()
            if self.RunProgram.isChecked():
                self.RunProgram.setChecked(False)
            self.statusbar.showMessage('摄像头已关闭')

    def show_statistic(self, statistic_dic):
        print("已经发送了show信号")
        print(statistic_dic)
        try:
            self.listWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [str(i[0]) + '：' + str(i[1]) for i in statistic_dic]

            # results=[]
            # for i in statistic_dic:
            #     results.append(str(str(i[0]) + '：' + str(i[1])))
            #     print(i[0],i[1])
            # #print(statistic_dic[0],":",statistic_dic[1])fg
            s=str(statistic_dic[1][1])+"人，未佩戴"
            results.append(s)


            self.listWidget.addItems(results)

        except Exception as e:
            print(repr(e))




    @staticmethod
    def show_image(img_src, label):
        print("show_image")
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))



if __name__=="__main__":
    app=QApplication(sys.argv)
    w=window()
    w.show()
    app.exec_()
    #
    # self.names = {0: 'person',
    #               1: 'bicycle',
    #               2: 'car',
    #               3: 'motorcycle',
    #               4: 'airplane',
    #               5: 'bus',
    #               6: 'train',
    #               7: 'truck',
    #               8: 'boat',
    #               9: 'traffic light',
    #               10: 'fire hydrant',
    #               11: 'stop sign',
    #               12: 'parking meter',
    #               13: 'bench',
    #               14: 'bird',
    #               15: 'cat',
    #               16: 'dog',
    #               17: 'horse',
    #               18: 'sheep',
    #               19: 'cow',
    #               20: 'elephant',
    #               21: 'bear',
    #               22: 'zebra',
    #               23: 'giraffe',
    #               24: 'backpack',
    #               25: 'umbrella',
    #               26: 'handbag',
    #               27: 'tie',
    #               28: 'suitcase',
    #               29: 'frisbee',
    #               30: 'skis',
    #               31: 'snowboard',
    #               32: 'sports ball',
    #               33: 'kite',
    #               34: 'baseball bat',
    #               35: 'baseball glove',
    #               36: 'skateboard',
    #               37: 'surfboard',
    #               38: 'tennis racket',
    #               39: 'bottle',
    #               40: 'wine glass',
    #               41: 'cup',
    #               42: 'fork',
    #               43: 'knife',
    #               44: 'spoon',
    #               45: 'bowl',
    #               46: 'banana',
    #               47: 'apple',
    #               48: 'sandwich',
    #               49: 'orange',
    #               50: 'broccoli',
    #               51: 'carrot',
    #               52: 'hot dog',
    #               53: 'pizza',
    #               54: 'donut',
    #               55: 'cake',
    #               56: 'chair',
    #               57: 'couch',
    #               58: 'potted plant',
    #               59: 'bed',
    #               60: 'dining table',
    #               61: 'toilet',
    #               62: 'tv',
    #               63: 'laptop',
    #               64: 'mouse',
    #               65: 'remote',
    #               66: 'keyboard',
    #               67: 'cell phone',
    #               68: 'microwave',
    #               69: 'oven',
    #               70: 'toaster',
    #               71: 'sink',
    #               72: 'refrigerator',
    #               73: 'book',
    #               74: 'clock',
    #               75: 'vase',
    #               76: 'scissors',
    #               77: 'teddy bear',
    #               78: 'hair drier',
    #               79: 'toothbrush'}
    #
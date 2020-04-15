#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from QtUI import MainWin
from QtUI import widget1
from QtUI import widget2
from QtUI import RunWidget1
from QtUI import RunWidget2
import sip

class Min(QMainWindow, MainWin.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("机器学习")
        # 为self.frame_2设置堆叠布局
        self.qsl = QStackedLayout(self.frame_2)

        one = RunWidget1.RunWidget1()
        two = RunWidget2.RunWidget2()

        png = QPixmap("../Resource/Icon/机器学习.png")
        # png = QPixmap('Resource/Icon/brain.png')

        self.label.setPixmap(png)

        self.qsl.addWidget(one)
        self.qsl.addWidget(two)
        # MainWin将信号与槽关联
        self.pushButton.clicked.connect(lambda:self.show_panel())
        self.pushButton_2.clicked.connect(lambda:self.show_panel())

    def show_panel(self):  # 按钮信号
        dic = {
            'pushButton': 1,
            'pushButton_2': 0,
        }
        index = dic[self.sender().objectName()]
        self.qsl.setCurrentIndex(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon("../../resource/Icon/brain.png"))  # 主窗口和应用程序图标
    app.setWindowIcon(QIcon("../Resource/Icon/jiqixuexi.svg"))  # 主窗口和应用程序图标
    min_ = Min()
    min_.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RunWidget2.py'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from QtUI import MainWin
from QtUI import widget1
from QtUI import widget2
from PythonFile.opt_calcu import MathOpt
from PythonFile.my_func import my_func_5x
from PythonFile import save_and_load
from PythonFile import save_and_load
from PythonFile import numpy_NN
import json


class RunWidget2(QWidget, widget2.Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model_ = None

        # Widget2将信号与槽关联
        # self.pushButton_5.clicked.connect(lambda: self.start_opt())
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_3.clicked.connect(self.save_setting)
        self.pushButton_4.clicked.connect(self.get_setting)
        self.pushButton_5.clicked.connect(self.start_opt)

    '''打开文件按钮'''
    def openfile(self):
        try:
            file_name, filetype = QFileDialog.getOpenFileName(self, '选择模型文件', '/', 'json files(*.json)')  # @ '' -> '/'
            self.lineEdit.setText(file_name)  # file_name:文件路径
            model_ = save_and_load.load_(file_name)
            self.model_ = save_and_load.json2history(model_)
            input_dim = self.model_['params'][0]['W'].shape[1]
            limitation = str([0] * input_dim)
            self.lineEdit_4.setText(limitation)
            self.lineEdit_3.setText(limitation)
            self.textEdit.setText('例如：填入x[0]+x[1]-1表示前两项项之和等于1')
            self.textEdit_2.setText('例如：填入x[0]+x[1]-1表示前两项项之和小于1')
        except:
            QMessageBox.information(self, "提示", "模型调用失败，请选择正确的模型。", QMessageBox.Ok)
            return

    def start_opt(self):
        try:
            numpy_NN.MODEL = self.model_
            my_opt = MathOpt(func=numpy_NN.trans_predict, lb=self.lineEdit_4.text(), ub=self.lineEdit_3.text(),
                             constraint_eq=self.textEdit.toPlainText(),
                             constraint_ueq=self.textEdit_2.toPlainText(), peak=self.comboBox.currentText(),
                             opt_method=self.comboBox_2.currentText(), repeaT=self.lineEdit_6.text())
            x = my_opt.best_x
            y = my_opt.best_y
            temp = ''
            for _ in range(len(x)):
                temp += 'x[{ind}]:{val}\n'.format(ind=str(_), val=str(round(x[_], 4)))
            self.textEdit_3.setText(temp)
            self.lineEdit_7.setText(str(round(y[0], 5)))
            numpy_NN.MODEL = None
        except:
            QMessageBox.information(self, "提示", "参数优化出错，这可能是由于所选模型和参数设置不匹配引起的。", QMessageBox.Ok)
            return


    def save_setting(self):    #保存设置
        try:
            dictionary = {'up': self.lineEdit_3.text(), 'down': self.lineEdit_4.text(),
                          'equal': self.textEdit.toPlainText(), 'unequal': self.textEdit_2.toPlainText(),
                          'peak': self.comboBox.currentIndex(), 'method': self.comboBox_2.currentIndex(),
                          'repeat': self.lineEdit_6.text()}

            dir_choose = QFileDialog.getExistingDirectory(self, "选取参数保存位置", '')
            self.lineEdit_8.setText(str(dir_choose))
            save_and_load.save_setting(dictionary, str(dir_choose))
            QMessageBox.information(self, "提示", "保存成功", QMessageBox.Ok)
        except:
            QMessageBox.information(self, "提示", "保存失败", QMessageBox.Ok)
            return

    def get_setting(self):    # 调用设置
        try:
            file_name, filetype = QFileDialog.getOpenFileName(self, '选择参数设置文件', '/', 'json files(*.json)')  # @ '' -> '/'
            self.lineEdit_8.setText(file_name)  # file_name:文件路径
            dictionary = save_and_load.load_(file_name)

            self.lineEdit_3.setText(dictionary['up'])
            self.lineEdit_4.setText(dictionary['down'])
            self.textEdit.setText(dictionary['equal'])
            self.textEdit_2.setText(dictionary['unequal'])
            self.comboBox.setCurrentIndex(dictionary['peak'])
            self.comboBox_2.setCurrentIndex(dictionary['method'])
            self.lineEdit_6.setText(dictionary['repeat'])
        except:
            QMessageBox.information(self, "提示", "参数设置失败，请选择匹配的参数文件。", QMessageBox.Ok)
            return



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("../Resource/Icon/jiqixuexi.svg"))  # 主窗口和应用程序图标
    min = RunWidget2()
    min.show()
    sys.exit(app.exec_())

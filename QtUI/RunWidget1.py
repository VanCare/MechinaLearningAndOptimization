#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from QtUI import MainWin
from QtUI import widget1
from QtUI import widget2
from PythonFile import numpy_NN
import numpy as np
import pandas as pd
from PythonFile import save_and_load


class RunWidget1(QWidget, widget1.Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.model = None

        # Widget1将信号与槽关联
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_7.clicked.connect(self.delete_)
        self.pushButton_5.clicked.connect(self.start_training)
        self.pushButton_8.clicked.connect(self.save_model)
        self.pushButton_3.clicked.connect(self.save_setting)
        self.pushButton_4.clicked.connect(self.get_setting)
        self.pushButton_6.clicked.connect(self.load_)

    '''打开文件按钮'''
    def openfile(self):
        try:
            file_name, filetype = QFileDialog.getOpenFileName(self, '选择文件', '/',
                                                              'data files(*.data , *.csv)')  # @ '' -> '/'
            self.lineEdit.setText(file_name)  # file_name:文件路径
            # 表格显示
            all_datas = pd.read_csv(file_name, encoding="gbk").head(10)
            model = pandasModel(all_datas)
            self.tableView.setModel(model)
            # 自变量因变量显示
            _ = len(all_datas.columns)
            ls = all_datas.columns.values.tolist()
            for i in range(_):
                self.comboBox.addItem(ls[i])
                self.listWidget.addItem(ls[i])
        except:
            QMessageBox.information(self, "提示", "数据文件选择出错，请检查所选文件", QMessageBox.Ok)
            return

    # 调用模型
    def load_(self):
        try:
            file_name, filetype = QFileDialog.getOpenFileName(self, '选择文件', '/',
                                                              'json files(*.json)')  # @ '' -> '/'
            self.lineEdit_5.setText(file_name)  # file_name:文件路径
            model_dict = save_and_load.load_(file_name)
            self.lineEdit_8.setText(str(round(float(model_dict['loss'][-1]), 4)))
        except:
            QMessageBox.information(self, "提示", "模型文件选择出错，请检查所选文件", QMessageBox.Ok)
            return

    '''自变量列表删除'''

    def delete_(self):
        row = self.listWidget.currentRow()
        self.listWidget.takeItem(row)

    '''开始训练按钮'''

    def start_training(self):
        try:
            file_name = self.lineEdit.text()
            # 数据集采集
            all_datas = pd.read_csv(file_name, encoding="gbk")
        except:
            QMessageBox.information(self, "提示", "数据文件选择出错，请检查所选文件", QMessageBox.Ok)
            return

        try:
            all_datas = all_datas.astype("float32")
        except:
            QMessageBox.information(self, "提示", "文件中部分数据不是数字，无法训练。", QMessageBox.Ok)
            return

        try:
            _ = self.lineEdit_2.text()
            training_ritio = int(_) / 100
        except:
            QMessageBox.information(self, "提示", "学习比例未填写", QMessageBox.Ok)
            return

        try:
            independent_variable = []
            for i in range(self.listWidget.count()):
                independent_variable.append(self.listWidget.item(i).text())
            dependent_variable = [self.comboBox.currentText()]
            independent_variable_num = len(independent_variable)
        except:
            QMessageBox.information(self, "提示", "自变量和因变量设置有误", QMessageBox.Ok)
            return

        try:
            learning_rate = float(self.lineEdit_3.text())
            number_of_hidden_nodes = list(eval(self.lineEdit_4.text()))
            activation_function = self.comboBox_4.currentText()
            num_epochs = int(self.lineEdit_7.text())
        except:
            QMessageBox.information(self, "提示", "超参数设置错误", QMessageBox.Ok)
            return

        try:
            all_datas = all_datas.sample(frac=1)
            all_datas = all_datas.reset_index()
            n_train = int(training_ritio * len(all_datas))
            train_data = all_datas[0:n_train]
            test_data = all_datas[n_train:-1]
        except:
            QMessageBox.information(self, "提示", "划分训练集测试集失败，请检查数据文件。", QMessageBox.Ok)
            return

        try:
            x_mean = np.mean(train_data[independent_variable].values, axis=0)
            x_std = np.std(train_data[independent_variable].values, axis=0)
            normalization = lambda x: (x - x_mean) / x_std
            train_data.loc[:, independent_variable] = np.apply_along_axis(func1d=normalization, axis=1,
                                                                          arr=train_data[independent_variable].values)
            # 训练集
            train_features = train_data[independent_variable].values
            train_labels = train_data[dependent_variable].values
            # 测试集
            test_features = test_data[independent_variable].values
            test_labels = test_data[dependent_variable].values
        except:
            QMessageBox.information(self, "提示", "训练数据标准化失败，请检查数据文件。", QMessageBox.Ok)
            return

        try:
            nn_architecture = numpy_NN.get_model(independent_variable_num, number_of_hidden_nodes, activation_function)
        except:
            QMessageBox.information(self, "提示", "生成神经网络模型失败，请检查“隐藏层节点数”输入栏的格式。", QMessageBox.Ok)
            return
        # 模型

        try:
            history = numpy_NN.train(X=np.transpose(train_features),
                                     Y=np.transpose(train_labels.reshape((train_labels.shape[0], 1))),
                                     nn_architecture=nn_architecture,
                                     epochs=num_epochs,
                                     learning_rate=learning_rate,
                                     x_std=x_std,
                                     x_mean=x_mean,
                                     X_name=independent_variable,
                                     Y_name=dependent_variable,
                                     seed=1964)
        except:
            QMessageBox.information(self, "提示", "神经网格训练出错，请确认输入信息是否符合规范和要求。", QMessageBox.Ok)
            return
        loss_list = np.squeeze(history['loss']).tolist()
        loss_train = str(round(float(loss_list[-1]), 4))
        loss_test = str(round(float(numpy_NN.predict(history, test_features, test_labels)), 4))
        self.lineEdit_8.setText(loss_train)
        self.lineEdit_6.setText(loss_test)
        self.model = history

    def save_model(self):
        try:
            dir_choose = QFileDialog.getExistingDirectory(self, "选取模型存储位置", r'C:\jupyter file')
            self.lineEdit_9.setText(str(dir_choose))
            # self.model
            dictionary = save_and_load.history2json(self.model)
            save_and_load.save_(dictionary, str(dir_choose))
            QMessageBox.information(self, "提示", "保存成功", QMessageBox.Ok)
        except:
            QMessageBox.information(self, "提示", "模型保存失败", QMessageBox.Ok)
            return

    def save_setting(self):    # 保存设置
        try:
            dictionary = {'ratio': self.lineEdit_2.text(), 'speed': self.lineEdit_3.text(),
                          'hidden_layers': self.lineEdit_4.text(), 'activation': self.comboBox_4.currentIndex(),
                          'repeat': self.lineEdit_7.text()}
            dir_choose = QFileDialog.getExistingDirectory(self, "选取参数保存位置", '')
            self.lineEdit_10.setText(str(dir_choose))
            save_and_load.save_setting(dictionary, str(dir_choose))
            QMessageBox.information(self, "提示", "保存成功", QMessageBox.Ok)
        except:
            QMessageBox.information(self, "提示", "参数保存失败", QMessageBox.Ok)
            return

    def get_setting(self):    # 调用设置
        try:
            file_name, filetype = QFileDialog.getOpenFileName(self, '选择参数设置文件', '/', 'json files(*.json)')  # @ '' -> '/'
            self.lineEdit_10.setText(file_name)  # file_name:文件路径
            dictionary = save_and_load.load_(file_name)

            self.lineEdit_2.setText(dictionary['ratio'])
            self.lineEdit_3.setText(dictionary['speed'])
            self.lineEdit_4.setText(dictionary['hidden_layers'])
            self.comboBox_4.setCurrentIndex(dictionary['activation'])
            self.lineEdit_7.setText(dictionary['repeat'])
        except:
            QMessageBox.information(self, "提示", "文件选取错误，参数设置失败", QMessageBox.Ok)
            return


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("../Resource/Icon/jiqixuexi.svg"))  # 主窗口和应用程序图标
    min = RunWidget1()
    min.show()
    sys.exit(app.exec_())

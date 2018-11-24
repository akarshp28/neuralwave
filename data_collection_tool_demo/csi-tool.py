#!/usr/bin/python3
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import os.path
import pickle
import socket
import _thread
import time
import math
import sys

import matplotlib
matplotlib.use("qt5agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from multiprocessing import Process

class App(QTabWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'CSI Data Collection Tool'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 500
        self.Devices = list()

        self.Port = 5005

        self.initUI()

    def initUI(self):
        #set title
        self.setWindowTitle(self.title)
        #set window size
        self.setGeometry(self.left, self.top, self.width, self.height)
        #set font
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.CollectDataTab = QWidget()
        self.addTab(self.CollectDataTab, "Collect Data")

        self.AddDevicesTab = QWidget()
        self.addTab(self.AddDevicesTab, "Add Devices")

        self.SettingsTab = QWidget()
        self.addTab(self.SettingsTab, "Settings")
 
        self.CollectDataTabUI()
        self.AddDevicesTabUI()
        self.SettingsTabUI()

        #display window
        self.show()

    def AddDevicesTabUI(self):
        self.AddDevicesTab.layout = QWidget(self.AddDevicesTab)

        self.Label14 = QLabel("Name", self.AddDevicesTab.layout)
        self.Label14.setGeometry(QRect(20, 20, 150, 20))
        self.LineEdit7 = QLineEdit(self.AddDevicesTab.layout)
        self.LineEdit7.setGeometry(QRect(120, 20, 150, 20))

        self.Label9 = QLabel("IP Address", self.AddDevicesTab.layout)
        self.Label9.setGeometry(QRect(20, 50, 150, 20))
        self.LineEdit4 = QLineEdit(self.AddDevicesTab.layout)
        self.LineEdit4.setGeometry(QRect(120, 50, 150, 20))
        self.LineEdit4.setInputMask('000.000.000.000')
        
        self.AddDevicesTab.layout.DeviceType = QWidget(self.AddDevicesTab.layout)
        self.AddDevicesTab.layout.DeviceType.setGeometry(20, 80, 600, 20)
        self.Label17 = QLabel("Device Type", self.AddDevicesTab.layout.DeviceType)
        self.Label17.setGeometry(QRect(0, 0, 200, 20))
        self.RadioButton8 = QRadioButton("Transmitter", self.AddDevicesTab.layout.DeviceType)
        self.RadioButton8.setGeometry(QRect(100, 0, 100, 20))
        self.RadioButton9 = QRadioButton("Receiver ", self.AddDevicesTab.layout.DeviceType)
        self.RadioButton9.setGeometry(QRect(200, 0, 100, 20))
        self.RadioButton9.setChecked(True)

        self.PushButton2 = QPushButton("Add", self.AddDevicesTab.layout)
        self.PushButton2.setGeometry(QRect(350, 80, 80, 20))
        self.PushButton2.clicked.connect(self.Add)

        self.PushButton4 = QPushButton("Remove", self.AddDevicesTab.layout)
        self.PushButton4.setGeometry(QRect(530, 440, 80, 20))
        self.PushButton4.clicked.connect(self.Remove)

        self.Label15 = QLabel("Devices", self.AddDevicesTab.layout)
        self.Label15.setStyleSheet("font-weight: bold")
        self.Label15.setGeometry(QRect(20, 190, 150, 20))
        self.Label16 = QLabel("{:<42} {:<41} {:<40}".format("Name", "IP Address", "Type"), self.AddDevicesTab.layout)
        self.Label16.setGeometry(QRect(20, 210, 600, 20))

        self.AddDevicesTab.layout.Name = QListWidget(self.AddDevicesTab.layout)
        self.AddDevicesTab.layout.Name.setGeometry(20, 230, 199, 200)
        self.AddDevicesTab.layout.Name.itemClicked.connect(self.ListSelectName)
        self.AddDevicesTab.layout.IP = QListWidget(self.AddDevicesTab.layout)
        self.AddDevicesTab.layout.IP.setGeometry(220, 230, 199, 200)
        self.AddDevicesTab.layout.IP.itemClicked.connect(self.ListSelectIP)
        self.AddDevicesTab.layout.Type = QListWidget(self.AddDevicesTab.layout)
        self.AddDevicesTab.layout.Type.setGeometry(420, 230, 199, 200)
        self.AddDevicesTab.layout.Type.itemClicked.connect(self.ListSelectType)

    def SettingsTabUI(self):
        self.SettingsTab.layout = QWidget(self.SettingsTab)

        self.SettingsTab.layout.RFBand = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.RFBand.setGeometry(20, 20, 600, 20)
        self.Label4 = QLabel("RF Band", self.SettingsTab.layout.RFBand)
        self.Label4.setGeometry(QRect(0, 0, 200, 20))
        self.RadioButton1 = QRadioButton("2.4 GHz", self.SettingsTab.layout.RFBand)
        self.RadioButton1.setGeometry(QRect(210, 0, 100, 20))
        self.RadioButton1.toggled.connect(self.SetChannels)
        self.RadioButton2 = QRadioButton("5 GHz", self.SettingsTab.layout.RFBand)
        self.RadioButton2.setGeometry(QRect(310, 0, 100, 20))
        self.RadioButton2.toggled.connect(self.SetChannels)

        self.SettingsTab.layout.ChannelWidth = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.ChannelWidth.setGeometry(20, 50, 600, 20)
        self.Label5 = QLabel("Channel Width", self.SettingsTab.layout.ChannelWidth)
        self.Label5.setGeometry(QRect(0, 0, 200, 20))
        self.RadioButton3 = QRadioButton("HT20", self.SettingsTab.layout.ChannelWidth)
        self.RadioButton3.setGeometry(QRect(210, 0, 100, 20))
        self.RadioButton3.toggled.connect(self.SetChannels)
        self.RadioButton4 = QRadioButton("HT40-", self.SettingsTab.layout.ChannelWidth)
        self.RadioButton4.setGeometry(QRect(310, 0, 100, 20))
        self.RadioButton4.toggled.connect(self.SetChannels)
        self.RadioButton5 = QRadioButton("HT40+", self.SettingsTab.layout.ChannelWidth)
        self.RadioButton5.setGeometry(QRect(410, 0, 100, 20))
        self.RadioButton5.toggled.connect(self.SetChannels)

        self.SettingsTab.layout.GuardIntervel = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.GuardIntervel.setGeometry(20, 80, 600, 20)
        self.Label6 = QLabel("Guard Intervel", self.SettingsTab.layout.GuardIntervel)
        self.Label6.setGeometry(QRect(0, 0, 200, 20))
        self.RadioButton6 = QRadioButton("400 ns", self.SettingsTab.layout.GuardIntervel)
        self.RadioButton6.setGeometry(QRect(210, 0, 100, 20))
        self.RadioButton7 = QRadioButton("800 ns", self.SettingsTab.layout.GuardIntervel)
        self.RadioButton7.setGeometry(QRect(310, 0, 100, 20))
        self.RadioButton7.setChecked(True)

        self.SettingsTab.layout.Modulation = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.Modulation.setGeometry(20, 110, 600, 20)
        self.Label8 = QLabel("Modulation Type/Coding Rate", self.SettingsTab.layout.Modulation)
        self.Label8.setGeometry(QRect(0, 0, 200, 20))
        self.ComboBox1 = QComboBox(self.SettingsTab.layout.Modulation)
        self.ComboBox1.setGeometry(QRect(210, 0, 120, 20))
        self.ComboBox1.addItems(["BPSK 1/2", "QPSK 1/2", "QPSK 3/4", "16-QAM 1/2", "16-QAM 3/4", "64-QAM 2/3", "64-QAM 3/4", "64-QAM 5/6"])

        self.SettingsTab.layout.NumberOfSeconds = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.NumberOfSeconds.setGeometry(20, 140, 600, 20)
        self.Label2 = QLabel("Number of seconds", self.SettingsTab.layout.NumberOfSeconds)
        self.Label2.setGeometry(QRect(0, 0, 150, 20))
        self.LineEdit2 = QLineEdit("10", self.SettingsTab.layout.NumberOfSeconds)
        self.LineEdit2.setGeometry(QRect(210, 0, 50, 20))
        self.LineEdit2.setValidator(QIntValidator(1, 86400))
        self.LineEdit2.setMaxLength(5)

        self.SettingsTab.layout.PacketsPerSecond = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.PacketsPerSecond.setGeometry(20, 170, 600, 20)
        self.Label3 = QLabel("Packets per second", self.SettingsTab.layout.PacketsPerSecond)
        self.Label3.setGeometry(QRect(0, 0, 150, 20))
        self.LineEdit3 = QLineEdit("2000", self.SettingsTab.layout.PacketsPerSecond)
        self.LineEdit3.setGeometry(QRect(210, 0, 40, 20))
        self.LineEdit3.setValidator(QIntValidator(1, 2000))
        self.LineEdit3.setMaxLength(4)

        self.SettingsTab.layout.Channel = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.Channel.setGeometry(20, 200, 600, 20)
        self.Label12 = QLabel("Channel", self.SettingsTab.layout.Channel)
        self.Label12.setGeometry(QRect(0, 0, 150, 20))
        self.ComboBox2 = QComboBox(self.SettingsTab.layout.Channel)
        self.ComboBox2.setGeometry(QRect(210, 0, 50, 20))
        self.RadioButton2.setChecked(True)
        self.RadioButton5.setChecked(True)

        self.SettingsTab.layout.Power = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.Power.setGeometry(20, 230, 600, 20)
        self.Label13 = QLabel("Power", self.SettingsTab.layout.Power)
        self.Label13.setGeometry(QRect(0, 0, 150, 20))
        self.SpinBox1 = QSpinBox(self.SettingsTab.layout.Power)
        self.SpinBox1.setGeometry(QRect(210, 0, 50, 20))
        self.SpinBox1.setRange(0, 30)
        self.SpinBox1.setValue(15)

        self.SettingsTab.layout.MAC = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.MAC.setGeometry(20, 260, 600, 20)
        self.Label18 = QLabel("MAC Address", self.SettingsTab.layout.MAC)
        self.Label18.setGeometry(QRect(0, 0, 150, 20))
        self.LineEdit5 = QLineEdit(self.SettingsTab.layout.MAC)
        self.LineEdit5.setGeometry(QRect(210, 0, 130, 20))
        self.LineEdit5.setInputMask('HH:HH:HH:HH:HH:HH')

        self.SettingsTab.layout.Save = QWidget(self.SettingsTab.layout)
        self.SettingsTab.layout.Save.setGeometry(20, 300, 600, 20)
        self.PushButton3 = QPushButton("Save", self.SettingsTab.layout.Save)
        self.PushButton3.setGeometry(QRect(400, 0, 100, 20))
        self.PushButton3.clicked.connect(self.Save)

    def CollectDataTabUI(self):
        self.CollectDataTab.layout = QWidget(self.CollectDataTab)

        self.Label1 = QLabel("Filename", self.CollectDataTab.layout)
        self.Label1.setGeometry(QRect(20, 20, 150, 20))
        self.LineEdit1 = QLineEdit(self.CollectDataTab.layout)
        self.LineEdit1.setGeometry(QRect(100, 20, 150, 20))

        self.PushButton1 = QPushButton("Start", self.CollectDataTab.layout)
        self.PushButton1.setGeometry(QRect(350, 20, 100, 20))
        self.PushButton1.clicked.connect(self.Start)

    def ListSelectName(self):
        row = self.AddDevicesTab.layout.Name.currentRow()
        self.AddDevicesTab.layout.IP.setCurrentRow(row)
        self.AddDevicesTab.layout.Type.setCurrentRow(row)

    def ListSelectIP(self):
        row = self.AddDevicesTab.layout.IP.currentRow()
        self.AddDevicesTab.layout.Name.setCurrentRow(row)
        self.AddDevicesTab.layout.Type.setCurrentRow(row)

    def ListSelectType(self):
        row = self.AddDevicesTab.layout.Type.currentRow()
        self.AddDevicesTab.layout.IP.setCurrentRow(row)
        self.AddDevicesTab.layout.Name.setCurrentRow(row)

    def Add(self):
        if (self.LineEdit7.text() and self.LineEdit4.text() and (self.RadioButton8.isChecked() or self.RadioButton9.isChecked())):
            if not any((d["IP"] == self.LineEdit4.text() or d["Name"] == self.LineEdit7.text()) for d in self.Devices):
                if self.RadioButton9.isChecked():
                    DevType = "Receiver"
                else:
                    DevType = "Transmitter"

                self.Devices.append({"IP":self.LineEdit4.text(), "Name":self.LineEdit7.text(), "Type": DevType})
                self.AddDevicesTab.layout.Name.addItem(self.LineEdit7.text())
                self.AddDevicesTab.layout.IP.addItem(self.LineEdit4.text())
                self.AddDevicesTab.layout.Type.addItem(DevType)

                self.LineEdit7.clear()
                self.LineEdit4.clear()
                    
    def Remove(self):
        row = self.AddDevicesTab.layout.Name.currentRow()
        self.AddDevicesTab.layout.Name.takeItem(row)
        self.AddDevicesTab.layout.IP.takeItem(row)
        self.AddDevicesTab.layout.Type.takeItem(row)
        del self.Devices[row]

    def SetChannels(self):
        self.ComboBox2.clear()

        if (self.RadioButton2.isChecked()):
            if (self.RadioButton5.isChecked() or self.RadioButton4.isChecked()):
                self.ComboBox2.addItems(["36", "44", "52", "60", "100", "108", "116", "124", "132", "140", "149", "157"])   
            else:
                self.ComboBox2.addItems(["36", "40", "44", "48", "52", "56", "60", "64", "100", "104", "108", "112", "116", "120", "124", "128", "132", "136", "140", "149", "153", "157", "161", "165"])   
        else:
            self.ComboBox2.addItems(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"])  

    def Start(self):	
        self.Save()

        time_start = math.floor(time.time())+2
        for ind in range(len(self.Devices)):
            if self.Devices[ind]['Type'] == "Receiver":
                self.Devices[ind]['socket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                self.Devices[ind]['socket'].connect((self.Devices[ind]['IP'],self.Port))
                self.Devices[ind]['socket'].send(str.encode("recv {} {} {}".format(str(self.LineEdit1.text()), str(int(self.LineEdit2.text())), str(time_start))))
                _thread.start_new_thread(self.recv, (ind, time_start, int(self.LineEdit2.text())))

        for ind in range(len(self.Devices)):
            if self.Devices[ind]['Type'] == "Transmitter":
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                s.connect((self.Devices[ind]['IP'],self.Port))
                s.send(str.encode("send {} {} {} {} {} {} {} {} {} {}".format(self.Band5, self.ChannelWidth, self.GI, self.Channel, self.Power, self.Modulation, self.MAC, str(int(int(self.LineEdit2.text()) * (1000000/self.Delay))), self.Delay, str(time_start))))
                s.close()

    def Save(self):
        if (self.RadioButton2.isChecked()):
        	self.Band5 = 1
        else:
        	self.Band5 = 0

        self.Channel = self.ComboBox2.currentText()

        if (self.RadioButton6.isChecked()):
        	self.GI = 1
        else:
        	self.GI = 0

        if (self.RadioButton3.isChecked()):  
        	self.ChannelWidth = "HT20"
        elif (self.RadioButton4.isChecked()):
        	self.ChannelWidth = "HT40-"
        elif (self.RadioButton5.isChecked()):
        	self.ChannelWidth = "HT40+"        

        self.Power = self.SpinBox1.value()
        self.Modulation = int(self.ComboBox1.currentIndex())+18
        self.MAC = self.LineEdit5.text()
        self.Delay = int(1000000/int(self.LineEdit3.text()))

    def showdialog(self, Text, Details=""):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(Text)
        msg.setDetailedText(Details)
        msg.setWindowTitle("Error!!")
        retval = msg.exec_()

    def signbit_convert(self, data, maxbit):
        if (data & (1 << (maxbit - 1))): 
            data -= (1 << maxbit);
        return data

    def get_csi(self, byts):
        BITS_PER_SYMBOL = 10
        BITS_PER_BYTE = 8
        bitmask = (1 << BITS_PER_SYMBOL) - 1

        if (len(byts) > 100):
            cur = 18
            num_tones = int.from_bytes(byts[cur:cur+1], byteorder='big', signed=False)
            cur += 1
            nr = int.from_bytes(byts[cur:cur+1], byteorder='big', signed=False)
            cur += 1
            nc = int.from_bytes(byts[cur:cur+1], byteorder='big', signed=False)
            cur += 6

            if (nr < 4 and nc < 4 and num_tones < 57):
                csi = np.zeros((nr, nc, num_tones), dtype=np.complex64)

                bits_left = 16
                h_data = byts[cur]
                cur += 1
                h_data += (byts[cur] << BITS_PER_BYTE)
                current_data = h_data & ((1 << 16) - 1)

                for i in range(num_tones):
                    for j in range(nc):
                        for k in range(nr):
                            if ((bits_left - BITS_PER_SYMBOL) < 0):
                                h_data = byts[cur]
                                cur += 1
                                h_data += (byts[cur] << BITS_PER_BYTE)
                                current_data += h_data << bits_left
                                bits_left += 16

                            imag = current_data & bitmask
                            imag = self.signbit_convert(imag, BITS_PER_SYMBOL)
                            bits_left -= BITS_PER_SYMBOL
                            current_data = current_data >> BITS_PER_SYMBOL

                            if ((bits_left - BITS_PER_SYMBOL) < 0):
                                h_data = byts[cur]
                                cur += 1
                                h_data += (byts[cur] << BITS_PER_BYTE)
                                current_data += h_data << bits_left
                                bits_left += 16

                            real = current_data & bitmask
                            real = self.signbit_convert(real, BITS_PER_SYMBOL)
                            bits_left -= BITS_PER_SYMBOL
                            current_data = current_data >> BITS_PER_SYMBOL

                            csi[k, j, i] = np.complex(real, imag)
                return csi
        return None

    def recv(self, ind, StartTime, NumberOfSeconds):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        while((time.time() - StartTime) <= NumberOfSeconds+2):
            byts = self.Devices[ind]['socket'].recv(1024)
            csi = self.get_csi(byts)
            if isinstance(csi, np.ndarray):
                ax.clear()
                ax.axis([0, 504, 0, 750])
                ax.plot(np.abs(csi.reshape(-1)), 'C0')
                plt.pause(1e-10)
        plt.close(fig)
        return 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
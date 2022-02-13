import logging as log
import sys
import warnings
from io import BytesIO
import numpy as np
import pandas as pd
import pyqtgraph as pg
import seaborn as sns
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib import pyplot as plt
from pyqtgraph.Qt import QtCore
import threading


from gui import Ui_MainWindow

warnings.filterwarnings("error")
log.basicConfig(filename='mainLogs.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

generating = False


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOpen.triggered.connect(lambda: open_file())
        self.ui.chunks_slider.sliderReleased.connect(lambda: adjustchunks())
        poly_degree_sliders = [self.ui.poly_1, self.ui.poly_2, self.ui.poly_3, self.ui.poly_4, self.ui.poly_5,
                               self.ui.poly_6, self.ui.poly_7, self.ui.poly_8, self.ui.poly_9, self.ui.poly_10]
        poly_degree_labels = [self.ui.p1_label, self.ui.p2_label, self.ui.p3_label, self.ui.p4_label, self.ui.p5_label,
                              self.ui.p6_label, self.ui.p7_label, self.ui.p8_label, self.ui.p9_label, self.ui.p10_label]
        self.ui.poly_1.sliderReleased.connect(lambda: sliderrelesed(1))
        self.ui.poly_2.sliderReleased.connect(lambda: sliderrelesed(2))
        self.ui.poly_3.sliderReleased.connect(lambda: sliderrelesed(3))
        self.ui.poly_4.sliderReleased.connect(lambda: sliderrelesed(4))
        self.ui.poly_5.sliderReleased.connect(lambda: sliderrelesed(5))
        self.ui.poly_6.sliderReleased.connect(lambda: sliderrelesed(6))
        self.ui.poly_7.sliderReleased.connect(lambda: sliderrelesed(7))
        self.ui.poly_8.sliderReleased.connect(lambda: sliderrelesed(8))
        self.ui.poly_9.sliderReleased.connect(lambda: sliderrelesed(9))
        self.ui.poly_10.sliderReleased.connect(lambda: sliderrelesed(10))
        self.ui.generate_btn.pressed.connect(lambda: generate_btn_toggle())
        self.ui.chunck_combobox.currentIndexChanged.connect(lambda: chunk_equation())
        radio_options = [self.ui.x_num_of_chunks, self.ui.x_poly_degree, self.ui.x_overlap, self.ui.y_num_of_chunks,
                         self.ui.y_poly_degree, self.ui.y_overlap]
        self.ui.clear_btn.pressed.connect(lambda: clear_radio())
        self.ui.clip_slider.sliderReleased.connect(
            lambda: self.ui.clip_label.setText(str(self.ui.clip_slider.value() * 10) + "%"))
        self.ui.clip_slider.setValue(10)
        self.ui.clip_slider.sliderReleased.connect(lambda: interpolate("clipping"))
        self.ui.progress_bar.setValue(0)
        self.ui.chunks_slider.valueChanged.connect(lambda: clip_slider_enable())

        polycoff = [[], [], [], [], [], [], [], [], [], []]
        chunks = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        data = {}
        generating = False

        def sliderrelesed(index):
            index = index - 1
            interpolate(str(index))
            current_degree = poly_degree_sliders[index].value()
            poly_degree_labels[index].setText(str(current_degree))
            chunk_equation()

        def open_file():
            filename = QFileDialog.getOpenFileName(filter="csv(*.csv)")
            if filename[1] != "csv(*.csv)":
                error = QMessageBox()
                error.setIcon(QMessageBox.Critical)
                error.setWindowTitle("File format Error!")
                error.setText("Please choose a .csv file!")
                error.exec_()
            else:
                filedata = pd.read_csv(filename[0], header=None, names=["x", "y"])
                data['time'] = filedata['x']
                data["amp"] = filedata['y']
                log.warning("File path = " + filename[0])
                plotsignal()

        def plotsignal():
            self.ui.maingraph_widget.clear()
            self.ui.maingraph_widget.setLimits(xMin=min(data['time']), xMax=max(data['time']), yMin=min(data['amp']),
                                               yMax=max(data['amp']))
            self.ui.maingraph_widget.plot(data['time'], data['amp'], pen="b", name="main")

        def tex2svg(formula):
            fig = plt.figure()
            font = {'family': 'normal',
                    'weight': 'light',
                    'size': 15}

            plt.rc('font', **font)
            fig.text(0, 0, '${}$'.format(formula))
            output = BytesIO()
            fig.savefig(output, transparent=True, format='svg', bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)
            output.seek(0)
            return output.read()

        def adjustchunks():
            self.ui.chunck_combobox.clear()
            self.ui.chunck_combobox.addItem("Choose..")
            chunks_number = self.ui.chunks_slider.value()
            self.ui.num_of_chunks_label.setText(str(chunks_number))
            setPolySliders()
            amp = np.array_split(data["amp"], chunks_number)
            time = np.array_split(data["time"], chunks_number)
            for i in range(chunks_number):
                poly_degree_sliders[i].show()
                poly_degree_labels[i].show()
                self.ui.chunck_combobox.addItem("Chunk " + str(i + 1))
                chunks[i]["amp"] = np.array(amp[i])
                chunks[i]["time"] = np.array(time[i])
                chunks[i]["degree"] = poly_degree_sliders[i].value()
                polycoff[i] = np.polyfit(chunks[i]["time"], chunks[i]["amp"], int(chunks[i]["degree"]))
            interpolate("init")

        def interpolate(method=""):
            if method == "init":
                for i in range(self.ui.chunks_slider.value()):
                    chunks[i]["degree"] = poly_degree_sliders[i].value()
                    polycoff[i] = np.polyfit(chunks[i]["time"], chunks[i]["amp"], chunks[i]["degree"])
                    poly = np.poly1d(polycoff[i])
                    chunks[i]["interpolated"] = poly(chunks[i]["time"])
                    log.warning("chunk slider " + str(i + 1) + " polyorder = " + str(poly_degree_sliders[i].value()))
            elif self.ui.chunks_slider.value() == 1 and method == "clipping":
                clip_value = self.ui.clip_slider.value() / 10
                length = len(data["amp"])
                chunks[0]["amp"] = data["amp"][0:int(length * clip_value)]
                chunks[0]["time"] = data["time"][0:int(length * clip_value)]
                chunks[0]["degree"] = poly_degree_sliders[0].value()
                polycoff[0] = np.polyfit(chunks[0]["time"], chunks[0]["amp"], chunks[0]["degree"])
                log.warning("[Changed clipping]chunk slider 0 polyorder = " + str(
                    poly_degree_sliders[0].value()) + " Clip value = " + str(clip_value * 100) + "%")
                chunks[0]["time"] = data["time"]
                poly = np.poly1d(polycoff[0])
                chunks[0]["interpolated"] = poly(chunks[0]["time"])
                chunks[0]["amp"] = chunks[0]["interpolated"]

            elif method != "clipping":
                index = int(method)
                chunks[index]["degree"] = poly_degree_sliders[index].value()
                try:
                    polycoff[index] = np.polyfit(chunks[index]["time"], chunks[index]["amp"], chunks[index]["degree"])
                except np.RankWarning:
                    error = QMessageBox()
                    error.setWindowTitle("Interpolation Error!")
                    error.setText(f"Please choose a smaller order than {poly_degree_sliders[index].value()} for this "
                                  f"ploynomial since this order will render it ill-conditioned")
                    error.setIcon(QMessageBox.Warning)
                    error.exec_()
                poly = np.poly1d(polycoff[index])
                chunks[index]["interpolated"] = poly(chunks[index]["time"])
                log.warning("[Changed]chunk slider " + str(index + 1) + " polyorder = " + str(
                    poly_degree_sliders[index].value()))
            updategraph()

        def updategraph():
            plotsignal()
            pen = pg.mkPen(color=(255, 0, 0), style=QtCore.Qt.DotLine, width=3)
            for i in range(self.ui.chunks_slider.value()):
                self.ui.maingraph_widget.plot(chunks[i]["time"], chunks[i]["interpolated"], pen=pen)

        def chunk_equation():
            index = self.ui.chunck_combobox.currentIndex() - 1
            if index < 0:
                return
            coff_list = polycoff[index]
            numcoff = len(coff_list) - 2
            equation = f"p(x)_{{{numcoff+1}}}="
            try:
                for i in range(numcoff):
                    equation = equation + f"{round(coff_list[i], 3)}x^{{{numcoff - i+1}}}"
                    if coff_list[i + 1] > 0:
                        equation = equation + f"+"
                equation = equation + f"{round(coff_list[numcoff], 3)}x"
                if coff_list[numcoff+1] > 0:
                    equation = equation + f"+"
                equation = equation + f"{round(coff_list[numcoff+1], 3)}"
                self.ui.fit_eq.load(tex2svg(equation))

                _, temp_error, _, _, _ = np.polyfit(chunks[index]["time"], chunks[index]["amp"],
                                                    chunks[index]["degree"], full=True)
                self.ui.error_precentage_label.setText(str(round(temp_error[0], 3)))
                self.ui.error_precentage_label.setStyleSheet("#error_precentage_label{font-size:12pt;}")
            except:
                pass

        def setPolySliders():
            self.ui.map_widget.load(clear_map())
            for i in range(10):
                poly_degree_sliders[i].hide()
                poly_degree_labels[i].hide()

        def clear_radio():
            self.ui.map_widget.load(clear_map())
            for i in range(6):
                if radio_options[i].isChecked():
                    radio_options[i].setAutoExclusive(False)
                    radio_options[i].setChecked(False)
                    radio_options[i].setAutoExclusive(True)

        def calculateerror(chunks_num=1, poly_order=1, overlaping=0):
            length = len(data["amp"])
            chunk_length = int(length / chunks_num)  # calculate chunk length  ex 500 point
            # overlap length 10% of 500 = 50 pts
            overlap_length = int(chunk_length * overlaping / 10)
            temp_time = data["time"]
            temp_amp = data["amp"]
            # slice time&data into over-lapped arrays
            temp_time = [temp_time[i:i + chunk_length] for i in range(0, length, chunk_length - overlap_length)]
            temp_amp = [temp_amp[i:i + chunk_length] for i in range(0, length, chunk_length - overlap_length)]
            total_error = 0
            for i in range(chunks_num):
                # poly_coff, sum of least squares, rank, singular_values, rcond
                _, temp_error, _, _, _ = np.polyfit(temp_time[i], temp_amp[i], poly_order, full=True)
                if len(temp_error) > 0:
                    total_error = total_error + temp_error

            return total_error[0]

        def generate_map():
            xdata = {}
            ydata = {}
            if self.ui.x_poly_degree.isChecked():
                xdata["data"] = np.arange(1, 11, 1)
                xdata["label"] = "poly_degree"
            elif self.ui.x_overlap.isChecked():
                xdata["data"] = np.arange(0, 5, 0.5)
                xdata["label"] = "overlaping"
            elif self.ui.x_num_of_chunks:
                xdata["data"] = np.arange(1, 11, 1)
                xdata["label"] = "chunks_num"
            if self.ui.y_poly_degree.isChecked():
                ydata["data"] = np.arange(1, 11, 1)
                ydata["label"] = "poly_degree"
            elif self.ui.y_overlap.isChecked():
                ydata["data"] = np.arange(0, 5, 0.5)
                ydata["label"] = "overlaping"
            elif self.ui.y_num_of_chunks:
                ydata["data"] = np.arange(1, 11, 1)
                ydata["label"] = "chunks_num"
            error = [[0 for x in range(10)] for x in range(10)]
            for i in range(10):
                for j in range(10):
                    if xdata["label"] == "poly_degree":
                        if ydata["label"] == "overlaping":
                            error[i][j] = calculateerror(poly_order=xdata["data"][i], overlaping=ydata["data"][j])
                        elif ydata["label"] == "chunks_num":
                            error[i][j] = calculateerror(poly_order=xdata["data"][i], chunks_num=ydata["data"][j])
                    elif xdata["label"] == "overlaping":
                        if ydata["label"] == "poly_degree":
                            error[i][j] = calculateerror(overlaping=xdata["data"][i], poly_order=ydata["data"][j])
                        elif ydata["label"] == "chunks_num":
                            error[i][j] = calculateerror(overlaping=xdata["data"][i], chunks_num=ydata["data"][j])
                    elif xdata["label"] == "chunks_num":
                        if ydata["label"] == "poly_degree":
                            error[i][j] = calculateerror(chunks_num=xdata["data"][i], poly_order=ydata["data"][j])
                        elif ydata["label"] == "overlaping":
                            error[i][j] = calculateerror(chunks_num=xdata["data"][i], overlaping=ydata["data"][j])
            self.ui.map_widget.load(get_error_map(error, xdata, ydata))

        def progressbar():
            global generating
            if generating:
                self.ui.progress_bar.setValue(self.ui.progress_bar.value() + 1)
                if self.ui.progress_bar.value() == 100:
                    mapThread= threading.Thread(target=generate_map())
                    mapThread.start()
                    generate_btn_toggle()
            else:
                self.timer.stop()
                self.ui.progress_bar.setValue(0)



        def thread():
            self.timer = QTimer()
            self.timer.setInterval(50)
            self.timer.timeout.connect(lambda: progressbar())
            self.timer.start()

        def generate_btn_toggle():
            global generating
            if not generating:
                self.ui.generate_btn.setText("Cancel")
                generating = True
                thread()
            else:
                generating = False
                self.ui.generate_btn.setText("Generate Map")

        def get_error_map(error, xdata, ydata):
            fig = plt.figure()
            fig.ax = sns.heatmap(error, vmin=np.min(error), vmax=np.max(error), cmap="gray",
                                 center=0, linewidths=.2, square=True, xticklabels=xdata["data"],
                                 yticklabels=ydata["data"])
            plt.xlabel(xdata["label"])
            plt.ylabel(ydata["label"])
            output = BytesIO()
            fig.savefig(output, transparent=True, format='svg', bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
            output.seek(0)
            return output.read()
        def clear_map():
            fig=plt.figure()
            error =  [ [0] * 10 for _ in range(10)]

            fig.ax = sns.heatmap(error, cmap="gray",vmin=0,vmax=0,
                                 center=0, linewidths=.2, square=True)
            output = BytesIO()
            fig.savefig(output, transparent=True, format='svg', bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
            output.seek(0)
            return output.read()
        err_eqn = '\sum_{i=1}^{n}(y_i-p(x_i))^2='
        self.ui.Error_eq.load(tex2svg(err_eqn))
        eqn_placeholder = 'p(x)_n=a_nx^n+a_{n-1}x^{n-1}+a_{n-2}x^{n-2}+...+a_1x+a_0'
        self.ui.fit_eq.load(tex2svg(eqn_placeholder))
        self.ui.map_widget.load(clear_map())

        def clip_slider_enable():
            if self.ui.chunks_slider.value() != 1:
                self.ui.clip_slider.setEnabled(False)
            else:
                self.ui.clip_slider.setEnabled(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()

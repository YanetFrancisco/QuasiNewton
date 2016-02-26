__author__ = 'Roly'
import sys
from Quasi_Newton import QuasiNewton
from PyQt4.QtGui import *
from PyQt4 import uic


class PSO_Optimizacion(QMainWindow):
    def __init__(self, parent=None):
        super(PSO_Optimizacion, self).__init__(parent)
        self.ui = uic.loadUi("PSO_Optimizacion.ui", self)
        #self.ui.setupUi(self)

        self.ui.cmBoxTipo.addItems(["Global", "Local"])

        self.ui.pBtnPSO.clicked.connect(self.Resolver)


    def ActInercia(self, value):
        self.ui.lblInerValue.setText(str(value))

    def ActCogn(self, value):
            self.ui.lblCognValue.setText(str(value))

    def ActSocial(self, value):
            self.ui.lblSocValue.setText(str(value))

    def Resolver(self):
        function = "x1+x2"
        particles_count = int(self.ui.linECantPart.text())
        low_bound = int(self.ui.linEMin.text())
        up_bound = int(self.ui.linEMax.text())
        w = self.ui.dSpinInercia.value()
        especial_param_p = self.ui.dSpinCognicion.value()
        especial_param_g = self.ui.dSpinSocial.value()
        stop_case = int(self.ui.linEIteraciones.text())
        g = False
        if self.ui.cmBoxTipo.currentText() == "Global":
            g = True
            pso = QuasiNewton(up_bound, low_bound, w, especial_param_p, especial_param_g, function, particles_count, stop_case)
        else:
            pso = QuasiNewton(up_bound, low_bound, w, especial_param_p, especial_param_g, function, particles_count, stop_case)
        #pso = PSO(f_str, particulas, b_lo, b_up, w, fi_p, fi_g, iteraciones, glob)
        minimo = pso.App(g)





app = QApplication(sys.argv)
pso = PSO_Optimizacion()
pso.show()
sys.exit(app.exec_())
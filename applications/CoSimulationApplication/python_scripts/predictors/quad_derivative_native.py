# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_predictor import CoSimulationPredictor
import KratosMultiphysics as KM

# Other imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import numpy as np
from collections import deque

def Create(settings, solver_wrapper, solY):
    cs_tools.SettingsTypeCheck(settings)
    return NativeQuadraticDerivativePredictor(settings, solver_wrapper, solY)

class NativeQuadraticDerivativePredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper, solY):
        super().__init__(settings, solver_wrapper)

        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.end_time = self.settings["prediction_end_time"].GetDouble()
        self.previousX = None
        self.secondPreviousX = None
        self.thirdPreviousX = None
        self.surrJac = None
        self.surrQ = None
        self.surrR = None
        self.deltaX = None
        self.X_tilde = deque(maxlen=2)
        self.R = deque(maxlen=2)
        self.interface_data_vel = solver_wrapper.GetInterfaceData("velocity")
        self.interface_data_accel = solver_wrapper.GetInterfaceData("acceleration")

    def ReceiveTime(self, t):
        self.currentT = t

    def ReceiveNewData(self, newDisp, newLoad):
        pass

    def Predict(self):
        if self.currentT >= self.launch_time and self.currentT < self.end_time:

            if not self.interface_data.IsDefinedOnThisRank(): return

            delta_time = self.interface_data.GetModelPart().ProcessInfo[KM.DELTA_TIME]

            current_data  = self.interface_data.GetData(0)
            current_velocity = self.interface_data_vel.GetData(0)
            current_accel = self.interface_data_accel.GetData(0)

            predicted_data = current_data + delta_time*current_velocity + (delta_time**2)*current_accel/2
            self._UpdateData(predicted_data)

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "prediction_launch_time" : 0.0,
            "prediction_end_time" : 100.0
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

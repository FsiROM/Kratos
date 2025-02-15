# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_predictor import CoSimulationPredictor
import KratosMultiphysics as KM

# Other imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import numpy as np

def Create(settings, solver_wrapper, solY):
    cs_tools.SettingsTypeCheck(settings)
    return QuadraticPredictor(settings, solver_wrapper, solY)

class QuadraticPredictor(CoSimulationPredictor):
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

    def ReceiveTime(self, t):
        self.currentT = t

    def ReceiveNewData(self, newDisp, newLoad):
        pass

    def Predict(self):
        if self.currentT >= self.launch_time and self.currentT < self.end_time:

            if not self.interface_data.IsDefinedOnThisRank(): return

            current_data  = self.interface_data.GetData(0)
            if self.thirdPreviousX is not None:
                previous_data = self.secondPreviousX.ravel()
                previous_data_2 = self.thirdPreviousX.ravel()
            else:
                previous_data = current_data.copy()
                previous_data_2 = current_data.copy()

            predicted_data = 3*current_data - 3*previous_data + previous_data_2
            self._UpdateData(predicted_data)

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.previousX is not None:
            if self.secondPreviousX is not None:
                self.thirdPreviousX = self.secondPreviousX.copy()
            self.secondPreviousX = self.previousX.copy()
        self.previousX = self.interface_data.GetData().copy()

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "prediction_launch_time" : 0.0,
            "prediction_end_time" : 100.0
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

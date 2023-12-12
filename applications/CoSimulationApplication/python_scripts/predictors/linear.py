# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_predictor import CoSimulationPredictor
import KratosMultiphysics as KM

# Other imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import numpy as np

def Create(settings, solver_wrapper):
    cs_tools.SettingsTypeCheck(settings)
    return LinearPredictor(settings, solver_wrapper)

class LinearPredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper):
        super().__init__(settings, solver_wrapper)

        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.previousX = None
        self.surrJac = None
        self.surrQ = None
        self.surrR = None
        self.deltaX = None

    def ReceiveTime(self, t):
        self.currentT = t

    def ReceiveNewData(self, newDisp, newLoad):
        pass

    def Predict(self):
        if self.currentT >= self.launch_time:

            if not self.interface_data.IsDefinedOnThisRank(): return

            current_data  = self.interface_data.GetData(0)
            previous_data  = self.interface_data.GetData(1)

            predicted_data = 2*current_data - previous_data
            #predicted_data = self.savedModes @ self.savedModes.T @ predicted_data

            self._UpdateData(predicted_data)

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "prediction_launch_time" : 100
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

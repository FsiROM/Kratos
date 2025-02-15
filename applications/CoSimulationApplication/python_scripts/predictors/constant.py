# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_predictor import CoSimulationPredictor
import KratosMultiphysics as KM

# Other imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import numpy as np

def Create(settings, solver_wrapper, solY):
    cs_tools.SettingsTypeCheck(settings)
    return ConstantPredictor(settings, solver_wrapper, solY)

class ConstantPredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper, solY):
        super().__init__(settings, solver_wrapper)

        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.previousX = None
        self.secondPreviousX = None
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

            if self.previousX is not None:
                current_data = self.previousX
                self._UpdateData(current_data)

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        self.previousX = self.interface_data.GetData().copy()

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "prediction_launch_time" : 0.0
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

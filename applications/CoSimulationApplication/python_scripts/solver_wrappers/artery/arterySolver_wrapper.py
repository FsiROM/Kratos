# Importing the Kratos Library
import KratosMultiphysics as KM
import KratosMultiphysics.CoSimulationApplication as KMC

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_solver_wrapper import CoSimulationSolverWrapper

# Other imports
from .arterySolver import ArterySolver
from KratosMultiphysics.CoSimulationApplication.utilities.data_communicator_utilities import GetRankZeroDataCommunicator
import numpy as np

def Create(settings, model, solver_name):
    return ArterySolverWrapper(settings, model, solver_name)

class ArterySolverWrapper(CoSimulationSolverWrapper):
    """ This class implements a wrapper for an SDof solver to be used in CoSimulation
    """
    def __init__(self, settings, model, solver_name, model_part_name="Artery"):
        super().__init__(settings, model, solver_name)

        self.mp = self.model.CreateModelPart(model_part_name)

        input_file_name = self.settings["solver_wrapper_settings"]["input_file"].GetString()
        self._artery_solver = self._CreateArterySolver(input_file_name)
        self.mp.ProcessInfo.SetValue(KM.DOMAIN_SIZE, self._artery_solver.N+1)
        self.mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT_X)
        self.mp.AddNodalSolutionStepVariable(KM.PRESSURE)
        self.interface = self.mp.CreateSubModelPart("circumference")
        # self.mp.ProcessInfo[KM.DOMAIN_SIZE] = self._artery_solver.N+1
        for i in range(self._artery_solver.N+1):
            node = self.mp.CreateNewNode(i, self._artery_solver.grid[i, 0],0.,0.)
            node.AddDof(KM.DISPLACEMENT_X)
            node.AddDof(KM.PRESSURE)
            self.interface.AddNode(node)

    @classmethod
    def _CreateArterySolver(cls, input_file_name):
        return ArterySolver(input_file_name)

    def Initialize(self):
        super().Initialize()
        self._artery_solver.Initialize()
        KM.VariableUtils().SetSolutionStepValuesVector(self.interface.Nodes,
                                            KM.DISPLACEMENT_X, 1.*self._artery_solver.GetSolutionStepValue(), 0)
        KM.VariableUtils().SetSolutionStepValuesVector(self.interface.Nodes,
                                            KM.PRESSURE, 1.*self._artery_solver.pressure.copy(), 0)

    def OutputSolutionStep(self):
        self._artery_solver.OutputSolutionStep()

    def AdvanceInTime(self, current_time):
        return self._artery_solver.AdvanceInTime(current_time)

    def SolveSolutionStep(self):
        self._artery_solver.pressure = np.array(KM.VariableUtils().GetSolutionStepValuesVector(
                        self.interface.Nodes, KM.PRESSURE, 0))
        self._artery_solver.SolveSolutionStep()
        KM.VariableUtils().SetSolutionStepValuesVector(self.interface.Nodes,
                                            KM.DISPLACEMENT_X, 1.*self._artery_solver.GetSolutionStepValue(), 0)

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        self._artery_solver.pressure = np.array(KM.VariableUtils().GetSolutionStepValuesVector(
                        self.interface.Nodes, KM.PRESSURE, 0))
        self._artery_solver.FinalizeSolutionStep()

    def _GetDataCommunicator(self):
        # this solver does not support MPI
        return GetRankZeroDataCommunicator()

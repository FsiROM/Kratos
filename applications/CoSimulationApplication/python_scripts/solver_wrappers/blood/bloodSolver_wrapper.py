# Importing the Kratos Library
import KratosMultiphysics as KM
import KratosMultiphysics.CoSimulationApplication as KMC

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_solver_wrapper import CoSimulationSolverWrapper

# Other imports
from .bloodSolver import BloodSolver
from KratosMultiphysics.CoSimulationApplication.utilities.data_communicator_utilities import GetRankZeroDataCommunicator
import numpy as np

def Create(settings, model, solver_name):
    return BloodSolverWrapper(settings, model, solver_name)

class BloodSolverWrapper(CoSimulationSolverWrapper):
    def __init__(self, settings, model, solver_name, model_part_name="Blood"):
        super().__init__(settings, model, solver_name)

        self.mp = self.model.CreateModelPart(model_part_name)
        self.mp.SetBufferSize(2)

        input_file_name = self.settings["solver_wrapper_settings"]["input_file"].GetString()
        self._Blood_solver = self._CreateBloodSolver(input_file_name)
        self.mp.ProcessInfo.SetValue(KM.DOMAIN_SIZE, self._Blood_solver.N+1)
        # self.mp.ProcessInfo[KM.DOMAIN_SIZE] = self._Blood_solver.N+1
        self.mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT_X)
        self.mp.AddNodalSolutionStepVariable(KM.PRESSURE)
        self.mp.AddNodalSolutionStepVariable(KM.VELOCITY_X)
        self.interface = self.mp.CreateSubModelPart("circumference")
        for i in range(self._Blood_solver.N+1):
            node = self.mp.CreateNewNode(i, self._Blood_solver.grid[i, 0],0,0)
            node.AddDof(KM.DISPLACEMENT_X)
            node.AddDof(KM.PRESSURE)
            node.AddDof(KM.VELOCITY_X)
            self.interface.AddNode(node)

    @classmethod
    def _CreateBloodSolver(cls, input_file_name):
        return BloodSolver(input_file_name)

    def Initialize(self):
        super().Initialize()
        self._Blood_solver.Initialize()
        KM.VariableUtils().SetSolutionStepValuesVector(self.interface.Nodes,
                                            KM.DISPLACEMENT_X, 1.*self._Blood_solver.Section.copy(), 0)
        KM.VariableUtils().SetSolutionStepValuesVector(self.interface.Nodes,
                                            KM.PRESSURE, 1.*self._Blood_solver.GetSolutionStepValue(), 0)

    def OutputSolutionStep(self):
        self._Blood_solver.OutputSolutionStep()

    def FinalizeSolutionStep(self):
        self._Blood_solver.FinalizeSolutionStep()

    def AdvanceInTime(self, current_time):
        return self._Blood_solver.AdvanceInTime(current_time)

    def SolveSolutionStep(self):
        self._Blood_solver.Section = np.array(KM.VariableUtils().GetSolutionStepValuesVector(
                        self.interface.Nodes, KM.DISPLACEMENT_X, 0))
        # self._Blood_solver.Section = self.mp[KM.DISPLACEMENT_X]
        self._Blood_solver.SolveSolutionStep()
        # self.mp[KM.REACTION_X] = self._Blood_solver.GetSolutionStepValue()
        KM.VariableUtils().SetSolutionStepValuesVector(self.interface.Nodes,
                                            KM.PRESSURE, 1.*self._Blood_solver.GetSolutionStepValue(), 0)

    def _GetDataCommunicator(self):
        # this solver does not support MPI
        return GetRankZeroDataCommunicator()

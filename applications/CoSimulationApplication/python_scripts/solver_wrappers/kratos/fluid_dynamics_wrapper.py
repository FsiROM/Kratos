# Importing the Kratos Library
import KratosMultiphysics as KM
from KratosMultiphysics.kratos_utilities import CheckIfApplicationsAvailable

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.solver_wrappers.kratos import kratos_base_wrapper

# Importing FluidDynamics
if not CheckIfApplicationsAvailable("FluidDynamicsApplication"):
    raise ImportError("The FluidDynamicsApplication is not available!")
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

# Other imports
from KratosMultiphysics.CoSimulationApplication.utilities import data_communicator_utilities
from collections import deque
import numpy as np

def Create(settings, model, solver_name):
    return FluidDynamicsWrapper(settings, model, solver_name)

class FluidDynamicsWrapper(kratos_base_wrapper.KratosBaseWrapper):
    """This class is the interface to the FluidDynamicsApplication of Kratos"""

    def __init__(self, settings, model, solver_name):
        super().__init__(settings, model, solver_name)
        self.initialize_data()
        self.section_part_created = False

    def extract_nodes(self, coords_data, model_part_name, tol = 5e-3):
        
        if not self.section_part_created:
            section_model_part = self._analysis_stage._GetSolver().GetComputingModelPart().CreateSubModelPart(model_part_name)
            if coords_data.Has("x"):
                for x_coor in coords_data["x"].values():
                    for node in self._analysis_stage._GetSolver().GetComputingModelPart().GetNodes():
                        if np.abs(node.X - x_coor.GetDouble()) < tol:
                            section_model_part.AddNode(node, 0)
            if coords_data.Has("y"):
                for y_coor in coords_data["y"].values():
                    for node in self._analysis_stage._GetSolver().GetComputingModelPart().GetNodes():
                        if np.abs(node.Y - y_coor.GetDouble()) < tol:
                            section_model_part.AddNode(node, 0)
            
            self.section_part_created = True

    def GetSectionData(self, section_variables):
        data_arr = np.array([])
        section_model_part = self._analysis_stage._GetSolver().GetComputingModelPart().GetSubModelPart("section_part")
        for var in section_variables:
            data_arr = np.concatenate((data_arr, KM.VariableUtils().GetSolutionStepValuesVector(section_model_part.GetNodes(), 
                                                              var, 0, 2)))
        return data_arr
    
    def initialize_data(self, ):
        self.load_data = deque()

    def update_load_data(self, current_load):
        self.load_data.append(current_load)
        np.save("./coSimData/load_data_fluid.npy",
                np.asarray(self.load_data)[:, :, 0].T)
        
    def SolveSolutionStep(self):
        super().SolveSolutionStep()

        current_load = self.GetInterfaceData(
            "load").GetData().reshape((-1, 1))
        self.update_load_data(current_load)

    def _CreateAnalysisStage(self):
        return FluidDynamicsAnalysis(self.model, self.project_parameters)

    def _GetDataCommunicator(self):
        if not KM.IsDistributedRun():
            return KM.ParallelEnvironment.GetDataCommunicator("Serial")

        # now we know that Kratos runs in MPI
        parallel_type = self.project_parameters["problem_data"]["parallel_type"].GetString()

        # first check if the solver uses MPI
        if parallel_type != "MPI":
            return data_communicator_utilities.GetRankZeroDataCommunicator()

        # now we know that the solver uses MPI, only question left is whether to use all ranks or a subset
        if self.project_parameters["solver_settings"]["solver_type"].GetString() == "ale_fluid":
            model_import_settings = self.project_parameters["solver_settings"]["fluid_solver_settings"]["model_import_settings"]
        else:
            model_import_settings = self.project_parameters["solver_settings"]["model_import_settings"]
        self._CheckDataCommunicatorIsConsistentlyDefined(model_import_settings, self.settings["mpi_settings"])

        return super()._GetDataCommunicator()

# Importing the Kratos Library
import KratosMultiphysics as KM
import KratosMultiphysics.CoSimulationApplication.colors as colors

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.solver_wrappers.kratos import structural_mechanics_wrapper
from KratosMultiphysics import StructuralMechanicsApplication

# Other imports
import numpy as np
from collections import deque
from rom_am import solid_rom
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import time


def Create(settings, model, solver_name):
    return StructuralROMWrapper(settings, model, solver_name)


class StructuralROMWrapper(structural_mechanics_wrapper.StructuralMechanicsWrapper):

    def __init__(self, settings, model, solver_name):

        super().__init__(settings, model, solver_name)

        self.ModelPart = self._analysis_stage._GetSolver().GetComputingModelPart()
        self.initialize_data()
        self.get_rom_settings()
        self.trained = False
        self.map_used = None
        self._already_recievedData = False
        self.rom_model = None

    def receive_input_data(self, input_data):

        self.current_load = input_data.reshape((-1, 1))
        self._already_recievedData = True

        if self.is_in_collect_data_mode():
            self.update_load_data(self.current_load)
        if self.save_tr_data:
            np.save("./coSimData/load_data.npy",
                    np.asarray(self.load_data)[:, :, 0].T)

    def get_rom_settings(self, ):
        self.launch_time = self.settings["launch_time"].GetDouble()
        self.start_collecting_time = self.settings["start_collecting_time"].GetDouble()
        self.imported_model = self.settings["imported_model"].GetBool()
        self.save_model = self.settings["save_model"].GetBool()
        self.input_data_name = self.settings["input_data"]["data"].GetString()
        self.output_data_name = self.settings["output_data"]["data"].GetString()
        self.interface_only = self.settings["interface_only"].GetBool()
        self.use_map = self.settings["use_map"].GetBool()
        self.save_tr_data = self.settings["save_training_data"].GetBool()
        self.force_norm_regr = self.settings["force_norm_regr"].GetBool()
        self.disp_norm_regr = self.settings["disp_norm_regr"].GetBool()
        self.force_norm = self.settings["force_norm"].GetString()
        self.disp_norm = self.settings["disp_norm"].GetString()
        self.inputReduc_model = None
        self.regression_model = None
        self.outputReduc_model = None

    def initialize_data(self, ):
        self.load_data = deque()
        self.displacement_data = deque()
        self.displacement_data2 = deque()
        self.recons_time = []
        self.RomResiduals = []
        self.EncReconsErr = []

    def is_in_prediction_mode(self, ):
        return self.prediction_mode_strategy()

    def is_in_collect_data_mode(self, ):
        return self.data_collect_mode_strategy()

    def prediction_mode_strategy(self, ):
        # ======= Condition to be met for launching ROM prediction ============
        return self._analysis_stage.time >= self.launch_time

    def data_collect_mode_strategy(self, ):
        # ======= Condition to be met for collecting training data ============
        return self._analysis_stage.time < self.launch_time

    def train_rom(self):

        if not self.trained:
            self.rom_model = solid_rom.solid_ROM()

            # ======= Import a trained ROM model ============
            if self.imported_model:
                file = self.settings["file"]
                import pickle
                with open(file["file_name"].GetString(), 'rb') as inp:
                    self.rom_model = pickle.load(inp)

            # ======= Train a ROM model ============
            else:
                #coords = np.asarray(self.GetInterfaceData(self.output_data_name).model_part.GetNodes())[:, :2]
                self.rom_model.train(np.asarray(self.load_data)[:, :, 0].T, np.asarray(self.displacement_data)[:, :, 0].T,
                                     rank_pres=50, rank_disp=.9999,
                                     map_used = self.map_used,
                                     norm_regr=[self.force_norm_regr, self.disp_norm_regr],
                                     norm=[self.force_norm, self.disp_norm],
                                     forcesReduc_model=self.inputReduc_model, regression_model=self.regression_model,
                                     dispReduc_model=self.outputReduc_model)

                # ======= Save ROM model in a file ============
                if self.save_model:
                    file = self.settings["file"]
                    import pickle
                    with open(file["file_name"].GetString(), 'wb') as outp:
                        pickle.dump(self.rom_model, outp,
                                    pickle.HIGHEST_PROTOCOL)
            self.trained = True
            self.last_fom_step = self.ModelPart.ProcessInfo[KM.STEP]
        else:
            pass

    def rom_output(self, current_load):

        predictedDisp = self.rom_model.pred(current_load).ravel()

        if self.use_map:
            dispArr = np.empty((self.SS, ))
            #disp_arr[self.ids_interface] = predictedDisp
            dispArr[self.ids_interface] = predictedDisp[self.ids_interface]
            return dispArr
        else:
            return predictedDisp

    def update_load_data(self, current_load):
        self.load_data.append(current_load)
        if self.save_tr_data:
            np.save("./coSimData/load_data.npy",
                    np.asarray(self.load_data)[:, :, 0].T)

    def update_disp_data(self, current_disp):
        self.displacement_data.append(current_disp)
        self.displacement_data2.append(self.GetInterfaceData(
                        self.output_data_name).GetData().reshape((-1, 1)))
        if self.save_tr_data:
            np.save("./coSimData/disp_data.npy",
                    np.asarray(self.displacement_data)[:, :, 0].T)
            np.save("./coSimData/disp_interf_data.npy",
                    np.asarray(self.displacement_data2)[:, :, 0].T)

    def FomSolutionStep(self,):
        super().SolveSolutionStep()

    def RomSolutionStep(self,):
        self.train_rom()
        if self._already_recievedData:
            current_load = self.current_load
        else:
            current_load = self.GetInterfaceData(
                self.input_data_name).GetData().reshape((-1, 1))
        predicted_disp = self.rom_output(current_load)

        # ======= Predict The interface displacement only ============
        if self.interface_only:
            self.GetInterfaceData(self.output_data_name).SetData(predicted_disp)

        # ======= Filling the displacement StepValuesVector  ============
        else:
            KM.VariableUtils().SetSolutionStepValuesVector(self.ModelPart.Nodes,
                                                        KM.DISPLACEMENT, 1.*predicted_disp, 0)
            x_vec = self.x0_vec + 1.*predicted_disp
            KM.VariableUtils().SetCurrentPositionsVector(self.ModelPart.Nodes,1.*x_vec)
            self.ModelPart.GetCommunicator().SynchronizeVariable(KM.DISPLACEMENT)

        if not self.use_map:
            # TODO Encoder Error
            # self.EncReconsErr.append(self.rom_model.forcesReduc.EncReconsErr)
            # with open("./coSimData/EncReconsErr.npy", 'wb') as f:
            #     np.save(f, np.array(self.EncReconsErr))
            if self.ToComputeResiduals():
                residuals, _ = self._ComputeResiduals()
                residualsNorm = np.linalg.norm(residuals)
                KM.Logger.PrintInfo("Residual Norm: ", residualsNorm)
                LoadNorm = np.linalg.norm(current_load)
                invLoadNorm = 1.
                if LoadNorm > 1e-8:
                    invLoadNorm = 1/LoadNorm
                relResidualNorm = invLoadNorm * residualsNorm
                KM.Logger.PrintInfo("Residual Rel Norm: ", relResidualNorm)
                # ----------------- Saving Residuals Data -----------------
                self.RomResiduals.append(relResidualNorm)
                with open("./coSimData/RelResidualNorms.npy", 'wb') as f:
                    np.save(f, np.array(self.RomResiduals))
                if False: # TODO Condition to launch the FOM
                    self.FomSolutionStep()

    def SolveSolutionStep(self):
        # ======= Store Load Training Data ============
        if not self.is_in_prediction_mode() and self.is_in_collect_data_mode() and not self._already_recievedData:
            if not self.imported_model:
                current_load = self.GetInterfaceData(
                    self.input_data_name).GetData().reshape((-1, 1))
                self.update_load_data(current_load)

        # ======= Predict using the FOM ============
        if not self.is_in_prediction_mode():
            self.FomSolutionStep()

        # ======= Predict using the ROM ============
        else:
            self.RomSolutionStep()

        # ======= Store Displacement Training Data ============
        self._already_recievedData = False # Resetting receiving data
        if not self.is_in_prediction_mode() and self.is_in_collect_data_mode():
            if not self.imported_model:
                if self.interface_only:
                    current_disp = self.GetInterfaceData(
                        self.output_data_name).GetData().reshape((-1, 1))
                else:
                    current_disp = KM.VariableUtils().GetSolutionStepValuesVector(
                        self.ModelPart.Nodes, KM.DISPLACEMENT, 0, 2)
                    current_disp = np.array(current_disp).reshape((-1, 1))
                self.update_disp_data(current_disp)

    def ToComputeResiduals(self, ):
        #TODO What condition to use in order to compute the residuals ?
        # ======= Condition to be met for computing the residuals ============
        return False

    def _ComputeResiduals(self,):

        residuals = self._analysis_stage._GetSolver()._GetSolutionStrategy().GetSystemVector()
        KM.UblasSparseSpace().SetToZeroVector(residuals)
        # Assemble the global residual vector
        self.BS.BuildRHS(self.SCH, self.ModelPart, residuals)
        residuals=np.array(residuals)
        internalForces = residuals[self.ids_Dirich]
        residuals = residuals[self.ids_NoDirich]

        return residuals, internalForces

    def Initialize(self):
        super().Initialize()
        np.save("./coSimData/coords_interf.npy",
                np.asarray(self.GetInterfaceData(self.output_data_name).model_part.GetNodes())[:, :2])

        self.SS = self.ModelPart.GetCommunicator().GetDataCommunicator().Sum(self.ModelPart.NumberOfNodes() * 2, 0)
        self.SCH = self._analysis_stage._GetSolver()._GetScheme()
        self.BS = self._analysis_stage._GetSolver()._GetBuilderAndSolver()

        # Assigning free and constrained nodes:
        self.saved_out_t = []
        self.ids_interface = []
        self.ids_global = []
        self.ids_Dirich = []

        for node in self.GetInterfaceData(self.output_data_name).model_part.GetNodes():
            self.ids_interface.append(node.Id)
        for node in self.ModelPart.Nodes:
            self.ids_global.append(node.Id)
            if node.GetDof(KM.DISPLACEMENT_X).IsFixed() and node.GetDof(KM.DISPLACEMENT_Y).IsFixed():
                self.ids_Dirich.append(node.Id)

        self.ids_Interior = (~np.in1d(self.ids_global,
                            self.ids_interface + self.ids_Dirich)).nonzero()[0]
        self.ids_interface = np.in1d(self.ids_global,
                            self.ids_interface).nonzero()[0]
        self.ids_NoDirich = (~np.in1d(self.ids_global,
                            self.ids_Dirich)).nonzero()[0]
        self.ids_Dirich = np.in1d(self.ids_global,
                            self.ids_Dirich).nonzero()[0]

        c = np.empty((2*self.ids_interface.size,), dtype=self.ids_interface.dtype)
        cNoDirich = np.empty((2*self.ids_NoDirich.size,), dtype=self.ids_NoDirich.dtype)
        cDirich = np.empty((2*self.ids_Dirich.size,), dtype=self.ids_Dirich.dtype)

        cInterior = np.empty((2*self.ids_Interior.size,), dtype=self.ids_Interior.dtype)

        c[::2] = 2*self.ids_interface
        c[1::2] = 2*self.ids_interface+1
        cNoDirich[::2] = 2*self.ids_NoDirich
        cNoDirich[1::2] = 2*self.ids_NoDirich+1
        cDirich[::2] = 2*self.ids_Dirich
        cDirich[1::2] = 2*self.ids_Dirich+1
        cInterior[::2] = 2*self.ids_Interior
        cInterior[1::2] = 2*self.ids_Interior+1

        self.ids_NoDirich = cNoDirich
        self.ids_Dirich = cDirich
        self.ids_interface = c
        self.ids_Interior = cInterior
        self.ids_global = np.array(self.ids_global)

        if self.save_tr_data or self.use_map:
            map_used = np.zeros((2*len(self.ids_global), len(self.ids_interface)))
            map_used[self.ids_interface, :] = np.eye(len(self.ids_interface), len(self.ids_interface))
            map_used = map_used.T
            np.save("./coSimData/map_used.npy", map_used)
        if self.use_map:
            self.map_used = map_used

        # Initial coordinates Vector saved here
        self.x0_vec = np.array(KM.VariableUtils().GetInitialPositionsVector(self.ModelPart.Nodes,2))

    def FinalizeSolutionStep(self,):
        if self.use_map and self.is_in_prediction_mode():
            self.rom_model.store_last_result()
            self.saved_out_t.append(self._analysis_stage.time)
        else:
            super().FinalizeSolutionStep()

        #residuals, _ = self._ComputeResiduals()

    def _GetNewSimulationName(self, ):
        return "::["+colors.yellow("Structural ROM")+"]:: "

    def InitializeSolutionStep(self,):
        if self.is_in_prediction_mode():
            KM.Logger.PrintInfo(self._GetNewSimulationName(), "STEP: ", self.ModelPart.ProcessInfo[KM.STEP])
            KM.Logger.PrintInfo(self._GetNewSimulationName(), "TIME: ", self._analysis_stage.time)
        if self.use_map and self.is_in_prediction_mode():
            pass
        else:
            super().InitializeSolutionStep()

    def Finalize(self):
        if self.use_map and self.rom_model is not None:
            self.u = self.rom_model.return_big_disps()
            self.export_results()

        super().Finalize()

    def OutputSolutionStep(self):
        if not (self.is_in_prediction_mode() and (self.use_map or self.interface_only)):
            super().OutputSolutionStep()


    def export_results(self):
        cs_tools.cs_print_info("Exporting fields on the complete physical domain")

        x0_vec = KM.VariableUtils().GetInitialPositionsVector(self.ModelPart.Nodes,2)
        t0 = time.time()
        for i in range(self.u.shape[1]):

            self.ModelPart.ProcessInfo[KM.STEP] = i + self.last_fom_step
            KM.VariableUtils().SetSolutionStepValuesVector(self.ModelPart.Nodes,
                                                KM.DISPLACEMENT, 1.*self.u[:, i], 0)
            self.ModelPart.GetCommunicator().SynchronizeVariable(KM.DISPLACEMENT)

            x_vec = x0_vec + 1.*self.u[:, i]
            KM.VariableUtils().SetCurrentPositionsVector(self.ModelPart.Nodes,1.*x_vec)
            super().OutputSolutionStep()
        t1 = time.time()
        self.recons_time.append(t1 - t0)
        with open("./coSimData/structure_recons_time.npy", 'wb') as f:
            np.save(f, np.array(self.recons_time))

    def ReceiveRomComponents(self, inputReduc_model=None, regression_model=None, outputReduc_model=None):
        self.inputReduc_model = inputReduc_model
        self.regression_model = regression_model
        self.outputReduc_model = outputReduc_model

    @classmethod
    def _GetDefaultParameters(cls):
        return KM.Parameters("""{
            "type"                    : "",
            "solver_wrapper_settings" : {},
            "io_settings"             : {},
            "data"                    : {},
            "mpi_settings"            : {},
            "echo_level"              : 0,
            "launch_time"             : 100.0,
            "force_norm_regr"         : false,
            "disp_norm_regr"          : false,
            "force_norm"              : "l2",
            "disp_norm"               : "l2",
            "start_collecting_time"   : 0.0,
            "imported_model"          : false,
            "save_model"              : false,
            "input_data"              : {},
            "output_data"             : {},
            "interface_only"          : false,
            "use_map"                 : true,
            "file"                    : {},
            "save_training_data"      : false
        }""")

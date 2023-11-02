# Importing the Kratos Library
import KratosMultiphysics as KM
import KratosMultiphysics.CoSimulationApplication as KratosCoSim

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_coupled_solver import CoSimulationCoupledSolver

# CoSimulation imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import KratosMultiphysics.CoSimulationApplication.factories.helpers as factories_helper
import KratosMultiphysics.CoSimulationApplication.colors as colors
from KratosMultiphysics.CoSimulationApplication.coupling_interface_data import CouplingInterfaceData
import numpy as np
import os
import time
import pickle

def Create(settings, models, solver_name):
    return GaussSeidelStrongCoupledSolver(settings, models, solver_name)

class GaussSeidelStrongCoupledSolver(CoSimulationCoupledSolver):
    def __init__(self, settings, models, solver_name):
        super().__init__(settings, models, solver_name)

        self.num_coupling_iterations = self.settings["num_coupling_iterations"].GetInt()
        self.secondary_interface = None

        # =========== Saving data ===========
        os.makedirs(os.path.dirname("./coSimData/"), exist_ok=True)
        self.iterations_table = []
        self.x_last = []
        self.input_fl_load = []
        self.solvers_times = {keys["name"].GetString():[] for keys in self.settings["coupling_sequence"].values()}
        self.save_tr_data = self.settings["save_tr_data"].GetBool()
        if self.save_tr_data:
            self.launch_train = self.settings["training_launch_time"].GetDouble()
            self.end_train = self.settings["training_end_time"].GetDouble()

        # =========== Initial Guess/Surrogate Settings ===========
        self.previous_surrogate_sol = None
        self.accel_solver =  self.settings["convergence_accelerators"][0]["solver"].GetString()
        self.accel_data =  self.settings["convergence_accelerators"][0]["data_name"].GetString()
        self.do_initial_guess = self.settings["initial_guess"].GetBool()
        self.initial_guess_Tstart = self.settings["initial_guess_launch_time"].GetDouble()
        self.fl_rom_model = None
        self.sol_rom_initialG = None

        # =========== Second interface Settings ===========
        self.launch_predict = self.settings["prediction_launch_time"].GetDouble()
        self.end_predict = self.settings["prediction_end_time"].GetDouble()


    def is_in_training_mode(self):
        t = self.process_info[KM.TIME]
        return (t >= self.launch_train) and (t <= self.end_train)
    
    def is_in_prediction_mode(self):
        t = self.process_info[KM.TIME]
        return (t >= self.launch_predict) and (t <= self.end_predict)
    
    def Initialize(self):
        super().Initialize()

        self.convergence_accelerators_list = factories_helper.CreateConvergenceAccelerators(
            self.settings["convergence_accelerators"],
            self.solver_wrappers,
            self.data_communicator,
            self.echo_level)

        self.convergence_criteria_list = factories_helper.CreateConvergenceCriteria(
            self.settings["convergence_criteria"],
            self.solver_wrappers,
            self.data_communicator,
            self.echo_level)

        for conv_acc in self.convergence_accelerators_list:
            conv_acc.Initialize()

        for conv_crit in self.convergence_criteria_list:
            conv_crit.Initialize()

    def Finalize(self):
        super().Finalize()

        for conv_acc in self.convergence_accelerators_list:
            conv_acc.Finalize()

        for conv_crit in self.convergence_criteria_list:
            conv_crit.Finalize()

    def _Get_fl_model(self, ):
        if self.fl_rom_model is None:
            with open('rom_model_saved/saved_fl_rom_E10_refined_Re_17.pkl', 'rb') as inp:
                self.fl_rom_model = pickle.load(inp)
        else:
            return self.fl_rom_model

    def _Get_sol_model(self, ):
        if self.sol_rom_initialG is None:
            with open('rom_model_saved/saved_sol_rom_E10_initial_g.pkl', 'rb') as inp:
                self.sol_rom_initialG = pickle.load(inp)
        else:
            return self.sol_rom_initialG
        
    def InitialGuess(self, previous_load, initial_disp):

        fl_rom_model = self._Get_fl_model()
        sol_rom_initialG = self._Get_sol_model()
        resid_norm = 100.
        w = 0.08
        past_iter_load = previous_load.copy()
        updated_disp = sol_rom_initialG.pred(previous_load.reshape((-1, 1)))
        inferred_load = fl_rom_model.predict(updated_disp, 
                                                    previous_load.reshape((-1, 1)))
        new_load = inferred_load.copy()
        r = new_load - previous_load.reshape((-1, 1))
        past_iter_r = r.copy()
        past_iter_load = new_load.copy()
        k = 0
        R = []
        X = []
        inv_J = None

        while resid_norm > 1e-5:
            if k>30:
                return None
            if k == 0:
                new_load = w * inferred_load + (1 - w) * past_iter_load.reshape((-1, 1))
            else:
                inv_J = np.array(X)[:, :, 0].T @ np.linalg.pinv(np.array(R)[:, :, 0].T)
                new_load = past_iter_newload - inv_J @ r + r

            updated_disp = sol_rom_initialG.pred(new_load)
            inferred_load = fl_rom_model.predict(updated_disp, 
                                                        previous_load.reshape((-1, 1)))
            r = inferred_load - new_load
            resid_norm = np.linalg.norm(r)/np.linalg.norm(new_load)

            R.append(r - past_iter_r)
            X.append(inferred_load - past_iter_load)

            past_iter_load = inferred_load.copy()
            past_iter_newload = new_load.copy()
            past_iter_r = r.copy()
            print("\n residual is ", resid_norm)
            k+=1
        self.solver_wrappers[self.accel_solver].GetInterfaceData(self.accel_data).SetData(previous_load + (inferred_load.ravel() - self.previous_surrogate_sol))
        self.previous_surrogate_sol = inferred_load.ravel()
        return inv_J
    
    def InitializeSolutionStep(self):
        super().InitializeSolutionStep()

        for conv_acc in self.convergence_accelerators_list:
            conv_acc.InitializeSolutionStep()

        for conv_crit in self.convergence_criteria_list:
            conv_crit.InitializeSolutionStep()

        self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] = 0
        if self.do_initial_guess:
            # Initial Surrogate SOL
            if self.previous_surrogate_sol is None:
                self.previous_surrogate_sol = self.solver_wrappers[self.accel_solver].GetInterfaceData(self.accel_data).GetData()
            
            if self.is_in_IGuess_prediction_mode():
                print("\n Performing an Initial Guess \n")
                inv_J = self.InitialGuess(self.solver_wrappers[self.accel_solver].GetInterfaceData(self.accel_data).GetData(), 
                                self.solver_wrappers["structure"].GetInterfaceData("disp").GetData())
                if inv_J is not None:
                    for conv_acc in self.convergence_accelerators_list:
                        conv_acc.ReceiveJacobian(inv_J)

    def is_in_IGuess_prediction_mode(self, ):
        return self.process_info[KM.TIME] >= self.initial_guess_Tstart

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()

        for conv_acc in self.convergence_accelerators_list:
            conv_acc.FinalizeSolutionStep()

        for conv_crit in self.convergence_criteria_list:
            conv_crit.FinalizeSolutionStep()


    def SolveSolutionStep(self):
        for k in range(self.num_coupling_iterations):
            self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] += 1

            if self.echo_level > 0:
                cs_tools.cs_print_info(self._ClassName(), colors.cyan("Coupling iteration:"), colors.bold(str(k+1)+" / " + str(self.num_coupling_iterations)))

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.InitializeCouplingIteration()

            for conv_acc in self.convergence_accelerators_list:
                conv_acc.InitializeNonLinearIteration()

            for conv_crit in self.convergence_criteria_list:
                conv_crit.InitializeNonLinearIteration()

            self._SaveInputflLoad()
            for solver_name, solver in self.solver_wrappers.items():
                self._SynchronizeInputData(solver_name)
                t0 = time.time()
                solver.SolveSolutionStep()
                t1 = time.time()
                self._SynchronizeOutputData(solver_name)
                self._SaveTimes(solver_name, t1-t0)

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.FinalizeCouplingIteration()

            for conv_acc in self.convergence_accelerators_list:
                conv_acc.FinalizeNonLinearIteration(self.process_info[KM.TIME])
                # current_CharDisp = self.solver_wrappers["structure"].GetInterfaceData("disp").GetData()[0]
                # conv_acc.FinalizeNonLinearIteration(self.process_info[KM.TIME], current_CharDisp)

            for conv_crit in self.convergence_criteria_list:
                conv_crit.FinalizeNonLinearIteration()

            is_converged = all([conv_crit.IsConverged() for conv_crit in self.convergence_criteria_list])

            if is_converged:
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.green("### CONVERGENCE WAS ACHIEVED ###"))
                self.__CommunicateIfTimeStepNeedsToBeRepeated(False)
                self._SaveNumIteration()
                if self.save_tr_data and self.is_in_training_mode():
                    self._SaveLastx(self.solver_wrappers[self.accel_solver].GetInterfaceData(self.accel_data).GetData())
                return True

            if k+1 >= self.num_coupling_iterations:
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.red("XXX CONVERGENCE WAS NOT ACHIEVED XXX"))
                self.__CommunicateIfTimeStepNeedsToBeRepeated(False)
                self._SaveNumIteration()
                if self.save_tr_data and self.is_in_training_mode():
                    self._SaveLastx(self.solver_wrappers[self.accel_solver].GetInterfaceData(self.accel_data).GetData())
                return False

            # if it reaches here it means that the coupling has not converged and this was not the last coupling iteration
            self.__CommunicateIfTimeStepNeedsToBeRepeated(True)

            # do relaxation only if this iteration is not the last iteration of this timestep
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.ComputeAndApplyUpdate()

    def _SaveInputflLoad(self):
        self.input_fl_load.append(self.solver_wrappers[self.accel_solver].GetInterfaceData(self.accel_data).GetData())
        with open("./coSimData/InputFlLoad.npy", 'wb') as f:
            np.save(f, np.array(self.input_fl_load).T) 

    def _SaveLastx(self, x_last):
        self.x_last.append(x_last)
        with open("./coSimData/finalX.npy", 'wb') as f:
            np.save(f, np.array(self.x_last).T)        

    def _SaveTimes(self, solver_name, t):
        self.solvers_times[solver_name].append(t)
        with open("./coSimData/"+solver_name+"_time.npy", 'wb') as f:
            np.save(f, np.array(self.solvers_times[solver_name]))

    def _SaveNumIteration(self, ):
        self.iterations_table.append(self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER])
        with open("./coSimData/iters.npy", 'wb') as f:
            np.save(f, np.array(self.iterations_table))
    
    def Check(self):
        super().Check()

        if len(self.convergence_criteria_list) == 0:
            raise Exception("At least one convergence criteria has to be specified")

        # TODO check if an accelerator was specified for a field that is manipulated in the input!

        for conv_crit in self.convergence_criteria_list:
            conv_crit.Check()

        for conv_crit in self.convergence_accelerators_list:
            conv_crit.Check()

    def _SynchronizeInputData(self, solver_name):
        super()._SynchronizeInputData(solver_name)

        self.rom_data = self.settings["rom_comm_data"]
        if self.rom_data.Has("coords_data") and (self.is_in_training_mode() or self.is_in_prediction_mode()):
            from_solver_name = self.rom_data["from_solver"].GetString()
            to_solver_name = self.rom_data["to_solver"].GetString()
            from_solver = self.solver_wrappers[from_solver_name]
            to_solver = self.solver_wrappers[to_solver_name]

            if self.secondary_interface is None:
                for (_, data_config) in self.rom_data["data"].items():
                    from_solver.extract_nodes(self.rom_data["coords_data"], data_config["model_part_name_local"].GetString())
                self.secondary_interface = {data_name : CouplingInterfaceData(data_config, from_solver.model, data_name, from_solver_name) for (data_name, data_config) in self.rom_data["data"].items()}

            input_data = np.array([])
            for (data_name, _)in self.rom_data["data"].items():
                input_data = np.concatenate((input_data, self.secondary_interface[data_name].GetData()))           
            to_solver.receive_input_data(input_data)


    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "convergence_accelerators"   : [],
            "convergence_criteria"       : [],
            "num_coupling_iterations"    : 10,
            "save_tr_data"               : false,
            "initial_guess"              : false,
            "initial_guess_launch_time"  : 0.1,
            "training_launch_time"       : 0.0,
            "training_end_time"          : 0.0,
            "prediction_launch_time"     : 0.0,
            "prediction_end_time"        : 0.0,
            "rom_comm_data"              : {}
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())

        return this_defaults

    def __CommunicateIfTimeStepNeedsToBeRepeated(self, repeat_time_step):
        # Communicate if the time step needs to be repeated with external solvers through IO
        export_config = {
            "type" : "repeat_time_step",
            "repeat_time_step" : repeat_time_step
        }

        for solver in self.solver_wrappers.values():
            solver.ExportData(export_config)


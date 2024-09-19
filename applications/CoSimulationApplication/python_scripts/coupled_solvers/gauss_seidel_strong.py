# Importing the Kratos Library
import KratosMultiphysics as KM
import KratosMultiphysics.CoSimulationApplication as KratosCoSim

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_coupled_solver import CoSimulationCoupledSolver

# CoSimulation imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import KratosMultiphysics.CoSimulationApplication.factories.helpers as factories_helper
import KratosMultiphysics.CoSimulationApplication.colors as colors
import numpy as np
import os
import time
from collections import deque

def Create(settings, models, solver_name):
    return GaussSeidelStrongCoupledSolver(settings, models, solver_name)

class GaussSeidelStrongCoupledSolver(CoSimulationCoupledSolver):
    def __init__(self, settings, models, solver_name):
        super().__init__(settings, models, solver_name)
        self.load_data = deque()

        self.num_coupling_iterations = self.settings["num_coupling_iterations"].GetInt()

        # =========== Saving data ===========
        os.makedirs(os.path.dirname("./coSimData/"), exist_ok=True)
        self.iterations_table = []
        self.solvers_times = {keys["name"].GetString():[] for keys in self.settings["coupling_sequence"].values()}
        self.save_tr_data = self.settings["save_tr_data"].GetBool()
        if self.save_tr_data:
            self.launch_train = self.settings["training_launch_time"].GetDouble()
            self.end_train = self.settings["training_end_time"].GetDouble()

    def is_in_saving_mode(self):
        t = self.process_info[KM.TIME]
        return (t >= self.launch_train) and (t <= self.end_train)

    def Initialize(self):
        super().Initialize()

        self.accelerated_input_solver = list(self.solver_wrappers.keys())[0]
        self.raw_input_solver = list(self.solver_wrappers.keys())[-1]
        self.accelerated_data = self.coupling_sequence[self.accelerated_input_solver]["input_data_list"][0]["data"].GetString()
        self.raw_data = self.coupling_sequence[self.accelerated_input_solver]["output_data_list"][0]["data"].GetString()

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

    def InitializeSolutionStep(self):
        super().InitializeSolutionStep()

        for conv_acc in self.convergence_accelerators_list:
            conv_acc.InitializeSolutionStep()

        for conv_crit in self.convergence_criteria_list:
            conv_crit.InitializeSolutionStep()

        self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] = 0


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

            for solver_name, solver in self.solver_wrappers.items():
                t0 = time.time()
                self._SynchronizeInputData(solver_name)
                self._ReceiveRomData(solver_name)
                solver.SolveSolutionStep()
                t1 = time.time()
                self._SynchronizeOutputData(solver_name)
                self._SaveTimes(solver_name, t1-t0)

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.FinalizeCouplingIteration()

            for conv_acc in self.convergence_accelerators_list:
                conv_acc.FinalizeNonLinearIteration(self.process_info[KM.TIME])

            for conv_crit in self.convergence_criteria_list:
                conv_crit.FinalizeNonLinearIteration()

            is_converged = all([conv_crit.IsConverged() for conv_crit in self.convergence_criteria_list])

            if is_converged:
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.green("### CONVERGENCE WAS ACHIEVED ###"))
                self.__CommunicateIfTimeStepNeedsToBeRepeated(False)
                self._SaveNumIteration()
                return True

            if k+1 >= self.num_coupling_iterations:
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.red("XXX CONVERGENCE WAS NOT ACHIEVED XXX"))
                self.__CommunicateIfTimeStepNeedsToBeRepeated(False)
                self._SaveNumIteration()
                return False

            # if it reaches here it means that the coupling has not converged and this was not the last coupling iteration
            self.__CommunicateIfTimeStepNeedsToBeRepeated(True)

            # do relaxation only if this iteration is not the last iteration of this timestep
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.ComputeAndApplyUpdate()

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

    def _SynchronizeOutputData(self, solver_name):
        super()._SynchronizeOutputData(solver_name)
        if solver_name == self.raw_input_solver:
            for predictor in self.predictors_list:
                if predictor.receives_data:
                    predictor.ReceiveNewData(self.solver_wrappers[self.accelerated_input_solver].GetInterfaceData(self.raw_data).GetData().reshape((-1, 1)),
                                            self.solver_wrappers[self.raw_input_solver].GetInterfaceData(self.accelerated_data).GetData().reshape((-1, 1)))

        if self.save_tr_data and self.is_in_saving_mode():
            self.load_data.append(self.solver_wrappers[self.raw_input_solver].GetInterfaceData(self.accelerated_data).GetData().reshape((-1, 1)))
            np.save("./coSimData/out"+self.raw_input_solver+"_"+self.accelerated_data+"_data.npy",
                    np.asarray(self.load_data)[:, :, 0].T)

    def _SynchronizeInputData(self, solver_name):
        super()._SynchronizeInputData(solver_name)

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "convergence_accelerators"   : [],
            "convergence_criteria"       : [],
            "num_coupling_iterations"    : 10,
            "save_tr_data"               : false,
            "training_launch_time"       : 0.0,
            "training_end_time"          : 0.0
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


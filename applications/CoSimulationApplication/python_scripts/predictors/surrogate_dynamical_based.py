# Importing the base class
import KratosMultiphysics as KM
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_predictor import CoSimulationPredictor
import KratosMultiphysics.CoSimulationApplication.colors as colors

# Other imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import numpy as np
import pickle
from rom_am.solid_rom import *
from rom_am.fluid_surrogate import *
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from collections import deque


def Create(settings, solver_wrapper, solver_wrapperY):
    cs_tools.SettingsTypeCheck(settings)
    return SurrogateDynamicalPredictor(settings, solver_wrapper, solver_wrapperY)


class SurrogateDynamicalPredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper, solver_wrapperY):
        super().__init__(settings, solver_wrapper)
        self.receives_data = True
        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.launch_retrain = self.settings["retraining_launch_time"].GetDouble(
        )
        self.max_retrain = self.settings["retraining_max_time"].GetDouble()
        self.rel_tolerance = self.settings["rel_tolerance"].GetDouble()
        self.maxIter = self.settings["max_iters"].GetInt()
        self.w0 = self.settings["w0"].GetDouble()
        self.save_log = self.settings["save_log"].GetBool()
        self.jump_start = self.settings["jump_start"].GetBool()
        fluidSurrofFileName = self.settings["file_nameFluid"].GetString()
        solidROMFileName = self.settings["file_nameSolid"].GetString()
        self.commonDispReducer = self.settings["commonDispReducer"].GetBool()
        weights = self.settings["weights"].GetBool()
        criterion = self.settings["criterion"].GetString()
        if criterion == "relative":
            self.criterion = True
        else:
            self.criterion = False

        if self.commonDispReducer:
            KM.Logger.PrintWarning(
                "surrogateBased Predictor", "The setting `commonDispReducer` will be soon deprecated, a common displacement encoder will always be used")
        self.re_train_thres = self.settings["re_train_thres"].GetInt()
        self.updateThres = self.settings["update_thres"].GetInt()
        self.extrap_order = self.settings["extrapolation_order"].GetInt()
        self.param_0_value = self.settings["param_0_value"].GetDouble()
        self.param_1_value = 100*self.settings["param_1_value"].GetDouble()
        self.param_array =np.array([[self.param_0_value], [self.param_1_value]])
        self.stepsize = self.settings["stepsize"].GetDouble()
        if self.stepsize < 0:
            self.stepsize = None
        # self.fluidSurrogate = FluidSurrog()
        self.fluidSurrogate = TrackedFluidSurrog()
        with open(fluidSurrofFileName, 'rb') as inp:
            self.fluidSurrogate = pickle.load(inp)
        self.solidSurrogate = FluidSurrog(maxLen = 11000)
        with open(solidROMFileName, 'rb') as inp:
            self.solidSurrogate = pickle.load(inp)

        importedSolidReduc = PodReducer(.999999)
        # with open("savedROMs/dt25/solidSavedReducer.pkl", 'rb') as inp:
        # with open("savedROMs/solidSavedReducer.pkl", 'rb') as inp:
        with open("savedROMs/dt20/solidSavedReducer.pkl", 'rb') as inp:
            importedSolidReduc = pickle.load(inp)

        if self.re_train_thres > 0:
            self.fluidSurrogate.reTrainThres = self.re_train_thres
            self.solidSurrogate.reTrainThres = self.re_train_thres
        if self.updateThres > 0:
            self.fluidSurrogate.updateThres = self.updateThres
        self.solidSurrogate.weights = weights
        self.fluidSurrogate.weights = weights



        self._local_iter = None
        self._success = None
        self._local_resid = None
        if self.save_log:
            self.local_iters = []
            self.local_successes = []
            self.local_resid = []

        self.previousX = None
        self.previousY = None
        self.surrJac = None
        self.surrQ = None
        self.surrR = None
        self.deltaX = None
        self.X_tilde = deque(maxlen=50)
        self.R = deque(maxlen=50)
        self.secondPreviousX = None
        self.thirdPreviousX = None
        self.surrJac = None
        self.interface_dataYvel = solver_wrapperY.GetInterfaceData("velocity")
        self.interface_dataY = solver_wrapperY.GetInterfaceData("disp")
        self.takes_accelerated = False
        self.predictorTime = []

        self.fluidSurrogate.initialize_predictions(self.param_array)

        # The following training will only compute the regression operator,
        # the Encoder-Decoder is precomputed
        # self.solidSurrogate.train(np.load("./savedROMs/dt25/loadData_forSolid.npy"),
        #                         np.load("./savedROMs/dt25/dispConvData_forSolid.npy"),
        #                         np.load("./savedROMs/dt25/dispData_forSolid.npy"),
        # self.solidSurrogate.train(np.load("./savedROMs/loadData_forSolid.npy"),
        #                         np.load("./savedROMs/dispConvData_forSolid.npy"),
        #                         np.load("./savedROMs/dispData_forSolid.npy"),
        self.solidSurrogate.train(np.load("./savedROMs/dt20/loadData_forSolid.npy"),
                                np.load("./savedROMs/dt20/dispConvData_forSolid.npy"),
                                np.load("./savedROMs/dt20/dispData_forSolid.npy"),
                                rank_pres=.999999, rank_disp=20, smoothing=1e-6,
                                kernel="polyC", degree = 1, norm = [True, True],
                                center=[True, True], normalization=["max", "max"],
                                norm_regr = "max",
                                solidReduc=self.fluidSurrogate.reducLoad,
                                precomputedReducLoad = importedSolidReduc)


    def ReceiveNewData(self, newDisp, newLoad):
        if self.currentT >= self.launch_retrain and self.currentT <= self.max_retrain:
            if self.previousX is not None:
                prevX = self.previousX.reshape((-1, 1))
                if self.commonDispReducer:
                    dispReduc_model = self.solidSurrogate.reducLoad
                else:
                    dispReduc_model = None
                self.fluidSurrogate.augmentData(
                    newDisp, prevX, newLoad[2:-2, [0]], self.currentT,
                    params = self.param_array,
                    solidReduc = dispReduc_model,
                    stepsize=self.stepsize
                    )

    def ReceiveNewDataS(self, newLoad, newDisp):
        if self.currentT >= self.launch_retrain and self.currentT <= self.max_retrain:
            if self.previousY is not None:
                prevY = self.previousY.reshape((-1, 1))
                prevY = np.vstack((prevY, self.previousYvel.reshape((-1, 1))))
                if self.commonDispReducer:
                    loadReduc_model = self.fluidSurrogate.reducLoad
                else:
                    loadReduc_model = None
                self.solidSurrogate.augmentData(
                    newLoad[2:-2, [0]], prevY, newDisp, self.currentT, solidReduc=loadReduc_model, changeTheBasis=self.fluidSurrogate.sendSignalBasis)


    def Predict(self):
        self.X_tilde = deque(maxlen=50)
        self.R = deque(maxlen=50)

        if not self.interface_data.IsDefinedOnThisRank():
            return

        if self.currentT >= self.launch_time:
            w = self.w0
            current_data = self.interface_data.GetData(0)[2:-2]
            if self.extrap_order > 0:
                if self.secondPreviousX is not None:
                    previous_data = self.secondPreviousX.ravel()
                else:
                    previous_data = current_data.copy()
                previous_data_2 = 0.
                alpha1 = 2.
                alpha2 = -1.
                alpha3 = 0.

                if self.extrap_order > 1 and self.thirdPreviousX is not None:
                    previous_data_2 = self.thirdPreviousX.ravel()
                    alpha1 = 3.
                    alpha2 = -3.
                    alpha3 = 1.

                initial_data = alpha1*current_data + alpha2 * \
                    previous_data + alpha3 * previous_data_2
            else:
                initial_data = current_data.copy()

            pred_ = initial_data.copy()
            isConverged = False

            if self.previousX is not None:
                previousYsol = np.vstack((self.previousY[:, np.newaxis], self.previousYvel[:, np.newaxis]))
                i = 0
                while i < self.maxIter and not isConverged:
                    if self.echo_level > 0:
                        cs_tools.cs_print_info(self._ClassName(), colors.darkcyan(
                            "Predictor fixed-point iteration:"), colors.bold(str(i)+" / " + str(self.maxIter)))
                    self._local_iter = i
                    if self.commonDispReducer:
                        loadReduc_model = self.fluidSurrogate.reducLoad
                    else:
                        loadReduc_model = None
                    solidSol = self.solidSurrogate.predict(pred_.reshape(
                        (-1, 1)), previousYsol, solidReduc=loadReduc_model, predict_low_dimensional=True)
                    if self.commonDispReducer:
                        dispReduc_model = self.solidSurrogate.reducLoad
                    else:
                        dispReduc_model = None
                    fluidSol = self.fluidSurrogate.predict(
                        solidSol, self.previousX[:, np.newaxis], solidReduc=dispReduc_model,
                        params = self.param_array, takes_low_dimensional_disp=True
                        ).ravel()
                    self.X_tilde.appendleft(fluidSol)
                    newResiduals = fluidSol - pred_
                    self.R.appendleft(newResiduals)
                    # The next two norms are squared ! but that's okay, since they are always divided by each other
                    nrm = np.dot(newResiduals, newResiduals)
                    pred_norm = np.dot(pred_, pred_)
                    if self.echo_level > 0:
                        cs_tools.cs_print_info(
                            self._ClassName(), "Residual: ", str(np.sqrt(nrm/pred_norm)))
                    if (nrm > 9*pred_norm) and i > 1:
                        self._success = 0
                        self._local_resid = np.sqrt(nrm/pred_norm)
                        if self.echo_level > 0:
                            cs_tools.cs_print_info(
                                self._ClassName(), colors.darkred("X CONVERGENCE FAILED X"))
                        return

                    if self.criterion:
                        if (nrm/pred_norm) < (self.rel_tolerance**2):
                            isConverged = True
                            self._local_resid = np.sqrt(nrm/pred_norm)
                    else:
                        if (nrm/newResiduals.shape[0]) < (self.rel_tolerance**2):
                            isConverged = True
                            self._local_resid = np.sqrt(nrm/pred_norm)

                    if not isConverged:
                        if i > 1:
                            diffResiduals = newResiduals-prevResidual
                            diff_norm_sq = np.dot(diffResiduals, diffResiduals)
                            if diff_norm_sq > 1e-16:  # Avoid division by zero
                                w = - w * np.dot(prevResidual, diffResiduals)/diff_norm_sq
                            if w < 0 and not isConverged:
                                w = self.w0
                        prevResidual = newResiduals.copy()
                        pred_ = w * fluidSol + (1-w) * pred_
                    i += 1

            if isConverged:
                self._success = 1
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.darkgreen(
                        "# CONVERGENCE WAS ACHIEVED #"))
                self._UpdateData(np.concatenate(
                    (np.array([0, 0]), pred_, np.array([0, 0])))) # corners
            else:
                self._success = 0
                if self.echo_level > 0:
                    cs_tools.cs_print_info(
                        self._ClassName(), colors.darkred("X CONVERGENCE FAILED X"))
                return

    def qr_filter(self, Q, R, V, W):

        epsilon = 3e-4
        cols = V.shape[1]
        i = 0
        while i < cols:
            if np.abs(np.diag(R)[i]) < epsilon:
                ids_tokeep = np.delete(np.arange(0, cols), i)
                V = V[:, ids_tokeep]
                cols = V.shape[1]
                W = W[:, ids_tokeep]
                Q, R = np.linalg.qr(V)
            else:
                i += 1

        return Q, R, V, W

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        self.surrJac = None
        self.surrQ = None
        if self.previousX is not None:
            if self.extrap_order > 1:
                if self.secondPreviousX is not None:
                    self.thirdPreviousX = self.secondPreviousX.copy()
            self.secondPreviousX = self.previousX.copy()
        self.previousX = self.interface_data.GetData().copy()[2:-2]
        self.previousY = self.interface_dataY.GetData().copy()
        self.previousYvel = self.interface_dataYvel.GetData().copy()
        if self.save_log:
            self.local_iters.append(self._local_iter)
            self.local_resid.append(self._local_resid)
            self.local_successes.append(self._success)

    def Finalize(self):
        super().Finalize()
        if self.save_log:
            np.save("./coSimData/local_iters.npy", np.array(self.local_iters))
            np.save("./coSimData/local_resid.npy", np.array(self.local_resid))
            np.save("./coSimData/local_successes.npy",
                    np.array(self.local_successes))
            np.save("./coSimData/number_of_retrainings.npy",
                    np.array(self.fluidSurrogate.retrain_count))
            np.save("./coSimData/moments_of_retrainings.npy",
                    np.array(self.fluidSurrogate.retrain_times))

        self.fluidSurrogate.save("./coSimData/lastStateFluid")
        self.solidSurrogate.save("./coSimData/lastStateSolid")

    def ReceiveTime(self, t):
        self.currentT = t

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "prediction_launch_time" : 100,
            "max_iters"              : 20,
            "rel_tolerance"          : 1e-2,
            "w0"                     : 0.04,
            "retraining_launch_time" : 100,
            "retraining_max_time" : 100,
            "file_nameFluid"              : "",
            "file_nameSolid"              : "",
            "criterion"                   : "relative",
            "commonDispReducer"           : true,
            "save_log"                    : true,
            "jump_start"                  : true,
            "extrapolation_order"         : 1,
            "re_train_thres"              : -1,
            "update_thres"                : -1,
            "param_0_value"               : 1.0,
            "param_1_value"               : 1.0,
            "stepsize"                    : -1,
            "weights"                     : false
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults


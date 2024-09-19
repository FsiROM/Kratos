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
from rom_am.fluidRecusrsiveROM import *
# ... See (1) ...
# from collections import deque


def Create(settings, solver_wrapper):
    cs_tools.SettingsTypeCheck(settings)
    return SurrogatePredictor(settings, solver_wrapper)


class SurrogatePredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper):
        super().__init__(settings, solver_wrapper)
        self.receives_data = True
        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.launch_retrain = self.settings["retraining_launch_time"].GetDouble(
        )
        self.rel_tolerance = self.settings["rel_tolerance"].GetDouble()
        self.maxIter = self.settings["max_iters"].GetInt()
        self.w0 = self.settings["w0"].GetDouble()
        self.save_log = self.settings["save_log"].GetBool()
        self.jump_start = self.settings["jump_start"].GetBool()
        fluidSurrofFileName = self.settings["file_nameFluid"].GetString()
        solidROMFileName = self.settings["file_nameSolid"].GetString()
        self.commonDispReducer = self.settings["commonDispReducer"].GetBool()
        if self.commonDispReducer:
            KM.Logger.PrintWarning(
                "surrogateBased Predictor", "The setting `commonDispReducer` will be soon deprecated, a common displacement encoder will always be used")
        self.re_train_thres = self.settings["re_train_thres"].GetInt()
        self.extrap_order = self.settings["extrapolation_order"].GetInt()
        # self.fluidSurrogate = FluidSurrog()
        with open(fluidSurrofFileName, 'rb') as inp:
            self.fluidSurrogate = pickle.load(inp)
        if self.re_train_thres > 0:
            self.fluidSurrogate.reTrainThres = self.re_train_thres
        self.solidSurrogate = solid_ROM()
        with open(solidROMFileName, 'rb') as inp:
            self.solidSurrogate = pickle.load(inp)

        #TODO
        #In next version
        #   if self.commonDispReducer:
        #     assert self.fluidSurrogate.disp_latent_dim == self.solidSurrogate.dispReduc_model.latent_dim, "The user indicates a common displacement Reducer, but the internal latent dimensions differ."

        self._local_iter = None
        self._success = None
        self._local_resid = None
        if self.save_log:
            self.local_iters = []
            self.local_successes = []
            self.local_resid = []

        self.previousX = None
        self.surrJac = None
        self.surrQ = None
        self.surrR = None
        self.deltaX = None
        self.secondPreviousX = None
        self.thirdPreviousX = None
        self.surrJac = None

    def ReceiveNewData(self, newDisp, newLoad):
        if self.currentT >= self.launch_time and self.currentT >= self.launch_retrain:
            if self.previousX is not None:
                prevX = self.previousX.reshape((-1, 1))
                if self.commonDispReducer:
                    dispReduc_model = self.solidSurrogate.dispReduc_model
                else:
                    dispReduc_model = None
                self.fluidSurrogate.augmentData(
                    newDisp, prevX, newLoad, self.currentT, dispReduc_model)

    def Predict(self):
        if not self.interface_data.IsDefinedOnThisRank():
            return

        if self.currentT >= self.launch_time:
            w = self.w0
            current_data = self.interface_data.GetData(0)
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
            # ... See (1) ...
            # R = deque( maxlen = 20 )
            # X = deque( maxlen = 20 )
            if self.previousX is not None:
                i = 0
                while i < self.maxIter and not isConverged:
                    if self.echo_level > 0:
                        cs_tools.cs_print_info(self._ClassName(), colors.darkcyan("Predictor fixed-point iteration:"), colors.bold(str(i)+" / " + str(self.maxIter)))
                    self._local_iter = i
                    solidSol = self.solidSurrogate.pred(pred_.reshape((-1, 1)))
                    if self.commonDispReducer:
                        dispReduc_model = self.solidSurrogate.dispReduc_model
                    else:
                        dispReduc_model = None
                    fluidSol = self.fluidSurrogate.predict(
                        solidSol, self.previousX[:, np.newaxis], solidReduc=dispReduc_model).ravel()
                    newResiduals = fluidSol - pred_
                    nrm = np.linalg.norm(newResiduals)
                    pred_norm = np.linalg.norm(pred_)
                    if self.echo_level > 0:
                        cs_tools.cs_print_info(self._ClassName(), "Residual: ", str(nrm/pred_norm))
                    if (nrm > 3*pred_norm) and i > 1:
                        self._success = 0
                        self._local_resid = nrm/pred_norm
                        if self.echo_level > 0:
                            cs_tools.cs_print_info(self._ClassName(), colors.darkred("X CONVERGENCE FAILED X"))
                        return

                    # ... (1) Future Work for enhancement of the inverse Jacobian ...
                    # R.appendleft(newResiduals.ravel().copy())
                    # X.appendleft(pred_.ravel().copy())
                    # if len(R) > 1:
                    #     deltaX = np.empty((len(R)-1, len(pred_)))
                    #     deltaR = np.empty((len(R)-1, len(pred_)))
                    #     for j in range(len(R)-1):
                    #         deltaX[j] = X[j] - X[j+1]
                    #         deltaR[j] = R[j] - R[j+1]
                    #     deltaX = deltaX.T
                    #     deltaR = deltaR.T

                    if (nrm/pred_norm) < self.rel_tolerance:
                        isConverged = True
                        self._local_resid = nrm/pred_norm
                        # ... See (1) ...
                        #  self.surrJac = deltaX @ np.linalg.pinv(deltaR, rcond=1e-8)

                    if not isConverged:
                        if i > 1:
                            w = - w * (np.dot(prevResidual, (newResiduals-prevResidual))
                                       )/(np.linalg.norm(newResiduals-prevResidual)**2)
                            if w < 0 and not isConverged:
                                w = self.w0
                        prevResidual = newResiduals.copy()
                        pred_ = w * fluidSol.ravel() + (1-w) * pred_.ravel()
                    i += 1

            if isConverged:
                self._success = 1
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.darkgreen("# CONVERGENCE WAS ACHIEVED #"))
                self._UpdateData(fluidSol)
            else:
                self._success = 0
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.darkred("X CONVERGENCE FAILED X"))
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
        self.previousX = self.interface_data.GetData().copy()
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
            "file_nameFluid"              : "",
            "file_nameSolid"              : "",
            "commonDispReducer"           : true,
            "save_log"                    : true,
            "jump_start"                  : true,
            "extrapolation_order"         : 1,
            "re_train_thres"              : -1
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

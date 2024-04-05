# Importing the base class
import KratosMultiphysics as KM
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_predictor import CoSimulationPredictor

# Other imports
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import numpy as np
import pickle
from ..convergence_accelerators.mySurrogates import *
from rom_am.solid_rom import *
from collections import deque

def Create(settings, solver_wrapper, solver_wrapperY):
    cs_tools.SettingsTypeCheck(settings)
    return SurrogatePredictor(settings, solver_wrapper, solver_wrapperY)

class SurrogatePredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper, solver_wrapperY):
        super().__init__(settings, solver_wrapper)

        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.launch_retrain = self.settings["retraining_launch_time"].GetDouble()
        self.rel_tolerance = self.settings["rel_tolerance"].GetDouble()
        self.maxIter = self.settings["max_iters"].GetInt()
        self.w0 = self.settings["w0"].GetDouble()
        self.save_log = self.settings["save_log"].GetBool()
        self.jump_start = self.settings["jump_start"].GetBool()
        fluidSurrofFileName = self.settings["file_nameFluid"].GetString()
        solidROMFileName = self.settings["file_nameSolid"].GetString()
        self.re_train_thres = self.settings["re_train_thres"].GetInt()
        self.extrap_order = self.settings["extrapolation_order"].GetInt()
        #self.fluidSurrogate = FluidSurrog()
        with open(fluidSurrofFileName, 'rb') as inp:
            self.fluidSurrogate = pickle.load(inp)
        if self.re_train_thres > 0:
            self.fluidSurrogate.reTrainThres = self.re_train_thres
        self.solidSurrogate = solid_ROM()
        with open(solidROMFileName, 'rb') as inp:
            self.solidSurrogate = pickle.load(inp)
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
        if self.currentT >= self.launch_time and self.currentT >= self.launch_retrain :
            if self.previousX is not None:
                prevX = self.previousX.reshape((-1, 1))
                self.fluidSurrogate.augmentData(newDisp, prevX, newLoad, self.currentT)

    def Predict(self):
        if not self.interface_data.IsDefinedOnThisRank(): return

        if self.currentT >= self.launch_time:
            w = self.w0
            current_data  = self.interface_data.GetData(0)
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


                initial_data = alpha1*current_data + alpha2 * previous_data + alpha3 * previous_data_2
            else:
                initial_data = current_data.copy()

            pred_ = initial_data.copy()
            isConverged = False
            # R = deque( maxlen = 20 )
            # X = deque( maxlen = 20 )
            if self.previousX is not None:
                i = 0
                while i<self.maxIter and not isConverged:
                    print("iteration ", i)
                    self._local_iter = i
                    solidSol = self.solidSurrogate.pred(pred_.reshape((-1, 1)))
                    fluidSol = self.fluidSurrogate.predict(solidSol, self.previousX[:, np.newaxis]).ravel()
                    newResiduals = fluidSol - pred_
                    nrm = np.linalg.norm(newResiduals)
                    pred_norm = np.linalg.norm(pred_)
                    print(nrm/pred_norm)
                    if (nrm > 3*pred_norm) and i > 1:
                        self._success = 0
                        self._local_resid = nrm/pred_norm
                        return

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
                        # self.surrJac = deltaX @ np.linalg.pinv(deltaR, rcond=1e-8)

                    if not isConverged:
                        if i > 1:
                            w = - w * (np.dot(prevResidual, (newResiduals-prevResidual)))/(np.linalg.norm(newResiduals-prevResidual)**2)
                            if w < 0 and not isConverged:
                                w = self.w0
                        prevResidual = newResiduals.copy()
                        pred_ = w * fluidSol.ravel() + (1-w) * pred_.ravel()
                    i+=1

            if isConverged:
                self._success = 1
                self._UpdateData(fluidSol)
            else:
                self._success = 0
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
            np.save("./coSimData/local_successes.npy", np.array(self.local_successes))
            np.save("./coSimData/number_of_retrainings.npy", np.array(self.fluidSurrogate.retrain_count))
            np.save("./coSimData/moments_of_retrainings.npy", np.array(self.fluidSurrogate.retrain_times))

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
            "save_log"                    : true,
            "jump_start"                  : true,
            "extrapolation_order"         : 1,
            "re_train_thres"              : -1
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

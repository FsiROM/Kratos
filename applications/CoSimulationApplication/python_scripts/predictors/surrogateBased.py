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

def Create(settings, solver_wrapper):
    cs_tools.SettingsTypeCheck(settings)
    return SurrogatePredictor(settings, solver_wrapper)

class SurrogatePredictor(CoSimulationPredictor):
    def __init__(self, settings, solver_wrapper):
        super().__init__(settings, solver_wrapper)

        self.launch_time = self.settings["prediction_launch_time"].GetDouble()
        self.launch_retrain = self.settings["retraining_launch_time"].GetDouble()
        self.rel_tolerance = self.settings["rel_tolerance"].GetDouble()
        self.maxIter = self.settings["max_iters"].GetInt()
        self.w0 = self.settings["w0"].GetDouble()
        fluidSurrofFileName = self.settings["file_nameFluid"].GetString()
        solidROMFileName = self.settings["file_nameSolid"].GetString()
        self.map = np.load("./coSimData/MeshData/map_used.npy")
        self.fluidSurrogate = FluidSurrog()
        with open(fluidSurrofFileName, 'rb') as inp:
            self.fluidSurrogate = pickle.load(inp)
        self.solidSurrogate = solid_ROM()
        # with open("./ROMs/Double_ROM_models/accelTestGlobal.pkl", 'rb') as inp:
        with open(solidROMFileName, 'rb') as inp:
            self.solidSurrogate = pickle.load(inp)

        self.previousX = None
        self.surrJac = None
        self.surrQ = None
        self.surrR = None
        self.deltaX = None

    def ReceiveNewData(self, newDisp, newLoad):
        if self.currentT >= self.launch_time and self.currentT >= self.launch_retrain :
            if self.previousX is not None:
                prevX = self.previousX.reshape((-1, 1))
                self.fluidSurrogate.augmentData(newDisp, prevX, newLoad)

    def Predict(self):
        if not self.interface_data.IsDefinedOnThisRank(): return

        if self.currentT >= self.launch_time:
            w = self.w0
            current_data  = self.interface_data.GetData(0)
            initial_data = 2*current_data - self.interface_data.GetData(1)

            pred_ = initial_data.copy()
            isConverged = False
            R = deque( maxlen = 20 )
            X = deque( maxlen = 20 )
            # R = []
            # X = []
            if self.previousX is not None:
                i = 0
                while i<self.maxIter and not isConverged:
                    print("iteration ", i)
                    solidSol = self.map @ self.solidSurrogate.pred(pred_.reshape((-1, 1)))
                    #solidSol = self.solidSurrogate.pred(pred_.reshape((-1, 1)))
                    fluidSol = self.fluidSurrogate.predict(solidSol, self.previousX[:, np.newaxis]).ravel()
                    newResiduals = fluidSol - pred_
                    nrm = np.linalg.norm(newResiduals)
                    print(nrm)
                    if (np.linalg.norm(newResiduals) > 3*np.linalg.norm(pred_)) and i > 1:
                        return
                    R.appendleft(newResiduals.ravel().copy())
                    X.appendleft(pred_.ravel().copy())

                    # if len(R) > 1:
                    #     deltaX = np.empty((len(R)-1, len(pred_)))
                    #     deltaR = np.empty((len(R)-1, len(pred_)))
                    #     for i in range(len(R)-1):
                    #         deltaX[i] = X[i] - X[i+1]
                    #         deltaR[i] = R[i] - R[i+1]
                    #     deltaX = deltaX.T
                    #     deltaR = deltaR.T

                    #     self.surrJac = deltaX @ np.linalg.pinv(deltaR)

                    if (np.linalg.norm(newResiduals)/np.linalg.norm(pred_)) < self.rel_tolerance:
                        isConverged = True

                    # if not isConverged:
                    #     if len(R) > 1:
                    #         pred_ = pred_ - self.surrJac @ newResiduals - (-newResiduals + deltaR @ np.linalg.pinv(deltaR) @ newResiduals)
                    #     else:
                    #         pred_ = w * fluidSol.ravel() + (1-w) * pred_.ravel()

                    if not isConverged:
                        if i > 1:
                            w = - w * (np.dot(prevResidual, (newResiduals-prevResidual)))/(np.linalg.norm(newResiduals-prevResidual)**2)
                            if w < 0 and not isConverged:
                                w = self.w0
                        prevResidual = newResiduals.copy()
                        pred_ = w * fluidSol.ravel() + (1-w) * pred_.ravel()
                    i+=1

            # if len(R)>1:
            #     self.surrQ , _ = np.linalg.qr(deltaR)

            if isConverged:
                self._UpdateData(fluidSol)
            else:
                return

            if len(R) > 2 and isConverged:
                deltaX = np.empty((len(R)-1, len(pred_)))
                deltaR = np.empty((len(R)-1, len(pred_)))
                for i in range(len(R)-1):
                    deltaX[i] = X[i] - X[i+1]
                    deltaR[i] = R[i] - R[i+1]
                deltaX = deltaX.T
                deltaR = deltaR.T

                self.surrJac = deltaX @ np.linalg.pinv(deltaR, rcond=1e-7)
                surrQ , surrR = np.linalg.qr(deltaR)

                surrQ, surrR, deltaR, deltaX = self.qr_filter(surrQ, surrR, deltaR, deltaX)
                self.surrR = surrR
                self.surrQ = surrQ
                self.deltaX = deltaX

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
        self.previousX = self.interface_data.GetData().copy()

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
            "file_nameSolid"              : ""
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

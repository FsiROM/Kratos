# @module iqnilsM
# This module contains the class IQNILS with a Surrogate acceleration
# Author: TIBA Azzeddine
# Date: Feb. 20, 2017

# Importing the Kratos Library
import KratosMultiphysics as KM

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.base_classes.co_simulation_convergence_accelerator import CoSimulationConvergenceAccelerator

# CoSimulation imports
from KratosMultiphysics.CoSimulationApplication.co_simulation_tools import cs_print_info, cs_print_warning, SettingsTypeCheck
import KratosMultiphysics.CoSimulationApplication.colors as colors

# Other imports
import numpy as np
import scipy as sp
from copy import deepcopy
from collections import deque
import pickle
from .mySurrogates import *
from rom_am.solid_rom import *


def Create(settings):
    SettingsTypeCheck(settings)
    return IQNILSSURR2ConvergenceAccelerator(settings)

# Class IQNILSConvergenceAccelerator.
# This class contains the implementation of the IQN-ILS method and helper functions.
# Reference: Joris Degroote, PhD thesis "Development of algorithms for the partitioned simulation of strongly coupled fluid-structure interaction problems", 84-91.


class IQNILSSURR2ConvergenceAccelerator(CoSimulationConvergenceAccelerator):
    # The constructor.
    # @param iteration_horizon Maximum number of vectors to be stored in each time step.
    # @param timestep_horizon Maximum number of time steps of which the vectors are used.
    # @param alpha Relaxation factor for computing the update, when no vectors available.
    def __init__(self, settings):
        super().__init__(settings)

        self.map = np.load("./coSimData/MeshData/map_used.npy")
        self.fluidSurrogate = FluidSurrog()
        with open("./coSimData/PodFilteringData/fluidSurrogate.pkl", 'rb') as inp:
            self.fluidSurrogate = pickle.load(inp)

        self.solidSurrogate = solid_ROM()
        with open("./ROMs/Double_ROM_models/accelTest.pkl", 'rb') as inp:
            self.solidSurrogate = pickle.load(inp)

        self.savedModes = np.load(
            "./coSimData/PodFilteringData/savedModes.npy")

        iteration_horizon = self.settings["iteration_horizon"].GetInt()
        timestep_horizon = self.settings["timestep_horizon"].GetInt()
        self.alpha = self.settings["alpha"].GetDouble()
        self.alpha2 = self.settings["alpha2"].GetDouble()
        self.alpha3 = self.settings["alpha3"].GetDouble()

        # ====== Training settings ======
        self.save_tr_data = self.settings["save_tr_data"].GetBool()
        self.surr_J = None
        # ====== Prediction settings ======
        self.launch_pred = self.settings["prediction_launch_time"].GetDouble(
        )
        self.end_pred = self.settings["prediction_end_time"].GetDouble()
        self.PerturbTimes = self.settings["PerturbTimes"].GetInt()
        self.epsRand = self.settings["epsRand"].GetDouble()
        # ====== Training Data storing ======
        self.x_k = []
        self.angles = []
        self.delta_x_par = []
        self.r_orth = []
        self.dists = []
        self.ort_w_arr = []
        self.sizes = []
        self.old_delt_x = []

        self.R = deque(maxlen=iteration_horizon)
        self.X = deque(maxlen=iteration_horizon)
        self.q = timestep_horizon - 1
        self.v_old_matrices = deque(maxlen=self.q)
        self.w_old_matrices = deque(maxlen=self.q)
        self.V_new = []
        self.W_new = []
        self.V_old = []
        self.W_old = []

    def is_in_prediction_region(self, current_t):
        return (current_t >= self.launch_pred) and (current_t <= self.end_pred)

    def _SaveData(self, x_k, delta_x_par, r_orth, n, dists, old_delt_x, new_ort_w, angles):
        self.x_k.append(x_k)
        self.delta_x_par.append(delta_x_par)
        self.old_delt_x.append(old_delt_x)
        self.r_orth.append(r_orth)
        self.dists.append(dists)
        self.sizes.append(n)
        self.ort_w_arr.append(new_ort_w)
        self.angles.append(angles)

        with open("./coSimData/X_k.npy", 'wb') as f:
            np.save(f, np.array(self.x_k).T)
        with open("./coSimData/R_orth.npy", 'wb') as f:
            np.save(f, np.array(self.r_orth).T)
        with open("./coSimData/ort_w.npy", 'wb') as f:
            np.save(f, np.array(self.ort_w_arr))
        with open("./coSimData/angles.npy", 'wb') as f:
            np.save(f, np.array(self.angles).T)

    def qr_filter(self, Q, R, V, W):

        epsilon = self.settings["epsilon"].GetDouble()
        cols = V.shape[1]
        i = 0
        while i < cols:
            if np.abs(np.diag(R)[i]) < epsilon:
                if self.echo_level > 2:
                    cs_print_info(self._ClassName(),
                                  "QR Filtering")
                ids_tokeep = np.delete(np.arange(0, cols), i)
                V = V[:, ids_tokeep]
                cols = V.shape[1]
                W = W[:, ids_tokeep]
                Q, R = np.linalg.qr(V)
            else:
                i += 1

        return Q, R, V, W

    # UpdateSolution(r, x)
    # @param r residual r_k
    # @param x solution x_k
    # Computes the approximated update in each iteration.
    def UpdateSolution(self, r, x):
        current_t = self.current_t
        self.R.appendleft(deepcopy(r))
        self.X.appendleft(deepcopy(x))  # r = x~ - x
        row = len(r)
        col = len(self.R) - 1
        k = col
        num_old_matrices = len(self.v_old_matrices)

        if self.V_old == [] and self.W_old == []:  # No previous vectors to reuse
            if k == 0:
                # For the first iteration in the first time step, do relaxation only
                if self.echo_level > 3:
                    cs_print_info(self._ClassName(
                    ), "Doing relaxation in the first iteration with factor = ", "{0:.1g}".format(self.alpha))
                return self.alpha * r
            else:
                if self.echo_level > 3:
                    cs_print_info(self._ClassName(),
                                  "Doing multi-vector extrapolation")
                    cs_print_info(self._ClassName(),
                                  "Number of new modes: ", col)
                # will be transposed later
                self.V_new = np.empty(shape=(col, row))
                for i in range(0, col):
                    self.V_new[i] = self.R[i] - self.R[i + 1]
                self.V_new = self.V_new.T
                V = self.V_new

                # Check the dimension of the newly constructed matrix
                if (V.shape[0] < V.shape[1]) and self.echo_level > 0:
                    cs_print_warning(self._ClassName(
                    ), ": " + colors.red("WARNING: column number larger than row number!"))

                # Construct matrix W(differences of predictions)
                # will be transposed later
                self.W_new = np.empty(shape=(col, row))
                for i in range(0, col):
                    self.W_new[i] = self.X[i] - self.X[i + 1]
                self.W_new = self.W_new.T
                W = self.W_new

                # Solve least-squares problem
                delta_r = -self.R[0]
                Q, R = np.linalg.qr(V)
                b = Q.T @ delta_r
                c = sp.linalg.solve_triangular(R, b)

                # Compute the update
                # delta_x = np.dot(W, c) - delta_r
                paral_part = Q @ b
                delt_r_orth = delta_r - paral_part

                if self.is_in_prediction_region(current_t):
                    # if self.surr_J is not None:
                    #     maxnColumns = min(
                    #         max(1, Q.shape[1]-2), self.surr_Q.shape[1])
                    #     print(Q.shape, "and", maxnColumns)

                    #     surrogQ = self.surr_Q
                    #     surrogR = self.surr_R

                    #     b2 = surrogQ.T @ delt_r_orth
                    #     c2 = sp.linalg.solve_triangular(surrogR, b2)
                    #     delta_x = np.dot((W), c) + np.dot((self.surrDeltaX), c2) - (
                    #         delt_r_orth - surrogQ @ surrogQ.T @ delt_r_orth)
                    delta_x = np.dot((W), c) - delt_r_orth
                else:
                    delta_x = np.dot((W), c) - delt_r_orth

                return delta_x
        else:  # previous vectors can be reused
            if k == 0:  # first iteration
                if self.echo_level > 3:
                    cs_print_info(self._ClassName(),
                                  "Using matrices from previous time steps")
                    cs_print_info(
                        self._ClassName(), "Number of previous matrices: ", num_old_matrices)

                self.ort_w = np.random.sample()

                V = self.V_old
                W = self.W_old
                # Solve least-squares problem
                delta_r = -self.R[0]
                Q, R = np.linalg.qr(V)
                Q, R, V, W = self.qr_filter(Q, R, V, W)
                b = Q.T @ delta_r
                c = sp.linalg.solve_triangular(R, b)

                # Compute the update
                delta_x = np.dot(W, c) - delta_r
                paral_part = Q @ b
                delt_r_orth = delta_r - paral_part

                if self.is_in_prediction_region(current_t):
                    if self.surr_J is not None:
                        maxnColumns = min(
                            max(1, Q.shape[1]-2), self.surr_Q.shape[1])
                        print(Q.shape, "and", maxnColumns)

                        surrogQ = self.surr_Q
                        surrogR = self.surr_R

                        b2 = surrogQ.T @ delt_r_orth
                        c2 = sp.linalg.solve_triangular(surrogR, b2)
                        delta_x = np.dot((W), c) + np.dot((self.surrDeltaX), c2) - 0.8 * (
                            delt_r_orth - surrogQ @ surrogQ.T @ delt_r_orth)
                    else:
                        delta_x = np.dot((W), c) - delt_r_orth
                else:
                    delta_x = np.dot((W), c) - delt_r_orth

                return delta_x
            else:
                # For other iterations, construct new V and W matrices and combine them with old ones
                if self.echo_level > 3:
                    cs_print_info(self._ClassName(),
                                  "Doing multi-vector extrapolation")
                    cs_print_info(self._ClassName(),
                                  "Number of new modes: ", col)
                    cs_print_info(
                        self._ClassName(), "Number of previous matrices: ", num_old_matrices)
                # Construct matrix V (differences of residuals)
                # will be transposed later
                self.V_new = np.empty(shape=(col, row))
                for i in range(0, col):
                    self.V_new[i] = self.R[i] - self.R[i + 1]
                self.V_new = self.V_new.T
                V = np.hstack((self.V_new, self.V_old))
                # Check the dimension of the newly constructed matrix
                if (V.shape[0] < V.shape[1]) and self.echo_level > 0:
                    cs_print_warning(self._ClassName(
                    ), ": " + colors.red("WARNING: column number larger than row number!"))

                # Construct matrix W(differences of predictions)
                # will be transposed later
                self.W_new = np.empty(shape=(col, row))
                for i in range(0, col):
                    self.W_new[i] = self.X[i] - self.X[i + 1]
                self.W_new = self.W_new.T
                W = np.hstack((self.W_new, self.W_old))

                # Solve least-squares problem
                delta_r = -self.R[0]
                Q, R = np.linalg.qr(V)
                Q, R, V, W = self.qr_filter(Q, R, V, W)
                b = Q.T @ delta_r
                c = sp.linalg.solve_triangular(R, b)

                # Compute the update
                delta_x = np.dot(W, c) - delta_r
                paral_part = Q @ b
                delt_r_orth = delta_r - paral_part

                if self.is_in_prediction_region(current_t):
                    # if self.surr_J is not None:
                    #     maxnColumns = min(
                    #         max(1, Q.shape[1]-2), self.surr_Q.shape[1])
                    #     print(Q.shape, "and", maxnColumns)

                    #     surrogQ = self.surr_Q
                    #     surrogR = self.surr_R

                    #     b2 = surrogQ.T @ delt_r_orth
                    #     c2 = sp.linalg.solve_triangular(surrogR, b2)
                    #     delta_x = np.dot((W), c) + np.dot((self.surrDeltaX), c2) - (
                    #         delt_r_orth - surrogQ @ surrogQ.T @ delt_r_orth)
                    delta_x = np.dot((W), c) - delt_r_orth
                else:
                    delta_x = np.dot((W), c) - delt_r_orth

                return delta_x

    def ReceiveJacobian(self, J, Q, R, X):
        if J is not None:
            self.surr_J = J.copy()
            self.surr_Q = Q.copy()
            self.surr_R = R.copy()
            self.surrDeltaX = X.copy()

    def ReceivePredictedSol(self, newX):
        pass

    def ReceivePreviousSol(self, newX):
        pass

    # FinalizeSolutionStep()
    # Finalizes the current time step and initializes the next time step.
    def FinalizeSolutionStep(self):
        if self.V_new != [] and self.W_new != []:
            self.v_old_matrices.appendleft(self.V_new)
            self.w_old_matrices.appendleft(self.W_new)
        if self.v_old_matrices and self.w_old_matrices:
            self.V_old = np.concatenate(self.v_old_matrices, 1)
            self.W_old = np.concatenate(self.w_old_matrices, 1)
        # Clear the buffer
        if self.R and self.X:
            if self.echo_level > 3:
                cs_print_info(self._ClassName(), "Cleaning")
            self.R.clear()
            self.X.clear()
        self.V_new = []
        self.W_new = []
        self.surr_J = None

    def FinalizeNonLinearIteration(self, current_t, currentCharDisp=0.):

        self.current_t = current_t
        self.currentCharDisp = currentCharDisp

    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "iteration_horizon"      : 20,
            "timestep_horizon"       : 1,
            "alpha"                  : 0.125,
            "alpha2"                  : 0.125,
            "alpha3"                  : 0.125,
            "epsilon"                : 3e-4,
            "save_tr_data"           : true,
            "prediction_end_time"    : 6.0,
            "prediction_launch_time" : 3.0,
            "PerturbTimes"           : 6,
            "epsRand"                : 1e-3,
            "orthogonal_w"           : {"type" : "fixed", "value" : 1.0}
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

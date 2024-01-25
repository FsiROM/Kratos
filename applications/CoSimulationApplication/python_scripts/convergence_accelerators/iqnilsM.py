## @module iqnilsM
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
#from .accel_surrogate import accel_SURROG
import pickle


def Create(settings):
    SettingsTypeCheck(settings)
    return IQNILSMConvergenceAccelerator(settings)

# Class IQNILSConvergenceAccelerator.
# This class contains the implementation of the IQN-ILS method and helper functions.
# Reference: Joris Degroote, PhD thesis "Development of algorithms for the partitioned simulation of strongly coupled fluid-structure interaction problems", 84-91.

class Specific_w():

    def __init__(self,):
        pass

    def train(self, dims = None, means = None, stds = None, amazon = None, imported = False):

        if imported:
            self.dims = dims
            self.means = means
            self.stds = stds
            self.amazon = amazon

    def pred(self, prev_load, prev_disp, dist, dim):

        input_array = np.array([1., np.linalg.norm(prev_load), prev_disp, dist]).reshape((1, 4))

        which_dim = np.where(self.dims == dim)[0]
        if len(which_dim)>0:
            input_array = (input_array - self.means[:, [which_dim[0]]])/(self.stds[:, [which_dim[0]]])
            pred_w = self.amazon[which_dim[0]].predict(input_array)[0]
            if pred_w < 0.:
                return 0.
            elif pred_w > 1.:
                return 1.
            else:
                return pred_w
        else:
            return 0.2

class IQNILSMConvergenceAccelerator(CoSimulationConvergenceAccelerator):
    # The constructor.
    # @param iteration_horizon Maximum number of vectors to be stored in each time step.
    # @param timestep_horizon Maximum number of time steps of which the vectors are used.
    # @param alpha Relaxation factor for computing the update, when no vectors available.
    def __init__(self, settings):
        super().__init__(settings)

        self.trained_w = False
        iteration_horizon = self.settings["iteration_horizon"].GetInt()
        timestep_horizon = self.settings["timestep_horizon"].GetInt()
        self.alpha = self.settings["alpha"].GetDouble()

        # ====== Training settings ======
        self.save_tr_data = self.settings["save_tr_data"].GetBool()
        self.launch_train = self.settings["training_launch_time"].GetDouble()
        self.end_train = self.settings["training_end_time"].GetDouble()
        #self.surrogate_model = accel_SURROG()
        self.surrogate_istrained = False
        self.surr_J = None
        # ====== Prediction settings ======
        self.launch_pred = self.settings["prediction_launch_time"].GetDouble(
        )
        self.end_pred = self.settings["prediction_end_time"].GetDouble()
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

    def ReceiveJacobian(self, jacobian):
        self.train_surrogate()
        self.surr_J = jacobian

    def train_surrogate(self):
        if not self.surrogate_istrained:
            """
            self.surrogate_model.train(np.load("./model_arch_saved/label_data.npy"), np.load("./model_arch_saved/r_orth_data.npy"), np.load(
                "./model_arch_saved/subsp_dims_data.npy"), np.load("./model_arch_saved/subsp_dists_data.npy"), "./model_arch_saved/surrog_accel.pt")
            self.surrogate_istrained = True
            """
            pass

    def is_in_training_region(self, current_t):
        return self.save_tr_data and (current_t >= self.launch_train) and (current_t <= self.end_train)

    def is_in_prediction_region(self, current_t):
        return (current_t >= self.launch_pred) and (current_t <= self.end_pred)

    def _train_w(self, ):
        if self.trained_w:
            pass
        else:
            # self.spec_w = Specific_w()
            # with open('./acceleration_NN/spec_w.pkl', 'rb') as inp:
            #     self.spec_w = pickle.load(inp)
            self.trained_w = True

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
        # with open("./coSimData/Delt_X_par.npy", 'wb') as f:
        #    np.save(f, np.array(self.delta_x_par).T)
        with open("./coSimData/R_orth.npy", 'wb') as f:
            np.save(f, np.array(self.r_orth).T)
        # with open("./coSimData/subsp_dists.npy", 'wb') as f:
        #     np.save(f, np.array(self.dists))
        # with open("./coSimData/subsp_dim.npy", 'wb') as f:
        #     np.save(f, np.array(self.sizes))
        # with open("./coSimData/old_delt_x.npy", 'wb') as f:
        #    np.save(f, np.array(self.old_delt_x).T)
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
        self.R.appendleft(deepcopy(r))
        self.X.appendleft(x + r)  # r = x~ - x
        row = len(r)
        col = len(self.R) - 1
        k = col
        num_old_matrices = len(self.v_old_matrices)
        current_t = self.current_t

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
                c = sp.linalg.solve_triangular(R, b, check_finite=False)

                # Compute the update
                delta_x = np.dot(W, c) - delta_r

                if self.is_in_prediction_region(current_t) or self.is_in_training_region(current_t):
                    self._train_w()
                    paral_part = Q @ b
                    _, s, _ = np.linalg.svd(Q.T @ np.eye(row, V.shape[1]))
                    angles = self._AnglesFromS(np.arccos(s), row)
                    delt_r_orth = delta_r - paral_part
                    dist = np.linalg.norm(np.arccos(s))
                    dim = V.shape[1]

                if self.is_in_training_region(current_t):
                    self._SaveData(x, delta_x + delta_r -
                                   paral_part, delt_r_orth, dim, dist, delta_x, -1, angles)

                if self.is_in_prediction_region(current_t):
                    self.train_surrogate()
                    if False:
                        return self.surrogate_model.pred_deltax(delt_r_orth.reshape((-1, 1)), np.array([dim]), np.array([dist])).ravel() + delta_x + delta_r - paral_part
                    else:
                        return delta_x
                else:
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
                c = sp.linalg.solve_triangular(R, b, check_finite=False)

                # Compute the update
                delta_x = np.dot(W, c) - delta_r

                if self.is_in_prediction_region(current_t) or self.is_in_training_region(current_t):
                    paral_part = Q @ b
                    _, s, _ = np.linalg.svd(Q.T @ np.eye(row, V.shape[1]))
                    angles = self._AnglesFromS(np.arccos(s), row)
                    delt_r_orth = delta_r - paral_part
                    dist = np.linalg.norm(np.arccos(s))
                    dim = V.shape[1]
                    self.ort_w = self._GetOrth_w(x+r, self.currentCharDisp, dist, dim)

                    # Compute the orthogonal part
                    deltax_ort = (1 - self.ort_w) * delta_r - (1 - self.ort_w) * paral_part
                    delta_x += deltax_ort


                if self.is_in_training_region(current_t):
                    self._SaveData(x, delta_x + delta_r -
                                   paral_part, delt_r_orth, dim, dist, delta_x, self.ort_w, angles)

                if self.is_in_prediction_region(current_t) and self.surr_J is not None:
                    return delta_x + self.surr_J @ delt_r_orth
                else:
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
                c = sp.linalg.solve_triangular(R, b, check_finite=False)

                # Compute the update
                delta_x = np.dot(W, c) - delta_r

                if self.is_in_prediction_region(current_t) or self.is_in_training_region(current_t):
                    paral_part = Q @ b
                    _, s, _ = np.linalg.svd(Q.T @ np.eye(row, V.shape[1]))
                    angles = self._AnglesFromS(np.arccos(s), row)
                    delt_r_orth = delta_r - paral_part
                    dist = np.linalg.norm(np.arccos(s))
                    dim = V.shape[1]
                    self.ort_w = self._GetOrth_w(x+r, self.currentCharDisp, dist, dim)

                    # Compute the orthogonal part
                    deltax_ort = (1 - self.ort_w) * delta_r - (1 - self.ort_w) * paral_part
                    delta_x += deltax_ort

                if self.is_in_training_region(current_t):
                    self._SaveData(x, delta_x + delta_r -
                                   paral_part, delt_r_orth, dim, dist, delta_x, self.ort_w, angles)

                if self.is_in_prediction_region(current_t) and self.surr_J is not None:
                    return delta_x + self.surr_J @ delt_r_orth
                else:
                    return delta_x

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

    def _AnglesFromS(self, s, space_dim):

        r = np.zeros(space_dim)
        r[:len(s)] = s.copy()

        return r

    def ReceivePredictedSol(self, newX):
        pass

    def ReceivePreviousSol(self, newX):
        pass

    def _GetOrth_w(self, load, currentCharDisp, dist, dim):

        to_predict_w = self.settings["orthogonal_w"].Has("type")
        if to_predict_w:
            if self.settings["orthogonal_w"]["type"].GetString() == "fixed":
                if self.settings["orthogonal_w"].Has("value"):
                    return self.settings["orthogonal_w"]["value"].GetDouble()
                else:
                    return 1.
            elif self.settings["orthogonal_w"]["type"].GetString() == "ML":
                return self.spec_w.pred(load, currentCharDisp, dist, dim)
            elif self.settings["orthogonal_w"]["type"].GetString() == "random":
                return np.random.sample()
            elif self.settings["orthogonal_w"]["type"].GetString() == "random2":
                return self.ort_w
        else:
            return 1.

    def FinalizeNonLinearIteration(self, current_t, currentCharDisp = 0.):

        self.current_t = current_t
        self.currentCharDisp = currentCharDisp


    @classmethod
    def _GetDefaultParameters(cls):
        this_defaults = KM.Parameters("""{
            "iteration_horizon"      : 20,
            "timestep_horizon"       : 1,
            "alpha"                  : 0.125,
            "epsilon"                : 3e-4,
            "save_tr_data"           : true,
            "training_launch_time"   : 0.0,
            "training_end_time"      : 0.0,
            "prediction_end_time"    : 6.0,
            "prediction_launch_time" : 3.0,
            "orthogonal_w"           : {"type" : "fixed", "value" : 1.0}
        }""")
        this_defaults.AddMissingParameters(super()._GetDefaultParameters())
        return this_defaults

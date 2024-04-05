import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from scipy.interpolate import RBFInterpolator
import collections
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


# class FluidSurrog:
# #    def __init__(self, maxLen = 5000, reTrainThres = 500):

#     def __init__(self, maxLen = 6900, reTrainThres = 240):
#         self.trainIn = collections.deque(maxlen = maxLen)
#         self.trainOut = collections.deque(maxlen = maxLen)
#         self.maxLen = maxLen
#         self.countAugment = 0
#         self.reTrainThres = reTrainThres
#         self.retrain_count = 0
#         self.retrain_times = []

#     def train(self, dispData, fluidPrevData, fluidData, ):
#         podLoad = POD()
#         self.romLoad = ROM(podLoad)
#         self.romLoad.decompose(fluidData, rank = 40, normalize = True, center = True)
#         self.podLoad = podLoad

#         podDisp = POD()
#         self.romDisp = ROM(podDisp)
#         self.romDisp.decompose(dispData, rank = 9, normalize = True, center = True)
#         self.podDisp = podDisp

#         input_ = np.vstack((podDisp.pod_coeff, podLoad.project(
#             self.romLoad.normalize(self.romLoad.center(fluidPrevData)))
#                            )).T

#         for i in range(input_.shape[0]):
#             self.trainIn.appendleft(input_.T[:, [i]])
#             self.trainOut.appendleft(podLoad.pod_coeff[:, [i]])

#         self.func = RBFInterpolator(input_.copy(), podLoad.pod_coeff.T,
#                                     kernel = 'cubic', smoothing=9.5e-2)

#     def augmentData(self, newdispData, newfluidPrevData, newfluidData, current_t=-1.):
#         dispCoeff = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newdispData)))
#         prevLoadCoeff = self.podLoad.project( self.romLoad.normalize(self.romLoad.center(newfluidPrevData)))
#         outLoadCoeff = self.podLoad.project( self.romLoad.normalize(self.romLoad.center(newfluidData)))
#         input_ = np.vstack((dispCoeff, prevLoadCoeff))

#         self.trainIn.appendleft(input_.copy())
#         self.trainOut.appendleft(outLoadCoeff.copy())

#         self.countAugment += 1
#         if self.countAugment > self.reTrainThres:
#             self._reTrain()
#             self.retrain_count += 1
#             self.retrain_times.append(current_t)
#             self.countAugment = 0

#     def _reTrain(self, ):
#         print("=== - Retraining the Interpolator - ===")
#         self.func = RBFInterpolator(np.hstack(self.trainIn).T,
#                                     np.hstack(self.trainOut).T,
#                                     kernel = 'cubic', smoothing=9.5e-2)

#     def predict(self, newDisp, newPrevLoad, ):

#         coeffAllDisp = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newDisp)))
#         xTest = np.vstack((coeffAllDisp, self.podLoad.project(self.romLoad.normalize(self.romLoad.center(
#             newPrevLoad)))
#                           ))
#         LoadReconsCoeff = self.func(xTest.T).T
#         predicted_ = self.romLoad.decenter(self.romLoad.denormalize(
#             self.podLoad.inverse_project(LoadReconsCoeff)))
#         return predicted_


class FluidSurrog:
#    def __init__(self, maxLen = 5000, reTrainThres = 500):

    def __init__(self, maxLen = 6900, reTrainThres = 240):
        self.trainIn = collections.deque(maxlen = maxLen)
        self.trainOut = collections.deque(maxlen = maxLen)
        self.maxLen = maxLen
        self.countAugment = 0
        self.reTrainThres = reTrainThres
        self.retrain_count = 0
        self.retrain_times = []

    def train(self, dispData, fluidPrevData, fluidData, input_u=None, kernel='thin_plate_pline', smoothing=9.5e-2,
              rank_pres=.9999, rank_disp=.9999, degree=2):
        print(" ----- Load Reduction -----")
        podLoad = POD()
        self.romLoad = ROM(podLoad)
        self.romLoad.decompose(fluidData, rank = rank_pres, normalize = True, center = True, to_copy=False, alg="snap")
        self.podLoad = podLoad

        print(" ----- Disp Reduction -----")
        podDisp = POD()
        self.romDisp = ROM(podDisp)
        self.romDisp.decompose(dispData, rank = rank_disp, normalize = True, center = True, to_copy=False, alg="snap")
        self.podDisp = podDisp

        input_ = np.vstack((podDisp.pod_coeff, podLoad.project(self.romLoad.normalize(self.romLoad.center(
            fluidPrevData)))
            )).T
        if input_u is not None:
            input_ = np.hstack((input_, input_u.T))

        for i in range(input_.shape[0]):
            self.trainIn.appendleft(input_.T[:, [i]])
            self.trainOut.appendleft(podLoad.pod_coeff[:, [i]])

        print(" ----- Regression -----")
        self.kernel = kernel
        self.smoothing =smoothing
        if kernel == "poly":
            #self.func = make_pipeline(PolynomialFeatures(2), Ridge(alpha=smoothing))
            self.func = make_pipeline(ColumnTransformer(transformers=[
                    ('polynomialCoefficients', PolynomialFeatures(degree, include_bias=True), slice(0, podLoad.kept_rank)),
                    ],
                    remainder='passthrough'), MinMaxScaler(),
                                      Ridge(alpha=smoothing))

            self.func.fit(input_.copy(), podLoad.pod_coeff.T)
        else:
            self.func = RBFInterpolator(input_.copy(), podLoad.pod_coeff.T,
                    kernel = kernel,smoothing=smoothing)

    def augmentData(self, newdispData, newfluidPrevData, newfluidData, current_t=-1):
        dispCoeff = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newdispData)))
        prevLoadCoeff = self.podLoad.project( self.romLoad.normalize(self.romLoad.center(newfluidPrevData)))
        outLoadCoeff = self.podLoad.project( self.romLoad.normalize(self.romLoad.center(newfluidData)))
        input_ = np.vstack((dispCoeff, prevLoadCoeff))

        self.trainIn.appendleft(input_.copy())
        self.trainOut.appendleft(outLoadCoeff.copy())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain()
            self.retrain_count += 1
            self.retrain_times.append(current_t)
            self.countAugment = 0

    def _reTrain(self, ):
        print("=== - Retraining the Interpolator - ===")
        if self.kernel == "poly":
            self.func.fit(np.hstack(self.trainIn).T,
                                        np.hstack(self.trainOut).T)
        else:
            self.func = RBFInterpolator(np.hstack(self.trainIn).T,
                                        np.hstack(self.trainOut).T,
                                        kernel = self.kernel, smoothing=self.smoothing)

    def predict(self, newDisp, newPrevLoad, input_u=None ):

        coeffAllDisp = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newDisp)))
        xTest = np.vstack((coeffAllDisp, self.podLoad.project(self.romLoad.normalize(self.romLoad.center(
            newPrevLoad)))
                          ))
        if input_u is not None:
            xTest = np.vstack((xTest, input_u))
        if self.kernel == "poly":
            LoadReconsCoeff = self.func.predict(xTest.T).T
        else:
            LoadReconsCoeff = self.func(xTest.T).T
        predicted_ = self.romLoad.decenter(self.romLoad.denormalize(
            self.podLoad.inverse_project(LoadReconsCoeff)))
        return predicted_


class FluidSurrogBrute:
#    def __init__(self, maxLen = 5000, reTrainThres = 500):

    def __init__(self, maxLen = 6900, reTrainThres = 240):
        #self.trainIn = collections.deque(maxlen = maxLen)
        #self.trainOut = collections.deque(maxlen = maxLen)

        self.hightrainIn = collections.deque(maxlen = maxLen)
        self.hightrainOut = collections.deque(maxlen = maxLen)

        self.maxLen = maxLen
        self.countAugment = 0
        self.reTrainThres = reTrainThres

    def train(self, dispData, fluidPrevData, fluidData, storeData=True, StrucTrain=True):
        self.dispDim = dispData.shape[0]
        self.loadDim = fluidData.shape[0]

        podLoad = POD()
        self.romLoad = ROM(podLoad)
        self.romLoad.decompose(fluidData, rank = 40, normalize = False, center = False)
        self.podLoad = podLoad

        if StrucTrain:
            podDisp = POD()
            self.romDisp = ROM(podDisp)
            self.romDisp.decompose(dispData, rank = .9999, normalize = True, center = True)
            self.podDisp = podDisp

        if storeData:
            input_ = np.vstack((dispData, fluidPrevData)).T
            for i in range(input_.shape[0]):
                self.hightrainIn.appendleft(input_.T[:, [i]])
                self.hightrainOut.appendleft(fluidData[:, [i]])

        if StrucTrain:
            reducedInput_ = np.vstack((podDisp.pod_coeff, podLoad.project(fluidPrevData))).T
        else:
            reducDisp = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(dispData)))
            reducedInput_ = np.vstack((reducDisp, podLoad.project(fluidPrevData))).T

        self.func = RBFInterpolator(reducedInput_.copy(), podLoad.pod_coeff.T,
                                    kernel = 'cubic', smoothing=9.5e-2)

    def augmentData(self, newdispData, newfluidPrevData, newfluidData):

        self.hightrainIn.appendleft(np.vstack((newdispData, newfluidPrevData)).copy())
        self.hightrainOut.appendleft(newfluidData.copy())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain()
            self.countAugment = 0

    def _reTrain(self, ):
        print("=== - Retraining the Interpolator - ===")

        # Recovering High Dim Data
        self.train(np.hstack(self.hightrainIn)[:self.dispDim, :][:, ::-1], np.hstack(self.hightrainIn)[self.dispDim:, :][:, ::-1],
                   np.hstack(self.hightrainOut)[:, ::-1], storeData=False, StrucTrain=True)

    def predict(self, newDisp, newPrevLoad, ):

        coeffAllDisp = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newDisp)))
        xTest = np.vstack((coeffAllDisp, self.podLoad.project(newPrevLoad)))
        LoadReconsCoeff = self.func(xTest.T).T
        predicted_ = self.podLoad.inverse_project(LoadReconsCoeff)
        return predicted_

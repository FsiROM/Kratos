import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from scipy.interpolate import RBFInterpolator
import collections

class FluidSurrog:

    def __init__(self, maxLen = 4300, reTrainThres = 240):
        self.trainIn = collections.deque(maxlen = maxLen)
        self.trainOut = collections.deque(maxlen = maxLen)
        self.maxLen = maxLen
        self.countAugment = 0
        self.reTrainThres = reTrainThres

    def train(self, dispData, fluidPrevData, fluidData, ):
        podLoad = POD()
        self.romLoad = ROM(podLoad)
        self.romLoad.decompose(fluidData, rank = 50, normalize = False, center = False)
        self.podLoad = podLoad

        podDisp = POD()
        self.romDisp = ROM(podDisp)
        self.romDisp.decompose(dispData, rank = .9999, normalize = True, center = True)
        self.podDisp = podDisp

        input_ = np.vstack((podDisp.pod_coeff, podLoad.project(fluidPrevData))).T

        for i in range(input_.shape[0]):
            self.trainIn.appendleft(input_.T[:, [i]])
            self.trainOut.appendleft(podLoad.pod_coeff[:, [i]])

        self.func = RBFInterpolator(input_.copy(), podLoad.pod_coeff.T,
                                    kernel = 'cubic', smoothing=9.5e-2)

    def augmentData(self, newdispData, newfluidPrevData, newfluidData):
        dispCoeff = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newdispData)))
        prevLoadCoeff = self.podLoad.project(newfluidPrevData)
        outLoadCoeff = self.podLoad.project(newfluidData)
        input_ = np.vstack((dispCoeff, prevLoadCoeff))

        self.trainIn.appendleft(input_.copy())
        self.trainOut.appendleft(outLoadCoeff.copy())

        self.countAugment += 1
        if self.countAugment > self.reTrainThres:
            self._reTrain()
            self.countAugment = 0

    def _reTrain(self, ):
        print("=== - Retraining the Interpolator - ===")
        self.func = RBFInterpolator(np.hstack(self.trainIn).T,
                                    np.hstack(self.trainOut).T,
                                    kernel = 'cubic', smoothing=9.5e-2)

    def predict(self, newDisp, newPrevLoad, ):

        coeffAllDisp = self.podDisp.project(self.romDisp.normalize(self.romDisp.center(newDisp)))
        xTest = np.vstack((coeffAllDisp, self.podLoad.project(newPrevLoad)))
        LoadReconsCoeff = self.func(xTest.T).T
        predicted_ = self.podLoad.inverse_project(LoadReconsCoeff)
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

# CoSimulation imports
from KratosMultiphysics.CoSimulationApplication.function_callback_utility import GenericCallFunction

# Other imports
import numpy as np
import json
import os
from scipy.optimize import root
from rom_am.solid_rom import solid_ROM
import pickle


class ArterySolver(object):

    def __init__(self, input_name):

        # mimicing two constructors
        if isinstance(input_name, dict):
            parameters = input_name

        elif isinstance(input_name, str):
            if not input_name.endswith(".json"):
                input_name += ".json"

            with open(input_name, 'r') as ProjectParameters:
                parameters = json.load(ProjectParameters)

        else:
            raise Exception(
                "The input has to be provided as a dict or a string")

        self.r0 = 1 / np.sqrt(np.pi)  # radius of the tube
        self.a0 = self.r0**2 * np.pi  # cross sectional area
        # timestep size, set it to a large value to enforce tau from precice_config.xml
        self.tau = 10**10
        self.N = 100  # number of elements in x direction
        self.p0 = 0  # pressure at outlet
        self.L = 10  # length of tube/simulation domain
        self.dt = .1

        self.E = 10000  # elasticity module
        self.c_mk = np.sqrt(self.E / 2 / self.r0)  # wave speed

        self.rom_use = False
        self.save_rom = False
        self.load_rom = False
        self.extrapolate_ = False
        self.sectionData = []

        # calculate initial
        self.dimensions = 2
        self.crossSectionLength = self.a0 * np.ones(self.N + 1)
        self.grid = np.zeros([self.N + 1, self.dimensions])
        self.grid[:, 0] = np.linspace(0, self.L, self.N + 1)  # x component
        # np.linspace(0, config.L, N+1)  # y component, leave blank
        self.grid[:, 1] = 0

        self.sol_rom = solid_ROM()
        self.xData = []
        self.yData = []
        self.trainedROM = False

    def strs(self, eps):
        res = np.empty_like(eps)

        res[(eps < .002)+(eps > -.002)] = 25/.002 * \
            eps[(eps < .002)+(eps > -.002)]
        a = (30-25)/(.004-.002)
        b = 30 - a*.004
        res[eps > .002] = a*eps[eps > .002]+b
        b = -30 + a * .004
        res[eps < -.002] = a*eps[eps < -.002]+b

        return res

    def solid_fom(self, pressure):
        def fun(x): return ((pressure*x)) - self.strs((x-self.r0)/self.r0)
        res = root(fun, self.Section, tol = 5e-12)
        return np.pi * res.x**2

    def Initialize(self):
        # solution buffer
        self.Section = self.a0 * np.ones(self.N + 1)
        self.SectionOld = self.Section.copy()
        self.time = 0.

        # apply external load as an initial impulse
        self.pressure = self.p0 * np.ones(self.N + 1)

    def OutputSolutionStep(self):
        self.sectionData.append(self.Section.copy().reshape((-1, 1)))
        np.save("section.npy", np.hstack(self.sectionData))

    def AdvanceInTime(self, current_time):
        self.time = current_time + self.dt
        return self.time

    def SolveSolutionStep(self):
        self.xData.append(self.pressure.reshape((-1, 1)))
        if self.time <= 5000.0:
            self.Section = self.solid_fom(self.pressure)

        else:
            if not self.trainedROM:
                # self.sol_rom.train(np.hstack(self.xData), np.hstack(self.yData), rank_disp=25, rank_pres=25,
                #                    regression_model="RBF", dispReduc_model="POD", norm_regr=[False, True],
                #                    norm=["minmax", "minmax"])
                # with open('./rom_good_data/accelTestGlobal.pkl', 'rb') as inp:
                with open('./rom_good_data/strongerROM.pkl', 'rb') as inp:
                    self.sol_rom = pickle.load(inp)
                self.trainedROM = True
                # with open('accelTestGlobal.pkl', 'wb') as outp:
                #     pickle.dump(self.sol_rom, outp, pickle.HIGHEST_PROTOCOL)

            print("--- ROM prediction Regime ---")
            self.Section = self.sol_rom.pred(self.pressure.reshape((-1, 1))).ravel()
        # self.Section = self.solid_fom(self.pressure)

        self.yData.append(self.Section.reshape((-1, 1)))
        np.save("solidPres.npy", np.hstack(self.xData))
        np.save("solidSect.npy", np.hstack(self.yData))

    def FinalizeSolutionStep(self):
        self.Section = self.solid_fom(self.pressure) # Finalizing with converged pressure
        self.SectionOld = self.Section.copy()

    def GetSolutionStepValue(self):
        return self.Section.copy()

    def SetSolutionStepValue(self, value):
        self.Section = value.copy()

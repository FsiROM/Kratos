# CoSimulation imports
from KratosMultiphysics.CoSimulationApplication.function_callback_utility import GenericCallFunction

# Other imports
import numpy as np
import json
import os
from scipy.optimize import root
from .thetaScheme import perform_partitioned_implicit_euler_step

class BloodSolver(object):

    def __init__(self, input_name):

        # mimicing two constructors
        if isinstance(input_name, dict):
            parameters = input_name

        elif isinstance(input_name, str):
            if not input_name.endswith(".json"):
                input_name += ".json"

            with open(input_name,'r') as ProjectParameters:
                parameters = json.load(ProjectParameters)

        else:
            raise Exception("The input has to be provided as a dict or a string")


        self.mu_0 = parameters["mu"]["mu_0"]
        self.mu_1 = parameters["mu"]["mu_1"]
        self.law = parameters["law"]
        self.bC = parameters["bC"]

        self.pressureData = []
        self.velocityData = []

        self.r0 = 1 / np.sqrt(np.pi)  # radius of the tube
        self.a0 = self.r0**2 * np.pi  # cross sectional area
        self.t_shift = 0  # temporal shift of variation
        self.p0 = 0  # pressure at outlet
        self.kappa = 100
        self.rho = 1
        self.p0 = 0.
        self.L = 10  # length of tube/simulation domain
        self.N = 100
        self.dx = self.L / self.kappa
        self.dt = 0.1

        self.grid = np.zeros([self.N + 1, 2])
        self.grid[:, 0] = np.linspace(0, self.L, self.N + 1)  # x component
        self.grid[:, 1] = 0  # np.linspace(0, config.L, N+1)  # y component, leave blank

        self.yData = []
        # helper function to create constant cross section
        self.ComputeInVelocity()

    def ComputeInVelocity(self, ):
        freq = self.mu_0
        # freq = 2.
        pres = 120.
        c = -.002
        ntt = 1200
        dt = .1
        b = 0.
        d2 = 3.
        e = -.02
        a = -1.
        d1 = -1


        def p(t): return pres * np.cos(freq * t)
        def d(t): return d1 + d2 * p(t)
        def v_dot(u, v, t): return c * u**3 + b * u**2 + a * u + d(t) + e * v

        input_t = np.arange(ntt)*dt

        from scipy.integrate import solve_ivp
        def f(t, y): return np.array([y[1], v_dot(y[0], y[1], t)])
        self.solInVeloc = solve_ivp(f, [0, ntt*dt], np.array([10., 0]), t_eval=input_t)

    def velocity_in(self, t):
        constant_ = self.mu_1
        if t <= 20.:
            ramp = 1.
        elif t > 20. and t<= 60.:
            ramp = 0.9+0.1 * np.sin(t * np.pi / (40))
        else:
            ramp = 0.8
        return (self.solInVeloc.y[0][int(t/self.dt)]/60. + constant_)*ramp

    def Initialize(self):
        #solution buffer
        self.pressure = self.p0 * np.ones(self.N + 1)
        self.Section = self.a0 * np.ones(self.N + 1)
        self.velocity = self.velocity_in(0.) * np.ones(self.N + 1)
        self.velocity_old = self.velocity.copy()
        self.Section_old = self.Section.copy()
        self.pressure_old = self.pressure.copy()

        self.time = 0.

    def OutputSolutionStep(self):
        self.pressureData.append(self.pressure.copy().reshape((-1, 1)))
        self.velocityData.append(self.velocity.copy().reshape((-1, 1)))
        np.save("coSimData/pressure.npy", np.hstack(self.pressureData))
        np.save("coSimData/velocity.npy", np.hstack(self.velocityData))

    def AdvanceInTime(self, current_time):
        self.time = current_time + self.dt
        return self.time

    def SolveSolutionStep(self):
        self.velocity, self.pressure, success = perform_partitioned_implicit_euler_step(
            self.velocity_old, self.pressure_old, self.Section_old, self.Section, self.dx, self.dt, self.velocity_in(
                self.time), custom_coupling=True, pres=0., law = self.law, bC=self.bC)

        self.yData.append(self.pressure.reshape((-1, 1)))
        np.save("coSimData/fluidPres.npy", np.hstack(self.yData))

    def FinalizeSolutionStep(self):
        self.velocity_old = self.velocity.copy()
        self.pressure_old = self.pressure.copy()
        self.Section_old = self.Section.copy()

    def GetSolutionStepValue(self):
        return self.pressure.copy()

    def SetSolutionStepValue(self, value):
        self.pressure = value.copy()

from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis

class CoRomSimulationAnalysis(CoSimulationAnalysis):

    def __init__(self, cosim_settings, models=None, inputReduc_model=None, regression_model=None, outputReduc_model=None):
        super().__init__(cosim_settings, models)
        self._GetSolver().ReceiveRomComponents(inputReduc_model, regression_model, outputReduc_model)

from statistics import mode
import time
import matplotlib.pyplot as plt
import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
# from lsdo_ecm.ecm_system import ODESystemNative
from lsdo_ecm.ecm_system_pretrain import ODESystemNative
# from lsdo_ecm.ecm_system_pretrain_ext import ODESystemNative
import csdl
import csdl_om
import numpy as np

from lsdo_ecm.ecm_preprocessing import ECMPreprocessingModel
"""
ECM using ozone
"""

# ODE problem CLASS


class ODEProblemTest(ODEProblem):
    def setup(self):
        num_cells = 1

        # Outputs. coefficients for field outputs must be defined as a CSDL variable before the integrator is created in RunModel
        # self.add_field_output('field_output',
        #                       state_name='SoC',
        #                       coefficients_name='coefficients')

        # add parameters
        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('power_profile',
                           dynamic=True,
                           shape=(self.num_times,1))

        self.add_parameter('n_parallel',
                           dynamic=False,
                           shape=(1,))

        # Inputs names correspond to respective upstream CSDL variables

        self.add_state('SoC',
                       'dSoC_dt',
                       initial_condition_name='SoC_0',
                       shape=num_cells,
                       output='SoC_integrated')
        self.add_state('U_Th',
                       'dU_Th_dt',
                       initial_condition_name='U_Th_0',
                       shape=num_cells,output='U_Th_integrated')
        self.add_state('T_cell',
                       'dT_cell_dt',
                       initial_condition_name='T_cell_0',output='T_cell_integrated',
                       shape=num_cells)

        self.add_times(step_vector='h')
        # self.add_profile_output('T_cell_all',
        #                         state_name='T_cell',
        #                         shape=(num_cells, ))
        # self.add_profile_output('U_Th_all',
        #                         state_name='U_Th',
        #                         shape=(num_cells, ))
        # self.add_profile_output('SoC_all',
        #                         state_name='SoC',
        #                         shape=(num_cells, ))

        # Create model containing integrator
        # We can also pass through parameters to the ODE system from this model.

        # Output Variables

        # Define ODE system. We have three possible choices as defined in 'ode_systems.py'. Any of the three methods yield identical results:

        self.set_ode_system(ODESystemNative)  # Uncomment for Method 2
        # self.set_ode_system(ODESystemNativeSparse)  # Uncomment for Method 3


# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('t_end')
        # self.parameters.declare('power_profile')
        self.parameters.declare('num_times')
        self.parameters.declare('num_cells')
        self.parameters.declare('num_segments')
        self.parameters.declare('n_s',default=200)

    def define(self):
        t_end = self.parameters['t_end']
        num_cells = self.parameters['num_cells']
        num_times = self.parameters['num_times']
        num_segments = self.parameters['num_segments']

        # n_s = 190
        n_s = self.parameters['n_s']
        # assuming 46800 3230

        # assuming 21700 16150
        # n_p = int(16150 / n_s)

        h_stepsize = t_end/(num_times-1)

        # Create given inputs
        # Coefficients for field output
        # coeffs = self.create_input('coefficients',
        #                            np.ones(num_times) / (num_times))
        # Initial condition for state
        self.create_input('SoC_0', np.ones(num_cells) * 1.)
        self.create_input('U_Th_0', np.ones(num_cells) * 0.)
        self.create_input('T_cell_0', np.ones(num_cells) * 20.0)

        # Create parameter for power_profile (dummy values right now)


        # num_times_ = 1
        input_power = self.declare_variable('input_power',
                          shape=(num_segments, 1))
        input_time  =self.declare_variable('input_time',shape=(num_segments,1))

        submodel = ECMPreprocessingModel(
            num_segments=num_segments,
            t_end=t_end,
            num_times=num_times)

        self.add(submodel, 'ECMPreprocessingModel')
        power_profile = self.declare_variable('power_profile',
                                      shape=(num_times, 1))
        n_parallel = self.declare_variable('n_parallel',
                                      shape=(1,))
        # self.register_output('n_parallel_dummay',n_parallel+1)
        # power_profile = csdl.reshape(power * 1000 / (n_s * n_p),
        #                              (num_times, ))

        # Add to csdl model which are fed into ODE Model
        # power_profilei = self.register_output('power_profile', power_profile)

        # Timestep vector
        h_vec = np.ones(num_times - 1) * h_stepsize
        self.create_input('h', h_vec)

        # Create model containing integrator
        # We can also pass through parameters to the ODE system from this model.
        # params_dict = {'power_profile': power_profile}
        ODEProblem = ODEProblemTest(
            # 'BackwardEuler',
            'ForwardEuler',
            # 'RK4',
            'time-marching',
            # 'solver-based',
            num_times,
            display='default',
            visualization='None',
        )
        # ODEProblem_instance
        para_dict={'n_s':n_s}
        self.add(ODEProblem.create_solver_model(ODE_parameters=para_dict), 'subgroup', ['*'])





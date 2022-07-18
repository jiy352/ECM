import time
import matplotlib.pyplot as plt
import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
from lsdo_ecm.ecm_system import ODESystemNative
import csdl
import csdl_om
import csdl_lite
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
        self.add_field_output('field_output',
                              state_name='SoC',
                              coefficients_name='coefficients')

        # add parameters
        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('power_profile',
                           dynamic=True,
                           shape=(self.num_times))

        # Inputs names correspond to respective upstream CSDL variables

        self.add_state('SoC',
                       'dSoC_dt',
                       initial_condition_name='SoC_0',
                       shape=num_cells,
                       output='SoC_integrated')
        self.add_state('U_Th',
                       'dU_Th_dt',
                       initial_condition_name='U_Th_0',
                       shape=num_cells)
        self.add_state('T_cell',
                       'dT_cell_dt',
                       initial_condition_name='T_cell_0',
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
        self.parameters.declare('delta_t')
        # self.parameters.declare('power_profile')
        self.parameters.declare('num_times')
        self.parameters.declare('num_cells')
        self.parameters.declare('num_segments')

    def define(self):
        delta_t = self.parameters['delta_t']
        num_cells = self.parameters['num_cells']
        num_times = self.parameters['num_times']
        num_segments = self.parameters['num_segments']

        ########################################################################################################
        # 1. load aircraft time profile
        t_taxi = 10
        t_takeoff = 40
        t_climb = 4.4 * 60
        t_cruise = 16.7 * 60
        t_landing = 70
        t_cruise_res = 20 * 60

        # 2. load aircraft power profile (kW)
        P_taxi = 46.8
        p_takeoff = 829.2
        P_climb = 524.1
        P_cruise = 282.0
        P_landing = 829.2
        P_cruise_res = 282.0
        p_list = [
            P_taxi, p_takeoff, P_climb, P_cruise, P_landing, P_cruise_res
        ]

        # 3. setup delat_t, and define each process in the discretized time step

        t_list = [
            t_taxi, t_takeoff, t_climb, t_cruise, t_landing, t_cruise_res
        ]
        t_total = sum(t_list)
        t_step_process = np.cumsum(t_list) / delta_t
        t_step_process_round = [int(round(num)) for num in t_step_process]

        time = np.arange(0,
                         round(t_total / delta_t + 1) * (delta_t),
                         delta_t)  # shape=258

        n_s = 190
        # n_s = 1000
        # assuming 46800 3230

        # assuming 21700 16150
        n_p = int(16150 / n_s)

        h_stepsize = delta_t

        # Create given inputs
        # Coefficients for field output
        coeffs = self.create_input('coefficients',
                                   np.ones(num_times) / (num_times))
        # Initial condition for state
        self.create_input('SoC_0', np.ones(num_cells) * 1.0)
        self.create_input('U_Th_0', np.ones(num_cells) * 0.)
        self.create_input('T_cell_0', np.ones(num_cells) * 20.0)

        # Create parameter for power_profile (dummy values right now)
        '''hard code the input_power and input_time'''

        num_nodes = 6
        # num_times_ = 1
        input_power = self.declare_variable('input_power',
                          val=np.array(p_list).reshape(num_nodes, 1))
        # input_time = self.declare_variable('input_time',
        #                   val=np.array(t_list).reshape(num_times_, 1))
        input_time  =self.declare_variable('input_time',shape=(num_segments,1))
        submodel = ECMPreprocessingModel(
            num_nodes=num_nodes,
            delta_t=delta_t,
            t_step_round=np.array(t_step_process_round) + 1)

        self.add(submodel, 'ECMPreprocessingModel')
        power = self.declare_variable('power',
                                      shape=(t_step_process_round[-1] + 1, 1))
        power_profile = csdl.reshape(power * 1000 / (n_s * n_p),
                                     (t_step_process_round[-1] + 1, ))

        # Add to csdl model which are fed into ODE Model
        power_profilei = self.register_output('power_profile', power_profile)
        dummy = self.register_output('dummy_time', input_time+0.)

        # Timestep vector
        h_vec = np.ones(num_times - 1) * h_stepsize
        self.create_input('h', h_vec)

        # Create model containing integrator
        # We can also pass through parameters to the ODE system from this model.
        # params_dict = {'power_profile': power_profile}
        ODEProblem = ODEProblemTest(
            'RK4',
            'time-marching',
            num_times,
            display='default',
            visualization='None',
        )
        # ODEProblem_instance
        self.add(ODEProblem.create_solver_model(), 'subgroup', ['*'])

        fo = self.declare_variable('field_output')
        self.register_output('fo', fo * 1.0)
        # self.add_design_variable('SoC_0', lower=10, upper=20)
        self.add_design_variable('U_Th_0', lower=3, upper=5)
        self.add_objective('fo')


# power = np.zeros(time.size)
# for i in range(time.size):
#     if i <= t_step_process_round[0]:
#         power[i] = p_list[0]
#     elif i > t_step_process_round[0] and i <= t_step_process_round[1]:
#         power[i] = p_list[1]
#     elif i > t_step_process_round[1] and i <= t_step_process_round[2]:
#         power[i] = p_list[2]
#     elif i > t_step_process_round[2] and i <= t_step_process_round[3]:
#         power[i] = p_list[3]
#     elif i > t_step_process_round[3] and i <= t_step_process_round[4]:
#         power[i] = p_list[4]
#     elif i > t_step_process_round[4] and i <= t_step_process_round[5]:
#         power[i] = p_list[5]
#####################################################################################################

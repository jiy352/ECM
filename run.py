from lib2to3.pgen2 import driver
import time
import matplotlib.pyplot as plt
import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
from ecm_system import ODESystemNative
import csdl
import csdl_om
import numpy as np
"""
This example showcases the following:
- ability to define multiple ODE functions with coupling. For this example, the Lotka–Volterra equations have two states (x, y) which are coupled.
- ability to pass csdl parameters to your ODE function model
- multiple ways to define the ODE model itself. More info in 'ode_systems.py' where they are defined
"""

# ODE problem CLASS


class ODEProblemTest(ODEProblem):
    def setup(self):

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
        # self.parameters.declare('power_profile')
        self.parameters.declare('num_times')
        self.parameters.declare('num_cells')

    def define(self):
        num_cells = self.parameters['num_cells']
        num_times = self.parameters['num_times']

        h_stepsize = 10.

        # Create given inputs
        # Coefficients for field output
        coeffs = self.create_input('coefficients',
                                   np.ones(num_times) / (num_times))
        # Initial condition for state
        self.create_input('SoC_0', np.ones(num_cells) * 1.0)
        self.create_input('U_Th_0', np.ones(num_cells) * 0.)
        self.create_input('T_cell_0', np.ones(num_cells) * 20.0)

        # Create parameter for power_profile (dummy values right now)
        power_profile = np.zeros(
            (num_times, ))  # dynamic parameter defined at every timestep
        # for t in range(num_times):
        #     power_profile[
        #         t] = 1.0 + t / num_times / 5.0  # dynamic parameter defined at every timestep
        power_profile = power * 1000 / (n_s * n_p)

        # Add to csdl model which are fed into ODE Model
        power_profilei = self.create_input('power_profile', power_profile)

        # Timestep vector
        h_vec = np.ones(num_times - 1) * h_stepsize
        self.create_input('h', h_vec)

        # Create model containing integrator
        # We can also pass through parameters to the ODE system from this model.
        # params_dict = {'power_profile': power_profile}
        ODEProblem = ODEProblemTest(
            'RK4',
            # ODEProblem = ODEProblemTest('BackwardEuler',
            # ODEProblem = ODEProblemTest('ForwardEuler',
            'time-marching',
            num_times,
            display='default',
            visualization='None')
        # ODEProblem_instance
        self.add(ODEProblem.create_solver_model(), 'subgroup', ['*'])
        # ODEProblem = ODEProblemTest('ExplicitMidpoint', 'time-marching', num_times, visualization='None')

        fo = self.declare_variable('field_output')
        self.register_output('fo', fo * 1.0)
        # self.add_design_variable('SoC_0', lower=10, upper=20)
        self.add_design_variable('U_Th_0', lower=3, upper=5)
        self.add_objective('fo')


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
p_list = [P_taxi, p_takeoff, P_climb, P_cruise, P_landing, P_cruise_res]

# 3. setup delat_t, and define each process in the discretized time step
delta_t = 10

t_list = [t_taxi, t_takeoff, t_climb, t_cruise, t_landing, t_cruise_res]
t_total = sum(t_list)
t_step_process = np.cumsum(t_list) / delta_t
t_step_process_round = [int(round(num)) for num in t_step_process]

time = np.arange(0,
                 round(t_total / delta_t + 1) * (delta_t),
                 delta_t)  # shape=258
power = np.zeros(time.size)
for i in range(time.size):
    if i <= t_step_process_round[0]:
        power[i] = p_list[0]
    elif i > t_step_process_round[0] and i <= t_step_process_round[1]:
        power[i] = p_list[1]
    elif i > t_step_process_round[1] and i <= t_step_process_round[2]:
        power[i] = p_list[2]
    elif i > t_step_process_round[2] and i <= t_step_process_round[3]:
        power[i] = p_list[3]
    elif i > t_step_process_round[3] and i <= t_step_process_round[4]:
        power[i] = p_list[4]
    elif i > t_step_process_round[4] and i <= t_step_process_round[5]:
        power[i] = p_list[5]
#####################################################################################################

n_s = 190
# n_s = 1000
# assuming 46800 3230

# assuming 21700 16150
n_p = int(16150 / n_s)

num_cells = 1
num_times = 260
# Simulator Object: Note we are passing in a parameter that can be used in the ode system
sim = csdl_om.Simulator(RunModel(num_times=num_times, num_cells=num_cells),
                        mode='rev')
sim.prob.run_model()
driver = sim.prob.driver = om.pyOptSparseDriver()
sim.prob.driver.options["optimizer"] = "SNOPT"
driver.options["optimizer"] = "SNOPT"
driver.opt_settings["Verify level"] = 1

driver.opt_settings["Major iterations limit"] = 100
driver.opt_settings["Minor iterations limit"] = 100000
driver.opt_settings["Iterations limit"] = 100000000
driver.opt_settings["Major step limit"] = 2.0
driver.opt_settings["Major feasibility tolerance"] = 1.0e-5
driver.opt_settings["Major optimality tolerance"] = 6.0e-6

# sim.prob.model.add_design_var("T_cell_0")
# sim.prob.run_driver()

# # Checktotals
# print(sim.prob['field_output'])
sim.prob.check_totals(of=['SoC_integrated'], wrt=['SoC_0'], compact_print=True)

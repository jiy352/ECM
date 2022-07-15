import csdl
from ozone.api import NativeSystem
import numpy as np

import scipy.sparse as sp
from lsdo_ecm.ecm_data_21700 import T_bp, SOC_bp, tU_oc, tC_Th, tR_Th, tR_0

from smt.surrogate_models import RMTB

from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class ECMPreprocessingModel(Model):
    """
    Compute various geometric properties for VLM analysis.

    parameters
    ----------
    time
    power
    Returns
    -------
    power profile
    """
    def initialize(self):
        # self.parameters.declare('surface_names', types=list)
        self.parameters.declare('num_nodes', default=5)
        self.parameters.declare('delta_t', default=40)
        self.parameters.declare('t_step_round')

    def define(self):
        # load options
        num_nodes = self.parameters['num_nodes']
        delta_t = self.parameters['delta_t']
        t_step_round = self.parameters['t_step_round']

        # loop through lifting surfaces to compute outputs
        input_time = self.declare_variable('input_time', shape=(num_nodes, 1))
        input_power = self.declare_variable('input_power',
                                            shape=(num_nodes, 1))
        power = self.create_output('power', shape=(t_step_round[-1], 1))
        delta_step = np.diff(np.array(t_step_round))

        input_power_flatten = csdl.reshape(input_power, (num_nodes, ))
        # print(power[t_step_round[4] + 1:t_step_round[5] + 1, :])
        # print(csdl.expand(input_power_flatten[5], (delta_step[4], 1), 'i->ji'))

        power[0:t_step_round[0] + 1, :] = csdl.expand(input_power_flatten[0],
                                                      (t_step_round[0] + 1, 1),
                                                      'i->ji')
        power[t_step_round[0] + 1:t_step_round[1] + 1, :] = csdl.expand(
            input_power_flatten[1], (delta_step[0], 1), 'i->ji')
        power[t_step_round[1] + 1:t_step_round[2] + 1, :] = csdl.expand(
            input_power_flatten[2], (delta_step[1], 1), 'i->ji')
        power[t_step_round[2] + 1:t_step_round[3] + 1, :] = csdl.expand(
            input_power_flatten[3], (delta_step[2], 1), 'i->ji')
        power[t_step_round[3] + 1:t_step_round[4] + 1, :] = csdl.expand(
            input_power_flatten[4], (delta_step[3], 1), 'i->ji')
        power[t_step_round[4] + 1:, :] = csdl.expand(input_power_flatten[5],
                                                     (delta_step[4] - 1, 1),
                                                     'i->ji')
        # for i in range(t_step_round.max()):
        #     if i <= t_step_round[0]:
        #         power[i] = input_power[0]
        #     elif i > t_step_round[0] and i <= t_step_round[1]:
        #         power[i] = input_power[1]
        #     elif i > t_step_round[1] and i <= t_step_round[2]:
        #         power[i] = input_power[2]
        #     elif i > t_step_round[2] and i <= t_step_round[3]:
        #         power[i] = input_power[3]
        #     elif i > t_step_round[3] and i <= t_step_round[4]:
        #         power[i] = input_power[4]
        #     elif i > t_step_round[4] and i <= t_step_round[5]:
        #         power[i] = input_power[5]


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
t_step_round = [int(round(num)) for num in t_step_process]
num_nodes = 6

model_1 = csdl.Model()

submodel = ECMPreprocessingModel(num_nodes=num_nodes,
                                 delta_t=delta_t,
                                 t_step_round=t_step_round)

model_1.add(submodel, 'ECMPreprocessingModel')

# time = np.arange(0,
#                  round(t_total / delta_t + 1) * (delta_t),
#                  delta_t)  # shape=258
# power = np.zeros(time.size)
# for i in range(time.size):
#     if i <= t_step_round[0]:
#         power[i] = p_list[0]
#     elif i > t_step_round[0] and i <= t_step_round[1]:
#         power[i] = p_list[1]
#     elif i > t_step_round[1] and i <= t_step_round[2]:
#         power[i] = p_list[2]
#     elif i > t_step_round[2] and i <= t_step_round[3]:
#         power[i] = p_list[3]
#     elif i > t_step_round[3] and i <= t_step_round[4]:
#         power[i] = p_list[4]
#     elif i > t_step_round[4] and i <= t_step_round[5]:
#         power[i] = p_list[5]

# import numpy as np
# import openmdao.api as om

# xcp = np.cumsum(t_list)
# ycp = np.array(p_list)
# n = 400
# x = np.linspace(0.0, 2586.0, n)

# prob = om.Problem()

# akima_option = {'delta_x': 1}
# comp = om.SplineComp(method='akima',
#                      x_cp_val=xcp,
#                      x_interp_val=x,
#                      interp_options=akima_option)

# prob.model.add_subsystem('akima1', comp)

# comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=ycp)

# prob.setup(force_alloc_complex=True)
# prob.run_model()

# print(prob.get_val('akima1.y_val'))

# import matplotlib.pyplot as plt

# plt.plot(x, prob.get_val('akima1.y_val').flatten())
# plt.plot(xcp, ycp, '.')
# plt.plot(time, power)
# plt.show()
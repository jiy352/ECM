
from unicodedata import name
import csdl
from ozone.api import NativeSystem
import numpy as np

import scipy.sparse as sp
from lsdo_ecm.ecm_data_21700 import T_bp, SOC_bp, tU_oc, tC_Th, tR_Th, tR_0

from smt.surrogate_models import RMTB

from csdl_om import Simulator
from csdl import Model
import csdl
import csdl_om
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
        self.parameters.declare('num_segments')
        self.parameters.declare('t_end')
        self.parameters.declare('num_times')

    def define(self):
        # load options
        num_segments = self.parameters['num_segments']
        t_end = self.parameters['t_end']
        num_times = self.parameters['num_times']

        # loop through lifting surfaces to compute outputs
        input_time = self.declare_variable('input_time', shape=(num_segments, 1))
        input_power = self.declare_variable('input_power',
                                            shape=(num_segments, 1))
        t_vec = self.create_output('t_vec', val=np.zeros((num_segments+1, 1)))

        t_vec[1:,:] = input_time

        t_cs = self.create_output('t_cs', val=np.zeros((num_segments+1, 1)))
        for i in range(num_segments+1):
            print(i)
            t_cs[i,:] = csdl.reshape(csdl.sum(t_vec[:i+1,:],axes=(0,)),(1,1))
        # self.print_var(t_cs)
        alpha_ = 1
        y = self.create_output('y',shape=(num_segments,num_times))
        t = self.create_input('t',val=np.linspace(0,t_end,num_times).reshape(1,num_times))


        for i in range(num_segments):

            t_cs_i_exp = csdl.expand(csdl.reshape(t_cs[i,:],(1,)),(1,num_times),'i->ji')
            t_cs_i_p1_exp = csdl.expand(csdl.reshape(t_cs[i+1,:],(1,)),(1,num_times),'i->ji')
            input_power_exp = csdl.expand(csdl.reshape(input_power[i,:],(1,)),(1,num_times),'i->ji')

            y[i,:] = input_power_exp*(0.5*csdl.tanh(alpha_*(t-t_cs_i_exp))-\
                            0.5*csdl.tanh(alpha_*(t-t_cs_i_p1_exp)))
        self.register_output('power_profile',csdl.reshape(csdl.sum(y,axes=(0,)),(num_times,1)))



if __name__ == "__main__":

    ########################################################################################################
    # 1. load aircraft time profile
    t_taxi = 10
    t_takeoff = 40
    t_climb = 4.4 * 60
    t_cruise = 16.7 * 60
    t_landing = 70
    t_cruise_res = 20 * 60
    t_list = [
        t_taxi, t_takeoff, t_climb, t_cruise, t_landing, t_cruise_res
    ]
    # 2. load aircraft power profile (kW)
    P_taxi = 46.8
    p_takeoff = 829.2
    P_climb = 524.1
    P_cruise = 282.0
    P_landing = 829.2
    P_cruise_res = 282.0
    p_list = [P_taxi, p_takeoff, P_climb, P_cruise, P_landing, P_cruise_res]

    # 3. setup delat_t, and define each process in the discretized time step
    t_end = 2600

    num_segments = 6
    num_times  =100
    model_1 = csdl.Model()
    input_time = model_1.create_input('input_time',np.array(t_list).reshape(num_segments,1))
    input_power = model_1.create_input('input_power',np.array(p_list).reshape(num_segments,1))
    submodel = ECMPreprocessingModel(num_segments=num_segments,
                                    t_end=t_end,
                                    num_times=num_times)

    model_1.add(submodel, 'ECMPreprocessingModel')

    sim = csdl_om.Simulator(model_1,
                                mode='rev')
    # sim.visualize_implementation()
    sim.run()

    import matplotlib.pyplot as plt
    # plt.plot(t_acs_step,p_step, label='original mission profile')
    plt.plot(np.linspace(0,2600,num_times),sim['power_profile'].flatten(), label='tanh approximation with alpha=1')
    # plt.plot(t,z, label='tanh approximation with alpha=10')
    plt.legend(['tanh approximation with alpha=1'])
    plt.show()
    sim.prob.check_config(checks=['unconnected_inputs'])
    a= sim.prob.check_partials(compact_print=True)
from lsdo_ecm.ecm_model import RunModel
import numpy as np
import csdl_lite
import csdl_om
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
delta_t = 20

t_list = [t_taxi, t_takeoff, t_climb, t_cruise, t_landing, t_cruise_res]
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

num_cells = 1
num_times = time.size
# Simulator Object: Note we are passing in a parameter that can be used in the ode system
# sim = csdl_om.Simulator(RunModel(num_times=num_times, num_cells=num_cells),
#                         mode='rev')
sim = csdl_lite.Simulator(RunModel(num_times=num_times, num_cells=num_cells),
                          mode='rev')
# sim.visualize_implementation()
sim.run()
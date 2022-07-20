import csdl
from ozone.api import NativeSystem
import numpy as np

import scipy.sparse as sp
from lsdo_ecm.ecm_data_21700 import T_bp, SOC_bp, tU_oc, tC_Th, tR_Th, tR_0

from smt.surrogate_models import RMTB
"""
This script contains 3 possible ways on defining the same ODE function dydt = f(y) to use for the integrator
1. CSDL model
2. NativeSystem with dense partials
3. NativeSystem with sparse partials

We can easily swap out these three different methods by setting
self.ode_system = 'ode system model' in the ODEProblem class
"""

# ------------------------- METHOD 2: NATIVESYSTEM -------------------------
# ODE Model with Native System:
# Need to define partials unlike csdl but better performance


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # NativeSystem does not require an initialization to access parameters
        n = self.num_nodes
        n_pack = self.parameters['n_pack']
        # Need to have ODE shapes similar as first example
        self.n_pack = n_pack
        self.Q_max = 5

        ###############
        # add inputs: states and parameters
        ###############
        self.add_input('SoC', shape=(n, self.n_pack))
        self.add_input('U_Th', shape=(n, self.n_pack))
        self.add_input('T_cell', shape=(n, self.n_pack))

        self.add_input('power_profile', shape=(n,1))
        self.add_input('n_parallel', shape=(n,1))
        ###############
        # add outputs
        ###############
        self.add_output('dSoC_dt', shape=(n, self.n_pack))
        self.add_output('dU_Th_dt', shape=(n, self.n_pack))
        self.add_output('dT_cell_dt', shape=(n, self.n_pack))

        # self.declare_partials(of='*', wrt='*')
        self.declare_partial_properties(of='*', wrt='*', empty=True)
        # self.declare_partial_properties('*',
        #                                 '*',
        #                                 complex_step_directional=True)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO
    def compute(self, inputs, outputs):
        n = self.num_nodes
        power_profile = inputs['power_profile']
        n_pack = self.parameters['n_pack']
        # print('n_pack',n_pack)
        self.n_pack = n_pack

        self.k = 10
        self.A = 1
        self.T_pack = 20
        self.m_cell = 48e-3 * 2
        self.c_cell = 0.83 * 10000
        # initiate the states to be zeros
        outputs['dSoC_dt'] = np.zeros((n, self.n_pack))
        outputs['dU_Th_dt'] = np.zeros((n, self.n_pack))
        outputs['dT_cell_dt'] = np.zeros((n, self.n_pack))

        # We have accessed a parameter passed in through the ODEproblem
        # !TODO:! how to treat dynamic parameter
        n_s = 190
        n_p = int(16150 / n_s)
        P_batt_i = power_profile / (n_s * n_p)*1000
        # outputs['dSoC_dt'] = np.zeros((n, self.n_pack))
        # outputs['dU_Th_dt'] = np.zeros((n, self.n_pack))
        # outputs['dT_cell_dt'] = np.zeros((n, self.n_pack))

        ######################
        # compute the outputs
        ######################
        # loop over stages
        for i in range(n):
            # rename the states (and parameters) at the current stage
            U_Th = inputs['U_Th'][i]
            T_cell = inputs['T_cell'][i]
            SoC = inputs['SoC'][i]

            
            # compute battery parameters
            R_Th, _, _ = self._PolarizationResistance(SoC, T_cell)
            C_Th, _ = self._EquivCapacitance(SoC)
            U_OC, _, _ = self._OCV(SoC, T_cell)
            R_0, _, _ = self._InternalResistance(SoC, T_cell)

            # compute I_L as the solution of a quadratic equation
            I_L = self._I_L_minus(U_OC, U_Th, R_0, P_batt_i)

            # compute the outputs
            # three states for the cells
            print('dSoC_dt',-I_L / self.Q_max / 3600)
            outputs['dSoC_dt'][i] = -I_L / self.Q_max / 3600
            outputs['dU_Th_dt'][i] = (I_L - U_Th / R_Th) / C_Th
            q_val = -self.k * self.A * (T_cell - self.T_pack)
            outputs['dT_cell_dt'][i] = (I_L**2 * (R_Th + R_0) + q_val) / (
                self.m_cell * self.c_cell) / 1e3
            # print('dT_cell_dt', outputs['dT_cell_dt'][i])

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        power_profile = inputs['power_profile']
        ############################
        # empty lists to store the
        # value of the derivatives
        ############################
        dSoC_dU_Th = []
        dSoC_dSoC = []
        dSoC_dT_cell = []

        dU_Th_dU_Th = []
        dU_Th_dSoC = []
        dU_Th_dT_cell = []

        dT_cell_dT_cell = []
        dT_cell_dSoC = []
        dT_cell_dU_Th = []

        # the power output required
        n_s = 190
        n_p = int(16150 / n_s)
        P_batt_i = power_profile / (n_s * n_p)*1000

        # loop over stages
        # The partials to compute.
        for i in range(n):

            U_Th = inputs['U_Th'][i]
            T_cell = inputs['T_cell'][i]
            SoC = inputs['SoC'][i]

            # compute battery parameters
            R_Th, dR_Th_dSOC, dR_Th_dT_cell = self._PolarizationResistance(
                SoC, T_cell)
            C_Th, dC_1_dSoC = self._EquivCapacitance(SoC)
            U_OC, dU_OC_dSoc_diag, dU_OC_dT_cell_diag = self._OCV(SoC, T_cell)
            R_0, dR0_dSOC, dR0_dT = self._InternalResistance(SoC, T_cell)

            # compute I_L as the solution of a quadratic equation
            I_L = self._I_L_minus(U_OC, U_Th, R_0, P_batt_i)

            PI_LpUTh = ((-(-U_OC + U_Th) / np.sqrt(-4 * P_batt_i * R_0 +
                                                   (U_OC - U_Th)**2) - 1) /
                        (2 * R_0))
            PI_LpUOC = ((-(U_OC - U_Th) / np.sqrt(-4 * P_batt_i * R_0 +
                                                  (U_OC - U_Th)**2) + 1) /
                        (2 * R_0))
            PI_LpR0 = (P_batt_i / (R_0 * np.sqrt(-4 * P_batt_i * R_0 +
                                                 (U_OC - U_Th)**2)) -
                       (U_OC - U_Th - np.sqrt(-4 * P_batt_i * R_0 +
                                              (U_OC - U_Th)**2)) /
                       (2 * R_0**2))
            PI_LpSoC = (PI_LpUOC * dU_OC_dSoc_diag + PI_LpR0 * dR0_dSOC)
            PI_LpTcell = (PI_LpUOC * dU_OC_dT_cell_diag + PI_LpR0 * dR0_dT)

            ####################################
            # compute derivative for each stage
            ####################################
            # partials for SoC_{dot}
            dSoC_dt_dU_Th_i = (-1 / self.Q_max / 3600) * PI_LpUTh
            dSoC_dt_dSoC_i = (-1 / self.Q_max / 3600) * PI_LpSoC
            dSoC_dt_dT_cell_i = (-1 / self.Q_max / 3600) * PI_LpTcell

            p1_pI_L = (1 / C_Th)
            p1_pC_Th = (-I_L / (C_Th**2))
            p2_pR_Th = (-(U_Th / C_Th) / (R_Th**2))
            p2_pC_Th = (-(U_Th / R_Th) / (C_Th**2))

            # partials for UTh_{dot}
            dU_Th_dU_Th_i = p1_pI_L * PI_LpUTh - 1 / (R_Th * C_Th)
            dU_Th_dSoC_i = p1_pI_L * PI_LpSoC + p1_pC_Th * dC_1_dSoC - (
                p2_pR_Th * dR_Th_dSOC + p2_pC_Th * dC_1_dSoC)
            dU_Th_dT_cell_i = p1_pI_L * PI_LpTcell - (p2_pR_Th * dR_Th_dT_cell)

            # partials for Tcell_{dot}
            dT_cell_dU_Th_i = 1 / (self.m_cell * self.c_cell) * 2 * I_L * (
                R_Th + R_0) * PI_LpUTh
            dT_cell_dSoC_i = 1 / (self.m_cell * self.c_cell) * (
                2 * I_L * (R_Th + R_0) * PI_LpSoC + I_L**2 * dR0_dSOC +
                I_L**2 * dR_Th_dSOC)

            deri_q_val_T_cell_i = -self.k * self.A
            dT_cell_dT_cell_i = 1 / (self.m_cell * self.c_cell) * (
                (2 * I_L * (R_Th + R_0) * PI_LpTcell + I_L**2 * dR0_dT +
                 I_L**2 * dR_Th_dT_cell) + deri_q_val_T_cell_i) / 1e3

            # change format to diagnal flat
            dSoC_dU_Th_i = np.diagflat(dSoC_dt_dU_Th_i)
            dSoC_dSoC_i = np.diagflat(dSoC_dt_dSoC_i)
            dSoC_dT_cell_i = np.diagflat(dSoC_dt_dT_cell_i)

            dU_Th_dU_Th_i = np.diagflat(dU_Th_dU_Th_i)
            dU_Th_dSoC_i = np.diagflat(dU_Th_dSoC_i)
            dU_Th_dT_cell_i = np.diagflat(dU_Th_dT_cell_i)

            dT_cell_dT_cell_i = np.diagflat(dT_cell_dT_cell_i)
            dT_cell_dSoC_i = np.diagflat(dT_cell_dSoC_i)
            dT_cell_dU_Th_i = np.diagflat(dT_cell_dU_Th_i)

            # append to get the full list
            dSoC_dU_Th.append(sp.lil_matrix(dSoC_dU_Th_i))
            dSoC_dSoC.append(sp.lil_matrix(dSoC_dSoC_i))
            dSoC_dT_cell.append(sp.lil_matrix(dSoC_dT_cell_i))

            dU_Th_dU_Th.append(sp.lil_matrix(dU_Th_dU_Th_i))
            dU_Th_dSoC.append(sp.lil_matrix(dU_Th_dSoC_i))
            dU_Th_dT_cell.append(sp.lil_matrix(dU_Th_dT_cell_i))

            dT_cell_dT_cell.append(sp.lil_matrix(dT_cell_dT_cell_i))
            dT_cell_dSoC.append(sp.lil_matrix(dT_cell_dSoC_i))
            dT_cell_dU_Th.append(sp.lil_matrix(dT_cell_dU_Th_i))

        partials['dSoC_dt']['U_Th'] = sp.block_diag(dSoC_dU_Th, format='csc')
        partials['dSoC_dt']['SoC'] = sp.block_diag(dSoC_dSoC, format='csc')
        partials['dSoC_dt']['T_cell'] = sp.block_diag(dSoC_dT_cell,
                                                      format='csc')

        partials['dU_Th_dt']['U_Th'] = sp.block_diag(dU_Th_dU_Th, format='csc')
        partials['dU_Th_dt']['SoC'] = sp.block_diag(dU_Th_dSoC, format='csc')
        partials['dU_Th_dt']['T_cell'] = sp.block_diag(dU_Th_dT_cell,
                                                       format='csc')

        partials['dT_cell_dt']['T_cell'] = sp.block_diag(dT_cell_dT_cell,
                                                         format='csc')
        partials['dT_cell_dt']['SoC'] = sp.block_diag(dT_cell_dSoC,
                                                      format='csc')
        partials['dT_cell_dt']['U_Th'] = sp.block_diag(dT_cell_dU_Th,
                                                       format='csc')

        # partials['dU_Th_dt']['U_Th'] = sp.block_diag(dU_Th_dU_Th, format='csc')
        # partials['dU_Th_dt']['SoC'] = sp.block_diag(dU_Th_dSoC, format='csc')

        # partials['dT_cell_dt']['T_cell'] = sp.block_diag(dT_cell_dT_cell,
        #                                                  format='csc')
        # partials['dT_cell_dt']['SoC'] = sp.block_diag(dT_cell_dSoC,
        #                                               format='csc')
        # partials['dT_cell_dt']['U_Th'] = sp.block_diag(dT_cell_dU_Th,
        #    format='csc')

        # The structure of partials has the following for n = self.num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal

    #############################
    # functions for fitting the
    # battery parameters
    #############################

    def _OCV(self, SoC, T_cell):
        '''compute ocv and d_ocv_dsoc, and d_ocv_dT_cell (f1)'''
        U_OC, dU_OC_dT_cell_diag, dU_OC_dSoc_diag = self._predict_eve(
            tU_oc, T_cell, SoC, order=4, num_ctrl_pts=4)
        # print('T_cell',T_cell.shape)
        # print('dU_OC_dT_cell_diag',dU_OC_dT_cell_diag.shape)
        # print('dU_OC_dSoc_diag',dU_OC_dSoc_diag.shape)
        return U_OC, dU_OC_dSoc_diag, dU_OC_dT_cell_diag

    def _InternalResistance(self, SoC, T_cell):
        '''compute R_0 and d_ocv_dsoc, and dR0_dT (f2) SoC'''

        R_0, dR_Th_dT_cell, dR_Th_dSOC = self._predict_eve(tR_0,
                                                           T_cell,
                                                           SoC,
                                                           order=3,
                                                           num_ctrl_pts=4)

        return R_0, dR_Th_dSOC, dR_Th_dT_cell

    # '''compute R_0 and d_ocv_dsoc, and dR0_dT (f2)'''

    # sm = smt_internal_resistance()
    # x = np.concatenate((temperature, SoC)).reshape(-1, 2, order='F')
    # R_0 = sm.predict_values(x)
    # # print(R_0)

    # dR0_dT = sm.predict_derivatives(x, 0).flatten()
    # dR0_dSOC = sm.predict_derivatives(x, 1).flatten()
    # return R_0, dR0_dSOC, dR0_dT

    def _PolarizationResistance(self, SoC, T_cell):
        '''compute R_th and d_ocv_dsoc, and d_ocv_dT_cell (f3)'''
        # print('SoC---------------', SoC)
        # print('exp---------------', np.exp(-120 * SoC))

        R_Th, dR_Th_dT_cell, dR_Th_dSOC = self._predict_eve(
            tR_Th,
            T_cell,
            SoC,
            order=3,
            num_ctrl_pts=5,
            regularization_weight=5e-6)

        return R_Th, dR_Th_dSOC, dR_Th_dT_cell

    def _EquivCapacitance(self, SoC):
        '''compute C_th and dC_1_dSoC (f4)'''
        C_1 = -23.6 * SoC**4 - 24.6 * SoC**3 - 5900 * SoC**2 + 7240 * SoC + 401
        dC_1_dSoC = -94.4 * SoC**3 - 73.8 * SoC**2 - 11800 * SoC + 7240
        return C_1, dC_1_dSoC

    def _I_L_minus(self, U_oc, U_Th, R_0, P_batt_i):
        if (U_oc - U_Th)**2 - 4 * R_0 * P_batt_i > 0:
            I_L_minus = ((U_oc - U_Th) - np.sqrt(
                (U_oc - U_Th)**2 - 4 * R_0 * P_batt_i)) / (2 * R_0)
        else:
            print('no solution!!!!!!!!!!!!!!')
            print('(U_oc - U_Th)**2!!!!!!!!!!!!!!', (U_oc - U_Th)**2)
            print('R_0!!!!!!!!!!!!!!', R_0)
            print('P_batt_i!!!!!!!!!!!!!!', P_batt_i)
            print('4 * R_0 * P_batt_i !!!!!!!!!!!!!!', 4 * R_0 * P_batt_i)
            I_L_minus = ((U_oc - U_Th)) / (2 * R_0)
        return I_L_minus

        print("I_L_minus", I_L_minus)
        print("U_oc", U_oc)
        print("U_Th", U_Th)
        print("R_0", R_0)
        print("P_batt_i", P_batt_i)
        return I_L_minus

    def _predict_eve(self,
                     y,
                     xt_eve,
                     xs_eve,
                     order,
                     num_ctrl_pts,
                     regularization_weight=1e-6):
        '''
        training set inputs (T; SoC) are automatically loaded from the x-57 paper
        y: training set outputs
        xt_eve: evaluation set for temperature
        xs_eve: evaluation set for soc
        '''
        x_T = np.tile(T_bp, SOC_bp.shape[1]).reshape(-1, 4).T.flatten()
        x_S0C = SOC_bp.T.flatten()
        # print('x_T', x_T.shape)
        # print('x_S0C', x_S0C.shape)
        x = np.concatenate((x_T, x_S0C)).reshape(-1, 2, order='F')
        xlimits = np.array([[-50, 60.0], [0., 1.000000001]])
        sm = RMTB(
            xlimits=xlimits,
            order=order,
            num_ctrl_pts=num_ctrl_pts,
            energy_weight=1e-15,
            regularization_weight=regularization_weight,
            print_global=False,
        )
        # print('x', x.shape)
        # print('y', y.shape)
        sm.set_training_values(x, y.flatten())
        sm.train()

        if xs_eve.size != 1:
            x_eve = np.concatenate((xt_eve, xs_eve)).reshape(-1, 2, order='F')
        else:
            x_eve = np.array([xt_eve, xs_eve]).reshape(-1, 2)
        # print()
        # print('x_eve', x_eve)
        y_predict = sm.predict_values(x_eve)

        dy_dt = sm.predict_derivatives(x_eve, 0).flatten()
        dy_dsoc = sm.predict_derivatives(x_eve, 1).flatten()
        # print('dy_dt', dy_dt)
        # print('dy_dsoc', dy_dsoc)
        return y_predict.flatten(), dy_dt.flatten(), dy_dsoc.flatten()
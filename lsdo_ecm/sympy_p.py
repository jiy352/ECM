from sympy import diff, sin, exp,tanh,sqrt
from sympy.abc import beta,eta,x
from sympy import *

U_oc = symbols('U_oc')
U_Th = symbols('U_Th')
R_0 = symbols('R_0')
p_pack = symbols('p_pack')
n_p = symbols('n_p')
n_s = symbols('n_s')

P_batt_i = p_pack/(n_s*n_p)*1000
I_L_minus = ((U_oc - U_Th) - (
                (U_oc - U_Th)**2 - 4 * R_0 * P_batt_i)**0.5) / (2 * R_0)

PUOC = diff(I_L_minus,U_oc)
PR0 = diff(I_L_minus,R_0)
PU_Th = diff(I_L_minus,U_Th)

Pp_pack = diff(I_L_minus,p_pack)
Pn_p= diff(I_L_minus,n_p)

#####################################
T_cell = symbols('T_cell')
T_pack = symbols('T_pack')
k = symbols('k')
A = symbols('A')
R_Th = symbols('R_Th')
I_L = symbols('I_L')
m_cell = symbols('m_cell')
c_cell = symbols('c_cell')

q_val = -k * A * (T_cell - T_pack)
T_cell_dot = (I_L_minus**2 * (R_Th + R_0) + q_val) / (
                m_cell * c_cell) 
PT_Pnp = diff(T_cell_dot,n_p)
PT_Ppower = diff(T_cell_dot,p_pack)

#####################################
C_Th = symbols('C_Th')

dU_Th_dt = (I_L - U_Th / R_Th) / C_Th
PU_Th_PIL = diff(dU_Th_dt,I_L)
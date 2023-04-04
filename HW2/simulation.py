import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
import math
import time 
from scipy.stats import norm


# define demand shock transition matrix
x_transition = np.array([[0.6, 0.2, 0.2],
                         [0.2, 0.6, 0.2],
                         [0.2, 0.2, 0.6]])

# define cournot pi function
def cournot_pi(N_t, x_idx_t, fc):
    x_t = [-5, 0, 5][x_idx_t]
    pi = ((10 + x_t)/(N_t + 1))**2 - fc
    return pi


def N_trans_incumbent(entry_TH, exit_TH, demand_shock_idx, theta_distribution):
    # calculate transition of N, conditional on remaining decision (incumbent's persepctive)
    # function name: N_trans_incumbent
    # function input: entry_TH, exit_TH, demand_shock_idx
    # idx=0 -> x=-5
    # idx=1 -> x=0
    # idx=2 -> x=5
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    
    #(1) Calculate First Row
    entry_prob_N1 = norm.cdf(entry_TH[1][demand_shock_idx], loc=gamma_loc, scale=gamma_scale)   # entry_TH[1]: row for N_t = 1
    element_11 = 1-entry_prob_N1
    element_12 = entry_prob_N1
    row_1 = [element_11, element_12, 0, 0, 0]

    #(2) Calculate Second Row
    entry_prob_N2 = norm.cdf(entry_TH[2][demand_shock_idx], loc=gamma_loc, scale=gamma_scale)   # entry_TH[2]: row for N_t = 2
    exit_prob_N2 = 1-norm.cdf(exit_TH[1][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[1]: row for N_t = 2 
    element_21 = (1-entry_prob_N2)*exit_prob_N2                                     # no entry, one exit
    element_22 = (1-entry_prob_N2)*(1-exit_prob_N2) + entry_prob_N2*exit_prob_N2    # no entry, no exit & one entry, one exit
    element_23 = entry_prob_N2 * (1-exit_prob_N2)                                   # one entry, no exit
    row_2 = [element_21, element_22, element_23, 0, 0]

    #(3) Calculate Third Row
    entry_prob_N3 = norm.cdf(entry_TH[3][demand_shock_idx], loc=gamma_loc, scale=gamma_scale)   # entry_TH[3]: row for N_t = 3
    exit_prob_N3 = 1-norm.cdf(exit_TH[2][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[2]: row for N_t = 3
    element_31 = (1-entry_prob_N3)*math.comb(2,2)*(exit_prob_N3**2)                 # no entry, two exit
    element_32 = (1-entry_prob_N3)*math.comb(2,1)*(exit_prob_N3)*(1-exit_prob_N3)+ \
                (entry_prob_N3)*math.comb(2,2)*(exit_prob_N3**2)                    # no entry, one exit & one entry, two exit
    element_33 = (1-entry_prob_N3)*math.comb(2,2)*((1-exit_prob_N3)**2)+ \
                (entry_prob_N3)*math.comb(2,1)*(exit_prob_N3)*(1-exit_prob_N3)      # no entry, no exit & one entry, one exit
    element_34 = (entry_prob_N3)*math.comb(2,2)*((1-exit_prob_N3)**2)               # one entry, no exit
    row_3 = [element_31, element_32, element_33, element_34, 0]

    #(4) Calculate Fourth Row
    entry_prob_N4 = norm.cdf(entry_TH[4][demand_shock_idx], loc=gamma_loc, scale=gamma_scale)   # entry_TH[4]: row for N_t = 4
    exit_prob_N4 = 1-norm.cdf(exit_TH[3][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[3]: row for N_t = 4
    element_41 = (1-entry_prob_N4)*math.comb(3,3)*(exit_prob_N4**3)                 # no entry, three exit
    element_42 = (1-entry_prob_N4)*math.comb(3,2)*(exit_prob_N4**2)*(1-exit_prob_N4)+ \
                (entry_prob_N4)*math.comb(3,3)*(exit_prob_N4**3)                    # no entry, two exit & one entry, three exit
    element_43 = (1-entry_prob_N4)*math.comb(3,1)*(exit_prob_N4)*((1-exit_prob_N4)**2)+ \
                (entry_prob_N4)*math.comb(3,2)*(exit_prob_N4**2)*(1-exit_prob_N4)   # no entry, one exit & one entry, two exit
    element_44 = (1-entry_prob_N4)*math.comb(3,0)*((1-exit_prob_N4)**3)+ \
                (entry_prob_N4)*math.comb(3,1)*(exit_prob_N4)*((1-exit_prob_N4)**2) # no entry, no exit & one entry, one exit
    element_45 = (entry_prob_N4)*math.comb(3,0)*((1-exit_prob_N4)**3)               # one entry, no exit
    row_4 = [element_41, element_42, element_43, element_44, element_45]

    #(5) Calculate Fifth Row
    exit_prob_N5 = 1-norm.cdf(exit_TH[4][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[4]: row for N_t = 5
    element_51 = math.comb(4,4)*(exit_prob_N5**4)
    element_52 = math.comb(4,3)*(exit_prob_N5**3)*(1-exit_prob_N5)
    element_53 = math.comb(4,2)*(exit_prob_N5**2)*((1-exit_prob_N5)**2)
    element_54 = math.comb(4,1)*(exit_prob_N5)*((1-exit_prob_N5)**3)
    element_55 = math.comb(4,0)*((1-exit_prob_N5)**4)
    row_5 = [element_51, element_52, element_53, element_54, element_55]

    return np.array([row_1, row_2, row_3, row_4, row_5])



def N_trans_entrant(exit_TH, demand_shock_idx, theta_distribution):
    # calculate transition of N, conditional on entry decision (entrant's persepctive)
    # function name: N_trans_entrant
    # function input: exit_TH, demand_shock_idx
    # idx=0 -> x=-5
    # idx=1 -> x=0
    # idx=2 -> x=5
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    
    #(1) Calculate First Row
    row_1 = [1,0,0,0,0]

    #(2) Calculate Second Row
    exit_prob_N1 = 1-norm.cdf(exit_TH[0][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[0]: row for N_t = 1 
    element_21 = exit_prob_N1
    element_22 = 1-exit_prob_N1
    row_2 = [element_21, element_22, 0, 0, 0]

    #(3) Calculate Third Row
    exit_prob_N2 = 1-norm.cdf(exit_TH[1][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[1]: row for N_t = 2 
    element_31 = math.comb(2,2)*(exit_prob_N2**2)
    element_32 = math.comb(2,1)*(exit_prob_N2)*(1-exit_prob_N2)
    element_33 = math.comb(2,0)*((1-exit_prob_N2)**2)
    row_3 = [element_31, element_32, element_33, 0, 0]

    #(4) Calculate Fourth Row
    exit_prob_N3 = 1-norm.cdf(exit_TH[2][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[2]: row for N_t = 3
    element_41 = math.comb(3,3)*(exit_prob_N3**3)
    element_42 = math.comb(3,2)*(exit_prob_N3**2)*(1-exit_prob_N3)
    element_43 = math.comb(3,1)*(exit_prob_N3**1)*((1-exit_prob_N3)**2)
    element_44 = math.comb(3,0)*((1-exit_prob_N3)**3)
    row_4 = [element_41, element_42, element_43, element_44, 0]

    #(5) Calculate Fifth Row
    exit_prob_N4 = 1-norm.cdf(exit_TH[3][demand_shock_idx], loc=mu_loc, scale=mu_scale)         # exit_TH[3]: row for N_t = 4
    element_51 = math.comb(4,4)*(exit_prob_N4**4)
    element_52 = math.comb(4,3)*(exit_prob_N4**3)*(1-exit_prob_N4)
    element_53 = math.comb(4,2)*(exit_prob_N4**2)*((1-exit_prob_N4)**2)
    element_54 = math.comb(4,1)*(exit_prob_N4)*((1-exit_prob_N4)**3)
    element_55 = math.comb(4,0)*((1-exit_prob_N4)**4)
    row_5 = [element_51, element_52, element_53, element_54, element_55]

    return np.array([row_1, row_2, row_3, row_4, row_5])



def Psi_1(N_t, x_idx_t, entry_TH, exit_TH, V_bar, theta_distribution): # N_t: 1,2,3,4,5 
    # calculate Psi_1 conditional on current market structure and demand shock. Expected future value of remaining in the market (incumbent's perspective)
    # function name: Psi_1
    # function input: N_t, demand_shock_idx
    # idx=0 -> x=-5
    # idx=1 -> x=0
    # idx=2 -> x=5
    N_transition = N_trans_incumbent(entry_TH, exit_TH, x_idx_t, theta_distribution)

    def joint_trans_prob(N_t_prime, x_idx_t_prime): # N_t_prime: 1,2,3,4,5
        x_trans_prob = x_transition[x_idx_t][x_idx_t_prime]
        N_trans_prob = N_transition[N_t-1][N_t_prime-1]
        return x_trans_prob*N_trans_prob

    N_t_prime_list = [1,2,3,4,5]
    x_idx_t_prime_list = [0,1,2]
    joint_trans_matrix = np.ones((5,3))

    for N_t_prime in N_t_prime_list:
        for x_idx_t_prime in x_idx_t_prime_list:
            joint_trans_matrix[N_t_prime-1][x_idx_t_prime] = joint_trans_prob(N_t_prime, x_idx_t_prime)

    Psi_1 = 0.9*(V_bar*joint_trans_matrix).sum()

    return Psi_1



def Psi_2(N_t, x_idx_t, exit_TH, V_bar, theta_distribution): # N_t: 0,1,2,3,4
    # calculate Psi_2 conditional on current market structure and demand shock. Expected future value of entering the market (entrant's perspective)
    # function name: Psi_2
    # function input: N_t, demand_shock_idx
    # idx=0 -> x=-5
    # idx=1 -> x=0
    # idx=2 -> x=5
    N_transition = N_trans_entrant(exit_TH, x_idx_t, theta_distribution)

    def joint_trans_prob(N_t_prime, x_idx_t_prime): # N_t_prime: 1,2,3,4,5
        x_trans_prob = x_transition[x_idx_t][x_idx_t_prime]
        N_trans_prob = N_transition[N_t][N_t_prime-1]
        return x_trans_prob*N_trans_prob

    N_t_prime_list = [1,2,3,4,5]
    x_idx_t_prime_list = [0,1,2]
    joint_trans_matrix = np.ones((5,3))

    for N_t_prime in N_t_prime_list:
        for x_idx_t_prime in x_idx_t_prime_list:
            joint_trans_matrix[N_t_prime-1][x_idx_t_prime] = joint_trans_prob(N_t_prime, x_idx_t_prime)

    Psi_2 = 0.9*(V_bar*joint_trans_matrix).sum()

    return Psi_2



def eqm_finder(init_constant, theta_distribution, theta_search, cournot_pi_matrix):
    # define parameters
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    learning_rate, epsilon = theta_search

    # feed the initial pre matrices to the while loop
    entry_TH_pre = np.full((5, 3), init_constant)
    exit_TH_pre = np.full((5, 3), init_constant)
    V_bar_pre = np.full((5, 3), init_constant)

    # search for the equilibrium 
    while True:
        #####################
        # Updating Block
        #####################
        # (1) update mu: exit_TH
        Psi_1_matrix = np.ones((5,3))
        N_t_list = [1,2,3,4,5]
        x_idx_t_list = [0,1,2]
        for N_t in N_t_list:
            for x_idx_t in x_idx_t_list:
                Psi_1_matrix[N_t-1][x_idx_t] = Psi_1(N_t, x_idx_t, entry_TH_pre, exit_TH_pre, V_bar_pre, theta_distribution)
        exit_TH_post = cournot_pi_matrix + Psi_1_matrix

        # (2) update gamma: entry_TH
        Psi_2_matrix = np.ones((5,3))
        N_t_list2 = [0,1,2,3,4]
        for N_t in N_t_list2:
            for x_idx_t in x_idx_t_list:
                Psi_2_matrix[N_t][x_idx_t] = Psi_2(N_t, x_idx_t, exit_TH_pre, V_bar_pre, theta_distribution)
        entry_TH_post = Psi_2_matrix

        # (3) update v_bar
        V_bar_post = np.ones((5,3))
        for N_t in N_t_list:
            for x_idx_t in x_idx_t_list:
                z = (exit_TH_post[N_t-1][x_idx_t] - mu_loc)/mu_scale
                term_1 = 1 - norm.cdf(z, loc=0, scale=1) #standardized
                # exception for infinity case
                term_2_temp = mu_loc + mu_scale*(norm.pdf(z, loc=0, scale=1)/(1 - norm.cdf(z, loc=0, scale=1)))
                if term_2_temp == np.inf:
                    term_2 = 100
                else: 
                    term_2 = term_2_temp
                
                term_3 = norm.cdf(z, loc=0, scale=1)
                term_4 = exit_TH_post[N_t-1][x_idx_t]
                term_V = term_1*term_2 + term_3*term_4
                V_bar_post[N_t-1][x_idx_t] = term_V
        
        ##########################################
        # Break Rule and Continue Condition
        ##########################################
        diff_1 = (abs(exit_TH_post - exit_TH_pre)).max()    
        diff_2 = (abs(entry_TH_post - entry_TH_pre)).max()    
        diff_3 = (abs(V_bar_post - V_bar_pre)).max()
        diff_max = max(diff_1, diff_2, diff_3)

        if diff_max < epsilon:
            break
        else:
            entry_TH_pre = learning_rate*entry_TH_post + (1-learning_rate)*entry_TH_pre
            exit_TH_pre  = learning_rate*exit_TH_post  + (1-learning_rate)*exit_TH_pre
            V_bar_pre    = learning_rate*V_bar_post    + (1-learning_rate)*V_bar_pre

    return entry_TH_post, exit_TH_post, V_bar_post




def eqm_finder_entry_cost(init_constant, theta_distribution, theta_search, cournot_pi_matrix):
    # define parameters
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    learning_rate, epsilon = theta_search

    # feed the initial pre matrices to the while loop
    entry_TH_pre = np.full((5, 3), init_constant)
    exit_TH_pre = np.full((5, 3), init_constant)
    V_bar_pre = np.full((5, 3), init_constant)

    # search for the equilibrium 
    while True:
        #####################
        # Updating Block
        #####################
        # (1) update mu: exit_TH
        Psi_1_matrix = np.ones((5,3))
        N_t_list = [1,2,3,4,5]
        x_idx_t_list = [0,1,2]
        for N_t in N_t_list:
            for x_idx_t in x_idx_t_list:
                Psi_1_matrix[N_t-1][x_idx_t] = Psi_1(N_t, x_idx_t, entry_TH_pre, exit_TH_pre, V_bar_pre, theta_distribution)
        exit_TH_post = cournot_pi_matrix + Psi_1_matrix

        # (2) update gamma: entry_TH
        Psi_2_matrix = np.ones((5,3))
        N_t_list2 = [0,1,2,3,4]
        for N_t in N_t_list2:
            for x_idx_t in x_idx_t_list:
                Psi_2_matrix[N_t][x_idx_t] = Psi_2(N_t, x_idx_t, exit_TH_pre, V_bar_pre, theta_distribution)
        entry_TH_post = Psi_2_matrix - np.full((5, 3), 5)

        # (3) update v_bar
        V_bar_post = np.ones((5,3))
        for N_t in N_t_list:
            for x_idx_t in x_idx_t_list:
                z = (exit_TH_post[N_t-1][x_idx_t] - mu_loc)/mu_scale
                term_1 = 1 - norm.cdf(z, loc=0, scale=1) #standardized
                # exception for infinity case
                term_2_temp = mu_loc + mu_scale*(norm.pdf(z, loc=0, scale=1)/(1 - norm.cdf(z, loc=0, scale=1)))
                if term_2_temp == np.inf:
                    term_2 = 1000
                else: 
                    term_2 = term_2_temp
                
                term_3 = norm.cdf(z, loc=0, scale=1)
                term_4 = exit_TH_post[N_t-1][x_idx_t]
                term_V = term_1*term_2 + term_3*term_4
                V_bar_post[N_t-1][x_idx_t] = term_V
        
        ##########################################
        # Break Rule and Continue Condition
        ##########################################
        diff_1 = (abs(exit_TH_post - exit_TH_pre)).max()    
        diff_2 = (abs(entry_TH_post - entry_TH_pre)).max()    
        diff_3 = (abs(V_bar_post - V_bar_pre)).max()
        diff_max = max(diff_1, diff_2, diff_3)

        if diff_max < epsilon:
            break
        else:
            entry_TH_pre = learning_rate*entry_TH_post + (1-learning_rate)*entry_TH_pre
            exit_TH_pre  = learning_rate*exit_TH_post  + (1-learning_rate)*exit_TH_pre
            V_bar_pre    = learning_rate*V_bar_post    + (1-learning_rate)*V_bar_pre

    return entry_TH_post, exit_TH_post, V_bar_post
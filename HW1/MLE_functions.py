import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
import math
from scipy.stats import norm




def prob_ent2(work_data, delta, mu, sigma):
    alpha, beta = [1,1]
    Z_f1, Z_f2 = list(work_data['Z_fm'])
    X_m = list(work_data['X_m'])[0]
    N_star = list(work_data['N_star'])[0]
    
    def entry_prob(cost_shifter, eqm_number):
        return norm.cdf(beta*X_m - delta*math.log(eqm_number) - alpha*cost_shifter, mu, sigma)
    
    prob_f1_in_eqm1 = entry_prob(Z_f1, 1)
    prob_f2_in_eqm1 = entry_prob(Z_f2, 1)
    prob_f1_in_eqm2 = entry_prob(Z_f1, 2)
    prob_f2_in_eqm2 = entry_prob(Z_f2, 2)
    #prob_f1_in_eqm3 = entry_prob(Z_f1, 3)
    #prob_f2_in_eqm3 = entry_prob(Z_f2, 3)
    
    if N_star == 0:
        H_01 = (1-prob_f1_in_eqm1)*(1-prob_f2_in_eqm1)
        prob = H_01
        
    elif N_star == 1:  
        H_02 = (1-prob_f1_in_eqm2)*(1-prob_f2_in_eqm2)
        H_12 = prob_f1_in_eqm2*(1-prob_f2_in_eqm2) + (1-prob_f1_in_eqm2)*prob_f2_in_eqm2
        H_01 = (1-prob_f1_in_eqm1)*(1-prob_f2_in_eqm1)
        prob = H_02 + H_12 - H_01
    
    elif N_star == 2:
        H_22 = prob_f1_in_eqm2*prob_f2_in_eqm2
        prob = H_22
        
    return prob

def prob_ent3(work_data, delta, mu, sigma):
    alpha, beta = [1,1]
    Z_f1, Z_f2, Z_f3 = list(work_data['Z_fm'])
    X_m = list(work_data['X_m'])[0]
    N_star = list(work_data['N_star'])[0]
    
    def entry_prob(cost_shifter, eqm_number):
        return norm.cdf(beta*X_m - delta*math.log(eqm_number) - alpha*cost_shifter, mu, sigma)
    
    prob_f1_in_eqm1 = entry_prob(Z_f1, 1)
    prob_f2_in_eqm1 = entry_prob(Z_f2, 1)
    prob_f3_in_eqm1 = entry_prob(Z_f3, 1)
    
    prob_f1_in_eqm2 = entry_prob(Z_f1, 2)
    prob_f2_in_eqm2 = entry_prob(Z_f2, 2)
    prob_f3_in_eqm2 = entry_prob(Z_f3, 2)
    
    prob_f1_in_eqm3 = entry_prob(Z_f1, 3)
    prob_f2_in_eqm3 = entry_prob(Z_f2, 3)
    prob_f3_in_eqm3 = entry_prob(Z_f3, 3)
    
    #prob_f1_in_eqm4 = entry_prob(Z_f1, 4)
    #prob_f2_in_eqm4 = entry_prob(Z_f2, 4)
    #prob_f3_in_eqm4 = entry_prob(Z_f3, 4)
    
    H_01 = (1-prob_f1_in_eqm1)*(1-prob_f2_in_eqm1)*(1-prob_f3_in_eqm1)
    H_02 = (1-prob_f1_in_eqm2)*(1-prob_f2_in_eqm2)*(1-prob_f3_in_eqm2)
    H_12 = prob_f1_in_eqm2*(1-prob_f2_in_eqm2)*(1-prob_f3_in_eqm2) \
           + (1-prob_f1_in_eqm2)*prob_f2_in_eqm2*(1-prob_f3_in_eqm2) \
           + (1-prob_f1_in_eqm2)*(1-prob_f2_in_eqm2)*prob_f3_in_eqm2
    H_03 = (1-prob_f1_in_eqm3)*(1-prob_f2_in_eqm3)*(1-prob_f3_in_eqm3)
    H_13 = prob_f1_in_eqm3*(1-prob_f2_in_eqm3)*(1-prob_f3_in_eqm3) \
           + (1-prob_f1_in_eqm3)*prob_f2_in_eqm3*(1-prob_f3_in_eqm3) \
           + (1-prob_f1_in_eqm3)*(1-prob_f2_in_eqm3)*prob_f3_in_eqm3
    H_23 = prob_f1_in_eqm3*(1-prob_f2_in_eqm3)*prob_f3_in_eqm3 \
           + prob_f1_in_eqm3*prob_f2_in_eqm3*(1-prob_f3_in_eqm3) \
           + (1-prob_f1_in_eqm3)*prob_f2_in_eqm3*prob_f3_in_eqm3
    H_33 = prob_f1_in_eqm3*prob_f2_in_eqm3*prob_f3_in_eqm3
    
    
    if N_star == 0:
        prob = H_01   
    
    elif N_star == 1:  
        prob = H_02 + H_12 - H_01 
    
    elif N_star == 2:
        prob = H_03 + H_13 + H_23 - (H_02 + H_12)
    
    elif N_star == 3:
        prob = H_33
    
    return prob

def prob_ent4(work_data, delta, mu, sigma):
    alpha, beta = [1,1]
    Z_f1, Z_f2, Z_f3, Z_f4 = list(work_data['Z_fm'])
    X_m = list(work_data['X_m'])[0]
    N_star = list(work_data['N_star'])[0]
    
    def entry_prob(cost_shifter, eqm_number):
        return norm.cdf(beta*X_m - delta*math.log(eqm_number) - alpha*cost_shifter, mu, sigma)
    
    prob_f1_in_eqm1 = entry_prob(Z_f1, 1)
    prob_f2_in_eqm1 = entry_prob(Z_f2, 1)
    prob_f3_in_eqm1 = entry_prob(Z_f3, 1)
    prob_f4_in_eqm1 = entry_prob(Z_f4, 1)
    
    prob_f1_in_eqm2 = entry_prob(Z_f1, 2)
    prob_f2_in_eqm2 = entry_prob(Z_f2, 2)
    prob_f3_in_eqm2 = entry_prob(Z_f3, 2)
    prob_f4_in_eqm2 = entry_prob(Z_f4, 2)
    
    prob_f1_in_eqm3 = entry_prob(Z_f1, 3)
    prob_f2_in_eqm3 = entry_prob(Z_f2, 3)
    prob_f3_in_eqm3 = entry_prob(Z_f3, 3)
    prob_f4_in_eqm3 = entry_prob(Z_f4, 3)
    
    prob_f1_in_eqm4 = entry_prob(Z_f1, 4)
    prob_f2_in_eqm4 = entry_prob(Z_f2, 4)
    prob_f3_in_eqm4 = entry_prob(Z_f3, 4)
    prob_f4_in_eqm4 = entry_prob(Z_f4, 4)
    
    H_01 = (1-prob_f1_in_eqm1)*(1-prob_f2_in_eqm1)*(1-prob_f3_in_eqm1)*(1-prob_f4_in_eqm1)
    
    H_02 = (1-prob_f1_in_eqm2)*(1-prob_f2_in_eqm2)*(1-prob_f3_in_eqm2)*(1-prob_f4_in_eqm2)
    
    H_12 = prob_f1_in_eqm2*(1-prob_f2_in_eqm2)*(1-prob_f3_in_eqm2)*(1-prob_f4_in_eqm2) \
           + (1-prob_f1_in_eqm2)*prob_f2_in_eqm2*(1-prob_f3_in_eqm2)*(1-prob_f4_in_eqm2) \
           + (1-prob_f1_in_eqm2)*(1-prob_f2_in_eqm2)*prob_f3_in_eqm2*(1-prob_f4_in_eqm2) \
           + (1-prob_f1_in_eqm2)*(1-prob_f2_in_eqm2)*(1-prob_f3_in_eqm2)*prob_f4_in_eqm2
    
    H_03 = (1-prob_f1_in_eqm3)*(1-prob_f2_in_eqm3)*(1-prob_f3_in_eqm3)*(1-prob_f4_in_eqm3)
    
    H_13 = prob_f1_in_eqm3*(1-prob_f2_in_eqm3)*(1-prob_f3_in_eqm3)*(1-prob_f4_in_eqm3) \
           + (1-prob_f1_in_eqm3)*prob_f2_in_eqm3*(1-prob_f3_in_eqm3)*(1-prob_f4_in_eqm3) \
           + (1-prob_f1_in_eqm3)*(1-prob_f2_in_eqm3)*prob_f3_in_eqm3*(1-prob_f4_in_eqm3) \
           + (1-prob_f1_in_eqm3)*(1-prob_f2_in_eqm3)*(1-prob_f3_in_eqm3)*prob_f4_in_eqm3
    
    H_23 = prob_f1_in_eqm3*prob_f2_in_eqm3*(1-prob_f3_in_eqm3)*(1-prob_f4_in_eqm3) \
           + prob_f1_in_eqm3*(1-prob_f2_in_eqm3)*prob_f3_in_eqm3*(1-prob_f4_in_eqm3) \
           + prob_f1_in_eqm3*(1-prob_f2_in_eqm3)*(1-prob_f3_in_eqm3)*prob_f4_in_eqm3 \
           + (1-prob_f1_in_eqm3)*prob_f2_in_eqm3*prob_f3_in_eqm3*(1-prob_f4_in_eqm3) \
           + (1-prob_f1_in_eqm3)*prob_f2_in_eqm3*(1-prob_f3_in_eqm3)*prob_f4_in_eqm3 \
           + (1-prob_f1_in_eqm3)*(1-prob_f2_in_eqm3)*prob_f3_in_eqm3*prob_f4_in_eqm3 \
    
    H_04 = (1-prob_f1_in_eqm4)*(1-prob_f2_in_eqm4)*(1-prob_f3_in_eqm4)*(1-prob_f4_in_eqm4)
    
    H_14 = prob_f1_in_eqm4*(1-prob_f2_in_eqm4)*(1-prob_f3_in_eqm4)*(1-prob_f4_in_eqm4) \
           + (1-prob_f1_in_eqm4)*prob_f2_in_eqm4*(1-prob_f3_in_eqm4)*(1-prob_f4_in_eqm4) \
           + (1-prob_f1_in_eqm4)*(1-prob_f2_in_eqm4)*prob_f3_in_eqm4*(1-prob_f4_in_eqm4) \
           + (1-prob_f1_in_eqm4)*(1-prob_f2_in_eqm4)*(1-prob_f3_in_eqm4)*prob_f4_in_eqm4
    
    H_24 = prob_f1_in_eqm4*prob_f2_in_eqm4*(1-prob_f3_in_eqm4)*(1-prob_f4_in_eqm4) \
           + prob_f1_in_eqm4*(1-prob_f2_in_eqm4)*prob_f3_in_eqm4*(1-prob_f4_in_eqm4) \
           + prob_f1_in_eqm4*(1-prob_f2_in_eqm4)*(1-prob_f3_in_eqm4)*prob_f4_in_eqm4 \
           + (1-prob_f1_in_eqm4)*prob_f2_in_eqm4*prob_f3_in_eqm4*(1-prob_f4_in_eqm4) \
           + (1-prob_f1_in_eqm4)*prob_f2_in_eqm4*(1-prob_f3_in_eqm4)*prob_f4_in_eqm4 \
           + (1-prob_f1_in_eqm4)*(1-prob_f2_in_eqm4)*prob_f3_in_eqm4*prob_f4_in_eqm4 \

    H_34 = prob_f1_in_eqm4*prob_f2_in_eqm4*prob_f3_in_eqm4*(1-prob_f4_in_eqm4) \
           + prob_f1_in_eqm4*prob_f2_in_eqm4*(1-prob_f3_in_eqm4)*prob_f4_in_eqm4 \
           + prob_f1_in_eqm4*(1-prob_f2_in_eqm4)*prob_f3_in_eqm4*prob_f4_in_eqm4 \
           + (1-prob_f1_in_eqm4)*prob_f2_in_eqm4*prob_f3_in_eqm4*prob_f4_in_eqm4
    
    H_44 = prob_f1_in_eqm4 * prob_f2_in_eqm4 * prob_f3_in_eqm4 * prob_f4_in_eqm4

    
    
    if N_star == 0:
        prob = H_01   
    
    elif N_star == 1:  
        prob = H_02 + H_12 - H_01 
    
    elif N_star == 2:
        prob = H_03 + H_13 + H_23 - (H_02 + H_12)
    
    elif N_star == 3:
        prob = H_04 + H_14 + H_24 + H_34 - (H_03 + H_13 + H_23)
        
    elif N_star == 4:
        prob = H_44
    
    return prob
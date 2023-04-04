import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
import math
import time 
from scipy.stats import norm
import ray


# define demand shock transition matrix
x_transition = np.array([[0.6, 0.2, 0.2],
                         [0.2, 0.6, 0.2],
                         [0.2, 0.2, 0.6]])

def x_simulator(current_x_idx, s):
    '''
    input: current x index
    output: tmr x index
    '''
    random.seed(s)
    tmr_x_idx = random.choices(population = [0,1,2], weights = x_transition[current_x_idx], k = 1)[0]
    return tmr_x_idx

@ray.remote
def forward_simulate_incumbent(beta, cournot_pi_matrix, entry_TH, exit_TH, theta_distribution, s, N_init, x_idx_init):
# forward simulate incumbent's life cycle conditional on today's remaining decision
# function name: forward_simulate_incumbent
# function input: N_init, x_idx_init
# function output: PDV
# N_init: 1,2,3,4,5
# x_dix_init: 0,1,2
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    t=0
    time_idx = [0]
    # Start with N_init
    N_history = [N_init]
    # Start with x_idx_init
    x_idx_history = [x_idx_init]
    # Remaining decision has not been made
    i_remain_history = []

    np.random.seed(s)
    # t=0 simulate
    
    #(1) x is updated exogenously 
    x_idx_pre = x_simulator(x_idx_init, s)
    
    # N_pre will be the initial value for the while loop for (t>=1)
    if N_init == 1:
        # When the incumbent is the single firm in the market
        # Potential entrant is the only factor that affects N'
        ## My decision is fixed as remain
        i_remain_decision = 1
        gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
        entry_TH_value = entry_TH[N_init][x_idx_init]
        ## Entrants decision is made
        entry_decision = int(gamma_draw < entry_TH_value)
        ## N is updated
        N_pre = N_init + entry_decision
    
    elif N_init == 5:
        # When there are five firms in the market, the potential entrant can not enter
        ## My decision is fixed as remain
        i_remain_decision = 1
        ## Other incumbents decisions
        mu_draw_array = np.random.normal(mu_loc, mu_scale, N_init-1)
        exit_TH_value_array = np.full(N_init-1, exit_TH[N_init-1][x_idx_init])
        ## N is updated
        N_pre = N_init - (mu_draw_array > exit_TH_value_array).sum()
        
    else:
        # When there are 2,3,4 firms in the market
        ## My decision is fixed as remain
        i_remain_decision = 1
        ## Other incumbents decisions
        mu_draw_array = np.random.normal(mu_loc, mu_scale, N_init-1)
        exit_TH_value_array = np.full(N_init-1, exit_TH[N_init-1][x_idx_init])
        gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
        entry_TH_value = entry_TH[N_init][x_idx_init]
        ## Entrants decision is made
        entry_decision = int(gamma_draw < entry_TH_value)
        ## N is updated
        N_pre = N_init - (mu_draw_array > exit_TH_value_array).sum() + entry_decision
        

    N_history.append(N_pre)                     # at this point len(N_history) == 2 
    x_idx_history.append(x_idx_pre)             # at this point len(x_idx_history) == 2 
    i_remain_history.append(i_remain_decision)  # at this point len(i_remain_history) == 1


    while(t < 1000):
        t += 1
        time_idx.append(t)
        N_current = N_history[-1]
        x_idx_current = x_idx_history[-1]
        
        np.random.seed(s*1000+t)
        #(1) decide i's exit decision
        i_mu_draw = np.random.normal(mu_loc, mu_scale, 1)[0]
        exit_TH_value = exit_TH[N_current-1][x_idx_current]
        i_remain_decision = int(i_mu_draw < exit_TH_value)
        
        # break condition conditional on i's exit decision
        if i_remain_decision == 0: # i exit
            i_remain_history.append(i_remain_decision)
            scrap_value = i_mu_draw
            break

        #(2) continue forward simulating
        #note that i decided to remain in the market at this point
        #(2-1) update the state variables
        x_idx_tmr = x_simulator(x_idx_current, s*1000+t)

        if N_current == 1:
            # i is the single incumbent in this market
            # so only the entrant will change the number of firms in the market
            gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
            entry_TH_value = entry_TH[N_current][x_idx_current]
            entry_decision = int(gamma_draw < entry_TH_value)
            N_tmr = N_current + entry_decision
            
        
        elif N_current == 5:
            # since there are 5 incumbents in the market, entrant can not enter the market tmr
            # hence, only the incumbents can change the number of firms in the market
            mu_draw_array = np.random.normal(mu_loc, mu_scale, N_current-1)
            exit_TH_value_array = np.full(N_current-1, exit_TH[N_current-1][x_idx_current])
            N_tmr = N_current - (mu_draw_array > exit_TH_value_array).sum()
            

        else:    
            mu_draw_array = np.random.normal(mu_loc, mu_scale, N_current-1)
            exit_TH_value_array = np.full(N_current-1, exit_TH[N_current-1][x_idx_current])
            gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
            entry_TH_value = entry_TH[N_current][x_idx_current]
            entry_decision = int(gamma_draw < entry_TH_value)
            N_tmr = N_current - (mu_draw_array > exit_TH_value_array).sum() + entry_decision
            
        #(2-2) record the decision and transition
        i_remain_history.append(i_remain_decision)
        N_history.append(N_tmr)
        x_idx_history.append(x_idx_tmr)

    df_incumbent_simulation = df({"t": time_idx, "N": N_history, "x_idx": x_idx_history, "remain_decision": i_remain_history})
    def incumbent_PDV(t, N, x_idx, remain_decision, scrap_value):
        if remain_decision == 0:
            pdv = (beta**t)*scrap_value
        else:
            pdv = (beta**t)*cournot_pi_matrix[N-1][x_idx]
        return pdv
    vec_incumbent_PDV = np.vectorize(incumbent_PDV)
    PDV = vec_incumbent_PDV(df_incumbent_simulation['t'].values, df_incumbent_simulation['N'].values, df_incumbent_simulation['x_idx'].values, df_incumbent_simulation['remain_decision'].values, scrap_value).sum()

    return PDV


@ray.remote
def forward_simulate_entrant(beta, cournot_pi_matrix, entry_TH, exit_TH, theta_distribution, s, N_init, x_idx_init):
# forward simulate entrant's life cycle conditional on today's entry decision
# function name: forward_simulate_entrant
# function input: N_init, x_idx_init
# function output: PDV
# N_init: 0,1,2,3,4
# x_dix_init: 0,1,2
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    t=0
    time_idx = [0]
    N_history = [N_init]
    x_idx_history = [x_idx_init]
    i_remain_history = [1] #meaningless


    np.random.seed(s)
    # t=0 simulate
    # N_pre will be the initial value for the following while loop for (t>=1)
    if N_init == 0:
        N_pre = N_init + 1
        x_idx_pre = x_simulator(x_idx_init, s)
    
    else:
        mu_draw_array = np.random.normal(mu_loc, mu_scale, N_init)
        exit_TH_value_array = np.full(N_init, exit_TH[N_init-1][x_idx_init])
        N_pre = N_init - (mu_draw_array > exit_TH_value_array).sum() + 1
        x_idx_pre = x_simulator(x_idx_init, s)

    N_history.append(N_pre)
    x_idx_history.append(x_idx_pre)


    while(t < 1000):
        t += 1
        time_idx.append(t)
        N_current = N_history[-1]
        x_idx_current = x_idx_history[-1]
        
        np.random.seed(1000*s + t)
        #(1) decide i(=e)'s exit decision
        i_mu_draw = np.random.normal(mu_loc, mu_scale, 1)[0]
        exit_TH_value = exit_TH[N_current-1][x_idx_current]
        i_remain_decision = int(i_mu_draw < exit_TH_value)
        
        # break condition conditional on i's exit decision
        if i_remain_decision == 0: # i exit
            i_remain_history.append(i_remain_decision)
            scrap_value = i_mu_draw
            break

        #(2) continue forward simulating
        #note that e decided to remain in the market at this point
        #(2-1) update the state variables
        if N_current == 1:
            # e is the single incumbent in this market
            # so only the entrant will change the number of firms in the market
            gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
            entry_TH_value = entry_TH[N_current][x_idx_current]
            entry_decision = int(gamma_draw < entry_TH_value)
            N_tmr = N_current + entry_decision
            x_idx_tmr = x_simulator(x_idx_current, 1000*s + t)
        
        elif N_current == 5:
            # since there are 5 incumbents in the market, entrant can not enter the market tmr
            # hence, only the incumbents can change the number of firms in the market
            np.random.seed(s)
            mu_draw_array = np.random.normal(mu_loc, mu_scale, N_current-1) 
            exit_TH_value_array = np.full(N_current-1, exit_TH[N_current-1][x_idx_current])
            N_tmr = N_current - (mu_draw_array > exit_TH_value_array).sum()
            x_idx_tmr = x_simulator(x_idx_current, 1000*s + t)

        else:
            np.random.seed(s)
            mu_draw_array = np.random.normal(mu_loc, mu_scale, N_current-1)
            exit_TH_value_array = np.full(N_current-1, exit_TH[N_current-1][x_idx_current])
            np.random.seed(s)
            gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
            entry_TH_value = entry_TH[N_current][x_idx_current]
            entry_decision = int(gamma_draw < entry_TH_value)
            N_tmr = N_current - (mu_draw_array > exit_TH_value_array).sum() + entry_decision
            x_idx_tmr = x_simulator(x_idx_current, 1000*s + t)

        #(2-2) record the decision and transition
        i_remain_history.append(i_remain_decision)
        N_history.append(N_tmr)
        x_idx_history.append(x_idx_tmr)
    
    df_entrant_simulation = df({"t": time_idx, "N": N_history, "x_idx": x_idx_history, "remain_decision": i_remain_history})
    df_entrant_simulation = df_entrant_simulation.loc[df_entrant_simulation['t'] != 0]
    def entrant_PDV(t, N, x_idx, remain_decision, scrap_value):    
        if remain_decision == 0:
            pdv = (beta**t)*scrap_value
        else:
            pdv = (beta**t)*cournot_pi_matrix[N-1][x_idx]
        return pdv
    vec_entrant_PDV = np.vectorize(entrant_PDV)
    PDV = vec_entrant_PDV(df_entrant_simulation['t'].values, df_entrant_simulation['N'].values, df_entrant_simulation['x_idx'].values, df_entrant_simulation['remain_decision'].values, scrap_value).sum()

    return PDV






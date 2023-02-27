import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
import math

# Main DGP
def DGP(alpha, beta, delta, mu, sigma, market_number):
    # generate base data
    market_id_list = []
    potential_entrant_list = []
    X_m_list = []
    Z_fm_list = []
    u_fm_list = []

    #within market i, construct data
    for i in range(market_number):
        
        #market characteristics 
        X_m = np.random.normal(3,1,1)[0]    

        #potential number of entrants is randomly chosen among [2,3,4]
        potential_entrant = random.choices(population = [2,3,4],
                                        weights = [1/3, 1/3, 1/3],
                                        k = 1)[0]

        #dupicate market level data for #(potential number of entrants) times                                        
        for j in range(potential_entrant):
            market_id_list.append(i)
            potential_entrant_list.append(potential_entrant)
            X_m_list.append(X_m)
        
        #draw cost shifter for #(potential number of entrants) times
        Z_fm = np.random.normal(0, 1, potential_entrant)

        #draw cost shock for #(potential number of entrants) times
        u_fm = np.random.normal(mu, sigma, potential_entrant)
        
        #append
        Z_fm_list = Z_fm_list + list(Z_fm)
        u_fm_list = u_fm_list + list(u_fm)
        
    #base_data is done --> need to calculate which firms will enter
    base_data = df({'market_id': market_id_list,
                'potential_entrant': potential_entrant_list,
                'X_m': X_m_list,
                'Z_fm': Z_fm_list,
                'u_fm': u_fm_list})
    base_data['phi_fm'] = base_data['Z_fm']*alpha + base_data['u_fm']

    # calculate N_star (eqm number of entrants)
    N_star = []
    for i in range(market_number):
        work_data = base_data.loc[base_data['market_id']==i]
        potential_entrant = list(work_data['potential_entrant'])[0]
        X_m = list(work_data['X_m'])[0]
        phi_list = list(work_data['phi_fm'])
        phi_list.sort()
        bool_temp = []

        # e.g) suppose N = 2
        # if phi_list[2-1] (i.e second best cost efficient firm) can be profitable, this market can rationalize 2 firms eqm
        for N in range(1, potential_entrant+1):
            V_N = beta*X_m - delta*math.log(N)
            bool_temp.append(int(V_N - phi_list[N-1] >= 0))
        N_star.append(np.array(bool_temp).sum())

    market_id_unique_list = [i for i in range(market_number)]
    N_star_data = df({'market_id': market_id_unique_list,
                    'N_star': N_star})

    master_data = pd.merge(base_data, N_star_data, how = 'left', left_on = 'market_id', right_on = 'market_id')
    master_data['cost_rank'] = master_data.groupby('market_id')['phi_fm'].rank(method='min', ascending = True)
    # cost efficient firms enter first
    master_data['entrant'] = master_data['cost_rank']<=master_data['N_star']

    return master_data

# Give higher probability to two firm cases only for some experiments
def DGP2(alpha, beta, delta, mu, sigma, market_number):
    # generate base data
    market_id_list = []
    potential_entrant_list = []
    X_m_list = []
    Z_fm_list = []
    u_fm_list = []

    #within market i, construct data
    for i in range(market_number):
        
        #market characteristics 
        X_m = np.random.normal(3,1,1)[0]    

        #potential number of entrants is randomly chosen among [2,3,4]
        potential_entrant = random.choices(population = [2,3,4],
                                        weights = [4/5, 1/10, 1/10],
                                        k = 1)[0]

        #dupicate market level data for #(potential number of entrants) times                                        
        for j in range(potential_entrant):
            market_id_list.append(i)
            potential_entrant_list.append(potential_entrant)
            X_m_list.append(X_m)
        
        #draw cost shifter for #(potential number of entrants) times
        Z_fm = np.random.normal(0, 1, potential_entrant)

        #draw cost shock for #(potential number of entrants) times
        u_fm = np.random.normal(mu, sigma, potential_entrant)
        
        #append
        Z_fm_list = Z_fm_list + list(Z_fm)
        u_fm_list = u_fm_list + list(u_fm)
        
    #base_data is done --> need to calculate which firms will enter
    base_data = df({'market_id': market_id_list,
                'potential_entrant': potential_entrant_list,
                'X_m': X_m_list,
                'Z_fm': Z_fm_list,
                'u_fm': u_fm_list})
    base_data['phi_fm'] = base_data['Z_fm']*alpha + base_data['u_fm']

    # calculate N_star (eqm number of entrants)
    N_star = []
    for i in range(market_number):
        work_data = base_data.loc[base_data['market_id']==i]
        potential_entrant = list(work_data['potential_entrant'])[0]
        X_m = list(work_data['X_m'])[0]
        phi_list = list(work_data['phi_fm'])
        phi_list.sort()
        bool_temp = []

        # e.g) suppose N = 2
        # if phi_list[2-1] (i.e second best cost efficient firm) can be profitable, this market can rationalize 2 firms eqm
        for N in range(1, potential_entrant+1):
            V_N = beta*X_m - delta*math.log(N)
            bool_temp.append(int(V_N - phi_list[N-1] >= 0))
        N_star.append(np.array(bool_temp).sum())

    market_id_unique_list = [i for i in range(market_number)]
    N_star_data = df({'market_id': market_id_unique_list,
                    'N_star': N_star})

    master_data = pd.merge(base_data, N_star_data, how = 'left', left_on = 'market_id', right_on = 'market_id')
    master_data['cost_rank'] = master_data.groupby('market_id')['phi_fm'].rank(method='min', ascending = True)
    # cost efficient firms enter first
    master_data['entrant'] = master_data['cost_rank']<=master_data['N_star']

    return master_data
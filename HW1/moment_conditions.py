import numpy as np
import pandas as pd
from pandas import DataFrame as df
import random
import math
from scipy.stats import norm
from scipy.optimize import minimize
import time
import ray
import warnings
from numpy.linalg import inv



def SMM_estimator2(theta, data_observable):
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    market_number = 250

    @ray.remote
    def inner_loop(m):
        work_data = data_observable.loc[data_observable['market_id']==m]
        z_array = np.array(work_data['Z_fm'])
        X_m = list(work_data['X_m'])[0]
        N_star = list(work_data['N_star'])[0]
        potential_entrant = list(work_data['potential_entrant'])[0]

    
        n_hat_list = [] 
        p_hat_1_list = []
        p_hat_2_list = []

        # draw cost shock for 200 many times
        for t in range(200):
            seed_num = m*1000 + t
            np.random.seed(seed_num)
            cost_shock_draw = np.random.normal(mu, sigma, potential_entrant)
            phi_array = np.array(alpha*z_array + cost_shock_draw)
            
            # calculate n_hat under the t th cost shock
            for k in range(potential_entrant): 
                # if potential_entrant == 4, N will be looped 4, 3, 2, 1
                N = potential_entrant - k
                pi_array = beta*X_m - delta*math.log(N) - phi_array
                positive_cnt = 0
                for pi in pi_array:
                    if pi >= 0:
                        positive_cnt += 1
                if positive_cnt >= N:
                    break
            n_hat_list.append(N)


            # to update cost rank for each t, pick the random 2 firms here
            work_data['phi'] = phi_array
            work_data['cost_rank'] = work_data['phi'].rank(method='min', ascending = True) #order the cost rank
            
            firm_id_list = list(work_data['firm_id'])
            random.seed(m)                                                  #fix the random seed for market m
            firm_id_list2 = random.sample(firm_id_list, 2)                  #pick the random 2 firms
            work_data_v12 = work_data.loc[work_data['firm_id'].isin(firm_id_list2)]

            p1_rank = list(work_data_v12['cost_rank'])[0]
            p2_rank = list(work_data_v12['cost_rank'])[1]

            # eqm selection is applied here
            # mis-specified model will be differentiated this part
            p_hat_1_list.append(int(p1_rank <= N) * 1 + (1-int(p1_rank <= N))*0)
            p_hat_2_list.append(int(p2_rank <= N) * 1 + (1-int(p2_rank <= N))*0)
            '''
            if p1_rank <= N:
                p_hat_1_list.append(1)
            else:
                p_hat_1_list.append(0)
            
            if p2_rank <= N:
                p_hat_2_list.append(1)
            else:
                p_hat_2_list.append(0)
            '''

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        return np.array([V_m0, V_m1, V_m2, V_m0*X_m, V_m1*X_m, V_m2*X_m])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar@g_bar


def SMM_estimator3(theta, data_observable):
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    market_number = 250

    @ray.remote
    def inner_loop(m):
        work_data = data_observable.loc[data_observable['market_id']==m]
        z_array = np.array(work_data['Z_fm'])
        X_m = list(work_data['X_m'])[0]
        N_star = list(work_data['N_star'])[0]
        potential_entrant = list(work_data['potential_entrant'])[0]

    
        n_hat_list = [] 
        p_hat_1_list = []
        p_hat_2_list = []

        # draw cost shock for 200 many times
        for t in range(200):
            seed_num = m*1000 + t
            np.random.seed(seed_num)
            cost_shock_draw = np.random.normal(mu, sigma, potential_entrant)
            phi_array = np.array(alpha*z_array + cost_shock_draw)
            
            # calculate n_hat under the t th cost shock
            for k in range(potential_entrant): 
                # if potential_entrant == 4, N will be looped 4, 3, 2, 1
                N = potential_entrant - k
                pi_array = beta*X_m - delta*math.log(N) - phi_array
                positive_cnt = 0
                for pi in pi_array:
                    if pi >= 0:
                        positive_cnt += 1
                if positive_cnt >= N:
                    break
            n_hat_list.append(N)


            # to update cost rank for each t, pick the random 2 firms here
            work_data['phi'] = phi_array
            work_data['cost_rank'] = work_data['phi'].rank(method='min', ascending = True) #order the cost rank
            
            firm_id_list = list(work_data['firm_id'])
            random.seed(m)                                                  #fix the random seed for market m
            firm_id_list2 = random.sample(firm_id_list, 2)                  #pick the random 2 firms
            work_data_v12 = work_data.loc[work_data['firm_id'].isin(firm_id_list2)]

            p1_rank = list(work_data_v12['cost_rank'])[0]
            p2_rank = list(work_data_v12['cost_rank'])[1]

            # eqm selection is applied here
            # mis-specified model will be differentiated this part
            p_hat_1_list.append(int(p1_rank <= N) * 1 + (1-int(p1_rank <= N))*0)
            p_hat_2_list.append(int(p2_rank <= N) * 1 + (1-int(p2_rank <= N))*0)
            '''
            if p1_rank <= N:
                p_hat_1_list.append(1)
            else:
                p_hat_1_list.append(0)
            
            if p2_rank <= N:
                p_hat_2_list.append(1)
            else:
                p_hat_2_list.append(0)
            '''

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        return np.array([V_m0, V_m1, V_m2, V_m0*potential_entrant, V_m1*potential_entrant, V_m2*potential_entrant, V_m0*X_m, V_m1*X_m, V_m2*X_m])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar@g_bar

def SMM_estimator4(theta, data_observable):
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    market_number = 250

    @ray.remote
    def inner_loop(m):
        work_data = data_observable.loc[data_observable['market_id']==m]
        z_array = np.array(work_data['Z_fm'])
        X_m = list(work_data['X_m'])[0]
        N_star = list(work_data['N_star'])[0]
        potential_entrant = list(work_data['potential_entrant'])[0]

    
        n_hat_list = [] 
        p_hat_1_list = []
        p_hat_2_list = []

        # draw cost shock for 200 many times
        for t in range(200):
            seed_num = m*1000 + t
            np.random.seed(seed_num)
            cost_shock_draw = np.random.normal(mu, sigma, potential_entrant)
            phi_array = np.array(alpha*z_array + cost_shock_draw)
            
            # calculate n_hat under the t th cost shock
            for k in range(potential_entrant): 
                # if potential_entrant == 4, N will be looped 4, 3, 2, 1
                N = potential_entrant - k
                pi_array = beta*X_m - delta*math.log(N) - phi_array
                positive_cnt = 0
                for pi in pi_array:
                    if pi >= 0:
                        positive_cnt += 1
                if positive_cnt >= N:
                    break
            n_hat_list.append(N)


            # to update cost rank for each t, pick the random 2 firms here
            work_data['phi'] = phi_array
            work_data['cost_rank'] = work_data['phi'].rank(method='min', ascending = True) #order the cost rank
            
            firm_id_list = list(work_data['firm_id'])
            random.seed(m)                                                  #fix the random seed for market m
            firm_id_list2 = random.sample(firm_id_list, 2)                  #pick the random 2 firms
            work_data_v12 = work_data.loc[work_data['firm_id'].isin(firm_id_list2)]

            p1_rank = list(work_data_v12['cost_rank'])[0]
            p2_rank = list(work_data_v12['cost_rank'])[1]

            # eqm selection is applied here
            # mis-specified model will be differentiated this part
            p_hat_1_list.append(int(p1_rank <= N) * 1 + (1-int(p1_rank <= N))*0)
            p_hat_2_list.append(int(p2_rank <= N) * 1 + (1-int(p2_rank <= N))*0)
            '''
            if p1_rank <= N:
                p_hat_1_list.append(1)
            else:
                p_hat_1_list.append(0)
            
            if p2_rank <= N:
                p_hat_2_list.append(1)
            else:
                p_hat_2_list.append(0)
            '''

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        return np.array([V_m0, V_m1, V_m2, V_m1**2, V_m2**2])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar@g_bar

def SMM_estimator5(theta, data_observable):
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    market_number = 250

    @ray.remote
    def inner_loop(m):
        work_data = data_observable.loc[data_observable['market_id']==m]
        z_array = np.array(work_data['Z_fm'])
        X_m = list(work_data['X_m'])[0]
        N_star = list(work_data['N_star'])[0]
        potential_entrant = list(work_data['potential_entrant'])[0]

    
        n_hat_list = [] 
        p_hat_1_list = []
        p_hat_2_list = []

        # draw cost shock for 200 many times
        for t in range(200):
            seed_num = m*1000 + t
            np.random.seed(seed_num)
            cost_shock_draw = np.random.normal(mu, sigma, potential_entrant)
            phi_array = np.array(alpha*z_array + cost_shock_draw)
            
            # calculate n_hat under the t th cost shock
            for k in range(potential_entrant): 
                # if potential_entrant == 4, N will be looped 4, 3, 2, 1
                N = potential_entrant - k
                pi_array = beta*X_m - delta*math.log(N) - phi_array
                positive_cnt = 0
                for pi in pi_array:
                    if pi >= 0:
                        positive_cnt += 1
                if positive_cnt >= N:
                    break
            n_hat_list.append(N)


            # to update cost rank for each t, pick the random 2 firms here
            work_data['phi'] = phi_array
            work_data['cost_rank'] = work_data['phi'].rank(method='min', ascending = True) #order the cost rank
            
            firm_id_list = list(work_data['firm_id'])
            random.seed(m)                                                  #fix the random seed for market m
            firm_id_list2 = random.sample(firm_id_list, 2)                  #pick the random 2 firms
            work_data_v12 = work_data.loc[work_data['firm_id'].isin(firm_id_list2)]

            p1_rank = list(work_data_v12['cost_rank'])[0]
            p2_rank = list(work_data_v12['cost_rank'])[1]

            # eqm selection is applied here
            # mis-specified model will be differentiated this part
            p_hat_1_list.append(int(p1_rank <= N) * 1 + (1-int(p1_rank <= N))*0)
            p_hat_2_list.append(int(p2_rank <= N) * 1 + (1-int(p2_rank <= N))*0)
            '''
            if p1_rank <= N:
                p_hat_1_list.append(1)
            else:
                p_hat_1_list.append(0)
            
            if p2_rank <= N:
                p_hat_2_list.append(1)
            else:
                p_hat_2_list.append(0)
            '''

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        return np.array([V_m0, V_m1, V_m2, V_m1**2, V_m2**2, V_m0*potential_entrant, V_m1*potential_entrant, V_m2*potential_entrant, V_m0*X_m, V_m1*X_m, V_m2*X_m])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar@g_bar
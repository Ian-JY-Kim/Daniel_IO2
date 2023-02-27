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

################################
##       Question 3          ##
################################

############################
# (1) correctly Speified  #
############################
# randomly choose two firms
# moment conditions: first and second moments
# main result 
def SMM_estimator(theta, data_observable):
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
            pd.set_option('mode.chained_assignment',  None)
            work_data['phi'] = phi_array
            work_data['cost_rank'] = work_data['phi'].rank(method='min', ascending = True) #order the cost rank
            
            firm_id_list = list(work_data['firm_id'])
            random.seed(m)                                                  #fix the random seed for market m
            firm_id_list2 = random.sample(firm_id_list, 2)                  #pick the random 2 firms
            work_data_v12 = work_data.loc[work_data['firm_id'].isin(firm_id_list2)]

            p1_rank = list(work_data_v12['cost_rank'])[0]
            p2_rank = list(work_data_v12['cost_rank'])[1]

            # eqm selection is applied here
            # mis-specified model will be differentiated by this part
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
    #g_list = [inner_loop.remote(m) for m in range(100)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar@g_bar

# theta: SMM_result.x
def SMM_weighting_matrix(theta, data_observable):
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    market_number = 250

    #1. calculate g_tilde_bar
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
            pd.set_option('mode.chained_assignment',  None)
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

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        # return v times f
        return np.array([V_m0, V_m1, V_m2, V_m1**2, V_m2**2])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    # array of g_tilde 
    g_tilde_array = np.array(ray.get(g_list))

    # mean value of the array of g_tild 
    g_tilde_bar = np.mean(g_tilde_array, axis = 0) 

    # calculate sigma part first
    error_array = g_tilde_array - g_tilde_bar
    
    # calculate omega hat and associated weighting matrix
    outer_product_list = [np.outer(error_array[m], error_array[m]) for m in range(market_number)]    
    outer_product_array = np.array(outer_product_list)
    omega_hat_sum = 0
    for matrix in outer_product_array:
        omega_hat_sum += matrix
    omega_hat = omega_hat_sum/250
    W_hat = inv(omega_hat)
    
    return W_hat

# two_step SMM
# weighting matrix is defined outside of the function globally
def efficient_SMM_estimator(theta, data_observable, W_hat):
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
            pd.set_option('mode.chained_assignment',  None)
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
    
    return g_bar@W_hat@g_bar

# calculate gbar
def SMM_gbar(theta, data_observable):
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
            pd.set_option('mode.chained_assignment',  None)
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
    #g_list = [inner_loop.remote(m) for m in range(100)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar

# calculate SE for the SMM estimator (i.e, correctly specified model)
def Pakes_Pollard_SE(theta_hat, W_hat, epsilon, data_observable):
    delta_deviation_1 = theta_hat.x + np.array([epsilon, 0, 0])
    delta_deviation_2 = theta_hat.x - np.array([epsilon, 0, 0])
    mu_deviation_1 = theta_hat.x + np.array([0, epsilon, 0])
    mu_deviation_2 = theta_hat.x - np.array([0, epsilon, 0])
    sigma_deviation_1 = theta_hat.x + np.array([0, 0, epsilon])
    sigma_deviation_2 = theta_hat.x - np.array([0, 0, epsilon])
    gamma_1 = (SMM_gbar(delta_deviation_1, data_observable) - SMM_gbar(delta_deviation_2, data_observable))/epsilon
    gamma_2 = (SMM_gbar(mu_deviation_1, data_observable) - SMM_gbar(mu_deviation_2, data_observable))/epsilon
    gamma_3 = (SMM_gbar(sigma_deviation_1, data_observable) - SMM_gbar(sigma_deviation_2, data_observable))/epsilon

    gamma = np.transpose(np.array([gamma_1, gamma_2, gamma_3]))
    gamma_T = np.transpose(gamma)
    var_matrix = inv(gamma_T@gamma)@gamma_T@inv(W_hat)@gamma@inv(gamma_T@gamma)

    std_error_delta = np.sqrt(var_matrix[0][0])/np.sqrt(250)
    std_error_mu = np.sqrt(var_matrix[1][1])/np.sqrt(250)
    std_error_sigma = np.sqrt(var_matrix[2][2])/np.sqrt(250)

    return std_error_delta, std_error_mu, std_error_sigma


############################
# (2) Misspeified         #
############################
def SMM_estimator_wrong(theta, data_observable):
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
            pd.set_option('mode.chained_assignment',  None)
            work_data['phi'] = phi_array
            work_data['cost_rank'] = work_data['phi'].rank(method='min', ascending = True) #order the cost rank
            
            firm_id_list = list(work_data['firm_id'])
            random.seed(m)                                                  #fix the random seed for market m
            firm_id_list2 = random.sample(firm_id_list, 2)                  #pick the random 2 firms
            work_data_v12 = work_data.loc[work_data['firm_id'].isin(firm_id_list2)]

            p1_rank = list(work_data_v12['cost_rank'])[0]
            p2_rank = list(work_data_v12['cost_rank'])[1]

            # eqm selection is applied here
            # mis-specified model will be differentiated by this part
            p_hat_1_list.append(int(p1_rank > potential_entrant - N) * 1 + (1-int(p1_rank > potential_entrant - N))*0)
            p_hat_2_list.append(int(p2_rank > potential_entrant - N) * 1 + (1-int(p2_rank > potential_entrant - N))*0)
            

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        return np.array([V_m0, V_m1, V_m2, V_m1**2, V_m2**2])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    #g_list = [inner_loop.remote(m) for m in range(100)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar@g_bar

def SMM_weighting_matrix_wrong(theta, data_observable):
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    market_number = 250

    #1. calculate g_tilde_bar
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
            pd.set_option('mode.chained_assignment',  None)
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
            p_hat_1_list.append(int(p1_rank > potential_entrant - N) * 1 + (1-int(p1_rank > potential_entrant - N))*0)
            p_hat_2_list.append(int(p2_rank > potential_entrant - N) * 1 + (1-int(p2_rank > potential_entrant - N))*0)

        N_hat = np.array(n_hat_list).mean()
        p_hat_1 = np.array(p_hat_1_list).mean()
        p_hat_2 = np.array(p_hat_2_list).mean()

        V_m0 = N_star - N_hat
        V_m1 = int(list(work_data_v12['entrant'])[0]) - p_hat_1
        V_m2 = int(list(work_data_v12['entrant'])[1]) - p_hat_2

        # return v times f
        return np.array([V_m0, V_m1, V_m2, V_m1**2, V_m2**2])

    g_list = [inner_loop.remote(m) for m in range(market_number)]
    # array of g_tilde 
    g_tilde_array = np.array(ray.get(g_list))

    # mean value of the array of g_tild 
    g_tilde_bar = np.mean(g_tilde_array, axis = 0) 

    # calculate sigma part first
    error_array = g_tilde_array - g_tilde_bar
    
    # calculate omega hat and associated weighting matrix
    outer_product_list = [np.outer(error_array[m], error_array[m]) for m in range(market_number)]    
    outer_product_array = np.array(outer_product_list)
    omega_hat_sum = 0
    for matrix in outer_product_array:
        omega_hat_sum += matrix
    omega_hat = omega_hat_sum/250
    W_hat = inv(omega_hat)
    
    return W_hat

def efficient_SMM_estimator_wrong(theta, data_observable, W_hat):
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
            pd.set_option('mode.chained_assignment',  None)
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
            p_hat_1_list.append(int(p1_rank > potential_entrant - N) * 1 + (1-int(p1_rank > potential_entrant - N))*0)
            p_hat_2_list.append(int(p2_rank > potential_entrant - N) * 1 + (1-int(p2_rank > potential_entrant - N))*0)

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
    
    return g_bar@W_hat@g_bar

def SMM_gbar_wrong(theta, data_observable):
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
            pd.set_option('mode.chained_assignment',  None)
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
            p_hat_1_list.append(int(p1_rank > potential_entrant - N) * 1 + (1-int(p1_rank > potential_entrant - N))*0)
            p_hat_2_list.append(int(p2_rank > potential_entrant - N) * 1 + (1-int(p2_rank > potential_entrant - N))*0)
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
    #g_list = [inner_loop.remote(m) for m in range(100)]
    g_array = np.array(ray.get(g_list))
    g_bar = np.mean(g_array, axis = 0) 
    
    return g_bar

def Pakes_Pollard_SE_wrong(theta_hat, W_hat, epsilon, data_observable):
    delta_deviation_1 = theta_hat.x + np.array([epsilon, 0, 0])
    delta_deviation_2 = theta_hat.x - np.array([epsilon, 0, 0])
    mu_deviation_1 = theta_hat.x + np.array([0, epsilon, 0])
    mu_deviation_2 = theta_hat.x - np.array([0, epsilon, 0])
    sigma_deviation_1 = theta_hat.x + np.array([0, 0, epsilon])
    sigma_deviation_2 = theta_hat.x - np.array([0, 0, epsilon])
    gamma_1 = (SMM_gbar_wrong(delta_deviation_1, data_observable) - SMM_gbar_wrong(delta_deviation_2, data_observable))/epsilon
    gamma_2 = (SMM_gbar_wrong(mu_deviation_1, data_observable) - SMM_gbar_wrong(mu_deviation_2, data_observable))/epsilon
    gamma_3 = (SMM_gbar_wrong(sigma_deviation_1, data_observable) - SMM_gbar_wrong(sigma_deviation_2, data_observable))/epsilon

    gamma = np.transpose(np.array([gamma_1, gamma_2, gamma_3]))
    gamma_T = np.transpose(gamma)
    var_matrix = inv(gamma_T@gamma)@gamma_T@inv(W_hat)@gamma@inv(gamma_T@gamma)

    std_error_delta = np.sqrt(var_matrix[0][0])/np.sqrt(250)
    std_error_mu = np.sqrt(var_matrix[1][1])/np.sqrt(250)
    std_error_sigma = np.sqrt(var_matrix[2][2])/np.sqrt(250)

    return std_error_delta, std_error_mu, std_error_sigma





################################
##       Question 4          ##
################################
# only use the number of entrants as the moment conditions
def SMM_q4(theta, data_observable): 
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

        N_hat_temp = []
        for t in range(1000):
            seed_num = m*1000 + t
            np.random.seed(seed_num)
            cost_shock_draw = np.random.normal(mu, sigma, potential_entrant)
            phi_array = np.array(alpha*z_array + cost_shock_draw)
            bool_temp = []
            for N in range(1, potential_entrant+1):
                pi_array = beta*X_m - delta*math.log(N) - phi_array
                positive_cnt = 0
                for pi in pi_array:
                    if pi >= 0:
                        positive_cnt += 1
                bool_temp.append(int(positive_cnt >= N))
            N_sample = np.array(bool_temp).sum()
            N_hat_temp.append(N_sample)
            
        return np.array(N_hat_temp).mean() - N_star

    error_list = [inner_loop.remote(m) for m in range(market_number)]
    error_array = np.array(ray.get(error_list))
    sum_sq_error = error_array@error_array
    
    return sum_sq_error
 
def MSM_estimator_0204_bootstrap(theta, data_observable, selection_list): 
    alpha, beta = [1,1]
    delta, mu, sigma = theta
    
    @ray.remote
    def inner_loop(m):
        work_data = data_observable.loc[data_observable['market_id']==m]
        z_array = np.array(work_data['Z_fm'])
        X_m = list(work_data['X_m'])[0]
        N_star = list(work_data['N_star'])[0]
        potential_entrant = list(work_data['potential_entrant'])[0]

        N_hat_temp = []
        for t in range(1000):
            seed_num = m*1000 + t
            np.random.seed(seed_num)
            cost_shock_draw = np.random.normal(mu, sigma, potential_entrant)
            phi_array = np.array(alpha*z_array + cost_shock_draw)
            bool_temp = []
            for N in range(1, potential_entrant+1):
                pi_array = beta*X_m - delta*math.log(N) - phi_array
                positive_cnt = 0
                for pi in pi_array:
                    if pi >= 0:
                        positive_cnt += 1
                bool_temp.append(int(positive_cnt >= N))
            N_sample = np.array(bool_temp).sum()
            N_hat_temp.append(N_sample)
            
        return np.array(N_hat_temp).mean() - N_star

    error_list = [inner_loop.remote(m) for m in selection_list]
    error_array = np.array(ray.get(error_list))
    sum_sq_error = error_array@error_array
    
    return sum_sq_error


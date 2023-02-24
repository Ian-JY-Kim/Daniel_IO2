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
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def long_to_wide(data_frame):
    data_frame = data_frame[['market_id', 'Z_fm', 'entrant']]
    data_frame['idx'] = data_frame.groupby('market_id').cumcount()+1
    data_frame = data_frame.pivot_table(index = 'market_id', columns = 'idx', values = ['Z_fm', 'entrant'], aggfunc = 'first')
    data_frame = data_frame.sort_index(axis=1, level=1)
    data_frame.columns = [f'{x}_{y}' for x,y in data_frame.columns]
    data_frame = data_frame.reset_index()

    return data_frame

def int2(str):
    return int(str,2)
vec_int2 = np.vectorize(int2)

def df_reshape(data_observable, potential_entrants):
    '''
    input: data_observable
    output: long to wide transformed data with designated number of potential entarnts
    '''
    data_observable_Xm = data_observable.drop_duplicates(subset = ['market_id'])
    data_observable_Xm = data_observable_Xm[['market_id', 'X_m']]

    df_reshaped = data_observable.loc[data_observable['potential_entrant'] == potential_entrants]
    df_reshaped = long_to_wide(df_reshaped)

    if potential_entrants == 2: 
        df_reshaped['market_structure'] = (df_reshaped['entrant_1'].astype(int)).astype(str) + (df_reshaped['entrant_2'].astype(int)).astype(str) 
    
    elif potential_entrants == 3:
        df_reshaped['market_structure'] = (df_reshaped['entrant_1'].astype(int)).astype(str) + (df_reshaped['entrant_2'].astype(int)).astype(str) \
                                            + (df_reshaped['entrant_3'].astype(int)).astype(str)

    elif potential_entrants == 4:
        df_reshaped['market_structure'] = (df_reshaped['entrant_1'].astype(int)).astype(str) + (df_reshaped['entrant_2'].astype(int)).astype(str) \
                                            + (df_reshaped['entrant_3'].astype(int)).astype(str) + (df_reshaped['entrant_4'].astype(int)).astype(str)

    df_reshaped['market_structure_B'] =  vec_int2(df_reshaped['market_structure'])

    df_reshaped = pd.merge(df_reshaped, data_observable_Xm, left_on= "market_id", right_on = "market_id")

    return df_reshaped

# attach bin number to the 2firm dataset
def bin_number_2firms(df_master):
    # discretize the X space into three different bins 
    # discretize the Z_1 space into two different bins
    # discretize the Z_2 space into two different bins
    # 4 x 2 x 2 = 16 cells
    df_2firms = df_master.copy()
    X_m_array = np.array(df_2firms['X_m'])
    
    X_p25 = np.percentile(X_m_array, 25)    
    X_p33 = np.percentile(X_m_array, 33)    
    X_p50 = np.percentile(X_m_array, 50)    
    X_p66 = np.percentile(X_m_array, 66)    
    X_p75 = np.percentile(X_m_array, 75)    
    
    
    df_2firms['bin_num_X'] = pd.cut(x = df_2firms['X_m'], 
                             bins = [-100, X_p50, 100],
                             labels = [1,2])


    # df_2firms['bin_num_X'] = pd.cut(x = df_2firms['X_m'], 
    #                          bins = [-100, X_p33, X_p66, 100],
    #                          labels = [1,2,3])
    
    df_2firms['bin_num_Z1'] = pd.cut(x = df_2firms['Z_fm_1'], 
                             bins = [-100, 0, 100],
                             labels = [1,2])

    df_2firms['bin_num_Z2'] = pd.cut(x = df_2firms['Z_fm_2'], 
                             bins = [-100, 0, 100],
                             labels = [1,2])
    
    df_2firms['bin_num'] = df_2firms['bin_num_X'].astype(int) * 100 \
                            + df_2firms['bin_num_Z1'].astype(int) * 10 \
                            + df_2firms['bin_num_Z2'].astype(int) 
    
    return df_2firms

# attach bin number to the 3firm dataset
def bin_number_3firms(df_master):
    # discretize the X space into two different bins 
    # discretize the Z_1 space into two different bins
    # discretize the Z_2 space into two different bins
    # discretize the Z_3 space into two different bins
    # 2 x 2 x 2 x 2 = 16 cells

    df_3firms = df_master.copy()
    X_m_array = np.array(df_3firms['X_m'])
    X_p50 = np.percentile(X_m_array, 50)    
    
    # df_3firms['bin_num'] = pd.cut(x = df_3firms['X_m'], 
    #                          bins = [-100, X_p50, 100],
    #                          labels = [1,2])
    
    
    df_3firms['bin_num_X'] = pd.cut(x = df_3firms['X_m'], 
                             bins = [-100, X_p50, 100],
                             labels = [1,2])
    
    df_3firms['bin_num_Z1'] = pd.cut(x = df_3firms['Z_fm_1'], 
                             bins = [-100, 0, 100],
                             labels = [1,2])

    df_3firms['bin_num_Z2'] = pd.cut(x = df_3firms['Z_fm_2'], 
                             bins = [-100, 0, 100],
                             labels = [1,2])
    
    df_3firms['bin_num_Z3'] = pd.cut(x = df_3firms['Z_fm_3'], 
                             bins = [-100, 0, 100],
                             labels = [1,2])
    
    df_3firms['bin_num'] = df_3firms['bin_num_X'].astype(int) * 1000 \
                            + df_3firms['bin_num_Z1'].astype(int) * 100 \
                            + df_3firms['bin_num_Z2'].astype(int) * 10 \
                            + df_3firms['bin_num_Z3'].astype(int)
    

    return df_3firms

# attach bin number to the 4firm dataset
def bin_number_4firms(df_master):
    # discretize the X space into 5 different bins 
    # 2 cells

    df_4firms = df_master.copy()
    X_m_array = np.array(df_4firms['X_m'])
    X_p20 = np.percentile(X_m_array, 20)    
    X_p25 = np.percentile(X_m_array, 25)    
    X_p40 = np.percentile(X_m_array, 40)    
    X_p50 = np.percentile(X_m_array, 50)    
    X_p60 = np.percentile(X_m_array, 60)    
    X_p75 = np.percentile(X_m_array, 75)    
    X_p80 = np.percentile(X_m_array, 80)    
    
    df_4firms['bin_num'] = pd.cut(x = df_4firms['X_m'], 
                             bins = [-100, X_p25, X_p50, X_p75, 100],
                             labels = [1,2,3,4])

    # df_4firms['bin_num'] = pd.cut(x = df_4firms['X_m'], 
    #                          bins = [-100, X_p50, 100],
    #                          labels = [1,2])
    
    # df_4firms['bin_num_X'] = pd.cut(x = df_4firms['X_m'], 
    #                          bins = [-100, X_p50, 100],
    #                          labels = [1,2])
    
    # df_4firms['bin_num_Z1'] = pd.cut(x = df_4firms['Z_fm_1'], 
    #                          bins = [-100, 0, 100],
    #                          labels = [1,2])

    # df_4firms['bin_num_Z2'] = pd.cut(x = df_4firms['Z_fm_2'], 
    #                          bins = [-100, 0, 100],
    #                          labels = [1,2])
    
    # df_4firms['bin_num_Z3'] = pd.cut(x = df_4firms['Z_fm_3'], 
    #                          bins = [-100, 0, 100],
    #                          labels = [1,2])

    # df_4firms['bin_num'] = df_4firms['bin_num_X'].astype(int) * 1000 \
    #                         + df_4firms['bin_num_Z1'].astype(int) * 100 \
    #                         + df_4firms['bin_num_Z2'].astype(int) * 10 

    return df_4firms

# calculate frequency of the 2firm case
def Freq_Est_2firms(df_2firms, bin_num):
    df_work = df_2firms.loc[df_2firms['bin_num']==bin_num]
    denominator = df_work.shape[0]
    market_structure_B_list = [int2("00"), int2("01"), int2("10"), int2("11")]
    
    for MS in market_structure_B_list:
        globals()['numerator_{}'.format(MS)] = (df_work.loc[df_work['market_structure_B'] == MS]).shape[0]
    return np.array([numerator_0, numerator_1, numerator_2, numerator_3])/denominator

# calculate frequency of the 3firm case
def Freq_Est_3firms(df_3firms, bin_num):
    df_work = df_3firms.loc[df_3firms['bin_num']==bin_num]
    denominator = df_work.shape[0]
    market_structure_B_list = [int2("000"), int2("001"), int2("010"), int2("011"), \
                               int2("100"), int2("101"), int2("110"), int2("111")]

    for MS in market_structure_B_list:
        globals()['numerator_{}'.format(MS)] = (df_work.loc[df_work['market_structure_B'] == MS]).shape[0]
    return np.array([numerator_0, numerator_1, numerator_2, numerator_3, 
                    numerator_4, numerator_5, numerator_6, numerator_7])/denominator

# calculate frequency of the 4firm case
def Freq_Est_4firms(df_4firms, bin_num):
    df_work = df_4firms.loc[df_4firms['bin_num']==bin_num]
    denominator = df_work.shape[0]
    market_structure_B_list = [int2("0000"), int2("0001"), int2("0010"), int2("0011"), \
                               int2("0100"), int2("0101"), int2("0110"), int2("0111"), \
                               int2("1000"), int2("1001"), int2("1010"), int2("1011"), \
                               int2("1100"), int2("1101"), int2("1110"), int2("1111")]

    for MS in market_structure_B_list:
        globals()['numerator_{}'.format(MS)] = (df_work.loc[df_work['market_structure_B'] == MS]).shape[0]
    return np.array([numerator_0, numerator_1, numerator_2, numerator_3, 
                    numerator_4, numerator_5, numerator_6, numerator_7,
                    numerator_8, numerator_9, numerator_10, numerator_11, 
                    numerator_12, numerator_13, numerator_14, numerator_15])/denominator



# calculate bound for 2firm case
def H_2firms(X_m, Z_1m, Z_2m, delta, mu, sigma):
    R = 500
    # mu and sigma affects cost shock draw
    np.random.seed(21)
    cost_shock_firm1 = np.random.normal(mu, sigma, R)
    np.random.seed(22)
    cost_shock_firm2 = np.random.normal(mu, sigma, R)

    def temp_calculate(firm1_decision, firm2_decision):
        num_entrants = firm1_decision + firm2_decision

        # delta affects firm's inside and outside option value
        if firm1_decision == 0:
            firm1_inside_option = np.zeros(R)
            firm1_outside_option = X_m - delta * np.log(num_entrants + 1) - (Z_1m + cost_shock_firm1)
        else: 
            firm1_inside_option = X_m - delta * np.log(num_entrants) - (Z_1m + cost_shock_firm1)
            firm1_outside_option = np.zeros(R)

        if firm2_decision == 0:
            firm2_inside_option = np.zeros(R)
            firm2_outside_option = X_m - delta * np.log(num_entrants + 1) - (Z_2m + cost_shock_firm2)
        else: 
            firm2_inside_option = X_m - delta * np.log(num_entrants) - (Z_2m + cost_shock_firm2)
            firm2_outside_option = np.zeros(R)

        firm1_profit_bool = (firm1_inside_option >= firm1_outside_option)
        firm2_profit_bool = (firm2_inside_option >= firm2_outside_option)
        temp = firm1_profit_bool * firm2_profit_bool
        return temp

    # temp_jk
    # both profitable under firm 1's decision is j and firm 2's decision is k
    # since we draw cost shock for R times, temp_jk is boolean list with size of R
    market_structure_B_list = ["00", "01", "10", "11"]   
    for MS in market_structure_B_list:
        globals()['temp_{}'.format(MS)] = temp_calculate(int(MS[0]), int(MS[1]))
    temp_sum = temp_00.astype(int) + temp_01.astype(int) + temp_10.astype(int) + temp_11.astype(int)

    H_1_00 = (temp_00*(temp_sum == 1)).sum()/R # number of case where 0,0 is the unique equilibrium / number of draw
    H_2_00 = (temp_00*(temp_sum >= 1)).sum()/R # number of case where 0,0 is one of the equilibria / number of draw

    H_1_01 = (temp_01*(temp_sum == 1)).sum()/R
    H_2_01 = (temp_01*(temp_sum >= 1)).sum()/R

    H_1_10 = (temp_10*(temp_sum == 1)).sum()/R
    H_2_10 = (temp_10*(temp_sum >= 1)).sum()/R

    H_1_11 = (temp_11*(temp_sum == 1)).sum()/R
    H_2_11 = (temp_11*(temp_sum >= 1)).sum()/R

    return H_1_00, H_1_01, H_1_10, H_1_11, H_2_00, H_2_01, H_2_10, H_2_11 

# calculate bound for 3firm case
def H_3firms(X_m, Z_1m, Z_2m, Z_3m, delta, mu, sigma):
    R = 500
    np.random.seed(31)
    cost_shock_firm1 = np.random.normal(mu, sigma, R)
    np.random.seed(32)
    cost_shock_firm2 = np.random.normal(mu, sigma, R)
    np.random.seed(33)
    cost_shock_firm3 = np.random.normal(mu, sigma, R)
    
    def temp_calculate(firm1_decision, firm2_decision, firm3_decision):
        ''' 
        output: temp_000, ... , temp_111
        '''
        decision_array = np.array([firm1_decision, firm2_decision, firm3_decision])
        entrants_num = decision_array.sum()

        #firm1
        if firm1_decision == 0:
            firm1_inside_option = np.zeros(R)
            firm1_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_1m + cost_shock_firm1)
        elif firm1_decision == 1: 
            firm1_inside_option = X_m - delta * np.log(entrants_num) - (Z_1m + cost_shock_firm1)
            firm1_outside_option = np.zeros(R)
        
        #firm2
        if firm2_decision == 0:
            firm2_inside_option = np.zeros(R)
            firm2_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_2m + cost_shock_firm2)
        elif firm2_decision == 1: 
            firm2_inside_option = X_m - delta * np.log(entrants_num) - (Z_2m + cost_shock_firm2)
            firm2_outside_option = np.zeros(R)

        #firm3
        if firm3_decision == 0:
            firm3_inside_option = np.zeros(R)
            firm3_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_3m + cost_shock_firm3)
        elif firm3_decision == 1: 
            firm3_inside_option = X_m - delta * np.log(entrants_num) - (Z_3m + cost_shock_firm3)
            firm3_outside_option = np.zeros(R)
        
        firm1_profit_bool = firm1_inside_option >= firm1_outside_option
        firm2_profit_bool = firm2_inside_option >= firm2_outside_option
        firm3_profit_bool = firm3_inside_option >= firm3_outside_option

        return (firm1_profit_bool * firm2_profit_bool * firm3_profit_bool).astype(int)
    
    temp_000 = temp_calculate(0,0,0)
    temp_001 = temp_calculate(0,0,1)
    temp_010 = temp_calculate(0,1,0)
    temp_011 = temp_calculate(0,1,1)
    temp_100 = temp_calculate(1,0,0)
    temp_101 = temp_calculate(1,0,1)
    temp_110 = temp_calculate(1,1,0)
    temp_111 = temp_calculate(1,1,1)
    temp_sum = temp_000 + temp_001 + temp_010 + temp_011 \
            + temp_100 + temp_101 + temp_110 + temp_111

    H_1_000 = (temp_000*(temp_sum == 1)).sum()/R
    H_2_000 = (temp_000*(temp_sum >= 1)).sum()/R

    H_1_001 = (temp_001*(temp_sum == 1)).sum()/R
    H_2_001 = (temp_001*(temp_sum >= 1)).sum()/R

    H_1_010 = (temp_010*(temp_sum == 1)).sum()/R
    H_2_010 = (temp_010*(temp_sum >= 1)).sum()/R

    H_1_011 = (temp_011*(temp_sum == 1)).sum()/R
    H_2_011 = (temp_011*(temp_sum >= 1)).sum()/R

    H_1_100 = (temp_100*(temp_sum == 1)).sum()/R
    H_2_100 = (temp_100*(temp_sum >= 1)).sum()/R

    H_1_101 = (temp_101*(temp_sum == 1)).sum()/R
    H_2_101 = (temp_101*(temp_sum >= 1)).sum()/R

    H_1_110 = (temp_110*(temp_sum == 1)).sum()/R
    H_2_110 = (temp_110*(temp_sum >= 1)).sum()/R

    H_1_111 = (temp_111*(temp_sum == 1)).sum()/R
    H_2_111 = (temp_111*(temp_sum >= 1)).sum()/R

    return H_1_000, H_1_001, H_1_010, H_1_011, H_1_100, H_1_101, H_1_110, H_1_111, H_2_000, H_2_001, H_2_010, H_2_011, H_2_100, H_2_101, H_2_110, H_2_111

# calculate bound for 4firm case
def H_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, delta, mu, sigma):
    R = 500
    np.random.seed(41)
    cost_shock_firm1 = np.random.normal(mu, sigma, R)
    np.random.seed(42)
    cost_shock_firm2 = np.random.normal(mu, sigma, R)
    np.random.seed(43)
    cost_shock_firm3 = np.random.normal(mu, sigma, R)
    np.random.seed(44)
    cost_shock_firm4 = np.random.normal(mu, sigma, R)
    
    def temp_calculate(firm1_decision, firm2_decision, firm3_decision, firm4_decision):

        decision_array = np.array([firm1_decision, firm2_decision, firm3_decision, firm4_decision])
        entrants_num = decision_array.sum()

        #firm1
        if firm1_decision == 0:
            firm1_inside_option = np.zeros(R)
            firm1_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_1m + cost_shock_firm1)
        elif firm1_decision == 1: 
            firm1_inside_option = X_m - delta * np.log(entrants_num) - (Z_1m + cost_shock_firm1)
            firm1_outside_option = np.zeros(R)
        
        #firm2
        if firm2_decision == 0:
            firm2_inside_option = np.zeros(R)
            firm2_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_2m + cost_shock_firm2)
        elif firm2_decision == 1: 
            firm2_inside_option = X_m - delta * np.log(entrants_num) - (Z_2m + cost_shock_firm2)
            firm2_outside_option = np.zeros(R)

        #firm3
        if firm3_decision == 0:
            firm3_inside_option = np.zeros(R)
            firm3_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_3m + cost_shock_firm3)
        elif firm3_decision == 1: 
            firm3_inside_option = X_m - delta * np.log(entrants_num) - (Z_3m + cost_shock_firm3)
            firm3_outside_option = np.zeros(R)

        #firm4
        if firm4_decision == 0:
            firm4_inside_option = np.zeros(R)
            firm4_outside_option = X_m - delta * np.log(entrants_num+1) - (Z_4m + cost_shock_firm4)
        elif firm4_decision == 1: 
            firm4_inside_option = X_m - delta * np.log(entrants_num) - (Z_4m + cost_shock_firm4)
            firm4_outside_option = np.zeros(R)
        
        firm1_profit_bool = firm1_inside_option >= firm1_outside_option
        firm2_profit_bool = firm2_inside_option >= firm2_outside_option
        firm3_profit_bool = firm3_inside_option >= firm3_outside_option
        firm4_profit_bool = firm4_inside_option >= firm4_outside_option

        return (firm1_profit_bool * firm2_profit_bool * firm3_profit_bool * firm4_profit_bool).astype(int)
    
    temp_0000 = temp_calculate(0,0,0,0)
    temp_0001 = temp_calculate(0,0,0,1)
    temp_0010 = temp_calculate(0,0,1,0)
    temp_0011 = temp_calculate(0,0,1,1)
    temp_0100 = temp_calculate(0,1,0,0)
    temp_0101 = temp_calculate(0,1,0,1)
    temp_0110 = temp_calculate(0,1,1,0)
    temp_0111 = temp_calculate(0,1,1,1)
    temp_1000 = temp_calculate(1,0,0,0)
    temp_1001 = temp_calculate(1,0,0,1)
    temp_1010 = temp_calculate(1,0,1,0)
    temp_1011 = temp_calculate(1,0,1,1)
    temp_1100 = temp_calculate(1,1,0,0)
    temp_1101 = temp_calculate(1,1,0,1)
    temp_1110 = temp_calculate(1,1,1,0)
    temp_1111 = temp_calculate(1,1,1,1)
    temp_sum = temp_0000 + temp_0001 + temp_0010 + temp_0011 \
            + temp_0100 + temp_0101 + temp_0110 + temp_0111 \
            + temp_1000 + temp_1001 + temp_1010 + temp_1011 \
            + temp_1100 + temp_1101 + temp_1110 + temp_1111

    H_1_0000 = (temp_0000*(temp_sum == 1)).sum()/R
    H_2_0000 = (temp_0000*(temp_sum >= 1)).sum()/R

    H_1_0001 = (temp_0001*(temp_sum == 1)).sum()/R
    H_2_0001 = (temp_0001*(temp_sum >= 1)).sum()/R

    H_1_0010 = (temp_0010*(temp_sum == 1)).sum()/R
    H_2_0010 = (temp_0010*(temp_sum >= 1)).sum()/R

    H_1_0011 = (temp_0011*(temp_sum == 1)).sum()/R
    H_2_0011 = (temp_0011*(temp_sum >= 1)).sum()/R

    H_1_0100 = (temp_0100*(temp_sum == 1)).sum()/R
    H_2_0100 = (temp_0100*(temp_sum >= 1)).sum()/R

    H_1_0101 = (temp_0101*(temp_sum == 1)).sum()/R
    H_2_0101 = (temp_0101*(temp_sum >= 1)).sum()/R

    H_1_0110 = (temp_0110*(temp_sum == 1)).sum()/R
    H_2_0110 = (temp_0110*(temp_sum >= 1)).sum()/R

    H_1_0111 = (temp_0111*(temp_sum == 1)).sum()/R
    H_2_0111 = (temp_0111*(temp_sum >= 1)).sum()/R

    H_1_1000 = (temp_1000*(temp_sum == 1)).sum()/R
    H_2_1000 = (temp_1000*(temp_sum >= 1)).sum()/R

    H_1_1001 = (temp_1001*(temp_sum == 1)).sum()/R
    H_2_1001 = (temp_1001*(temp_sum >= 1)).sum()/R

    H_1_1010 = (temp_1010*(temp_sum == 1)).sum()/R
    H_2_1010 = (temp_1010*(temp_sum >= 1)).sum()/R

    H_1_1011 = (temp_1011*(temp_sum == 1)).sum()/R
    H_2_1011 = (temp_1011*(temp_sum >= 1)).sum()/R

    H_1_1100 = (temp_1100*(temp_sum == 1)).sum()/R
    H_2_1100 = (temp_1100*(temp_sum >= 1)).sum()/R

    H_1_1101 = (temp_1101*(temp_sum == 1)).sum()/R
    H_2_1101 = (temp_1101*(temp_sum >= 1)).sum()/R

    H_1_1110 = (temp_1110*(temp_sum == 1)).sum()/R
    H_2_1110 = (temp_1110*(temp_sum >= 1)).sum()/R

    H_1_1111 = (temp_1111*(temp_sum == 1)).sum()/R
    H_2_1111 = (temp_1111*(temp_sum >= 1)).sum()/R

    return H_1_0000, H_1_0001, H_1_0010, H_1_0011, H_1_0100, H_1_0101, H_1_0110, H_1_0111, \
            H_1_1000, H_1_1001, H_1_1010, H_1_1011, H_1_1100, H_1_1101, H_1_1110, H_1_1111,\
            H_2_0000, H_2_0001, H_2_0010, H_2_0011, H_2_0100, H_2_0101, H_2_0110, H_2_0111, \
            H_2_1000, H_2_1001, H_2_1010, H_2_1011, H_2_1100, H_2_1101, H_2_1110, H_2_1111

# objective function
def min_obj(theta, df_2firms, df_3firms, df_4firms, freq_2firms_dict, freq_3firms_dict, freq_4firms_dict):
    # kill sigma
    # will come back later 
    delta, mu, sigma = theta
    market_number = df_2firms.shape[0] + df_3firms.shape[0] + df_4firms.shape[0]

    #calculate 2 firm case
    def Qn_2firms(X_m, Z_1m, Z_2m, bin_num):
        Freq_Est = freq_2firms_dict[bin_num]
        #Freq_Est = kde_freq_est_2firms(X_m, Z_1m, Z_2m, df_2firms)
        H_LB = np.array(H_2firms(X_m, Z_1m, Z_2m, delta, mu, sigma))[:4]
        H_UB = np.array(H_2firms(X_m, Z_1m, Z_2m, delta, mu, sigma))[4:]

        #lower bound
        LB = Freq_Est - H_LB
        LB_bool = LB < 0
        LB_temp = LB * LB_bool.astype(int)
        LB_dist = np.sqrt(LB_temp@LB_temp)

        #upper bound
        UB = Freq_Est - H_UB
        UB_bool = UB > 0
        UB_temp = UB * UB_bool.astype(int)
        UB_dist = np.sqrt(UB_temp@UB_temp)

        return LB_dist + UB_dist
   
    def Qn_3firms(X_m, Z_1m, Z_2m, Z_3m, bin_num):
        Freq_Est = freq_3firms_dict[bin_num]
        #Freq_Est = kde_freq_est_3firms(X_m, Z_1m, Z_2m, Z_3m, df_3firms)  
        H_LB = np.array(H_3firms(X_m, Z_1m, Z_2m, Z_3m, delta, mu, sigma))[:8]
        H_UB = np.array(H_3firms(X_m, Z_1m, Z_2m, Z_3m, delta, mu, sigma))[8:]

        #lower bound
        LB = Freq_Est - H_LB
        LB_bool = LB < 0
        LB_temp = LB * LB_bool.astype(int)
        LB_dist = np.sqrt(LB_temp@LB_temp)

        #upper bound
        UB = Freq_Est - H_UB
        UB_bool = UB > 0
        UB_temp = UB * UB_bool.astype(int)
        UB_dist = np.sqrt(UB_temp@UB_temp)

        return LB_dist + UB_dist

    def Qn_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, bin_num):
        Freq_Est = freq_4firms_dict[bin_num]
        #Freq_Est = kde_freq_est_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, df_4firms)
        H_LB = np.array(H_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, delta, mu, sigma))[:16]
        H_UB = np.array(H_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, delta, mu, sigma))[16:]

        #lower bound
        LB = Freq_Est - H_LB
        LB_bool = LB < 0
        LB_temp = LB * LB_bool.astype(int)
        LB_dist = np.sqrt(LB_temp@LB_temp)

        #upper bound
        UB = Freq_Est - H_UB
        UB_bool = UB > 0
        UB_temp = UB * UB_bool.astype(int)
        UB_dist = np.sqrt(UB_temp@UB_temp)

        return LB_dist + UB_dist


    vec_Qn_2firms = np.vectorize(Qn_2firms)
    vec_Qn_3firms = np.vectorize(Qn_3firms)
    vec_Qn_4firms = np.vectorize(Qn_4firms)

    Qn_array_2 = vec_Qn_2firms(df_2firms['X_m'].values, df_2firms['Z_fm_1'].values, df_2firms['Z_fm_2'].values, df_2firms['bin_num'].values)
    Qn_array_3 = vec_Qn_3firms(df_3firms['X_m'].values, df_3firms['Z_fm_1'].values, df_3firms['Z_fm_2'].values, df_3firms['Z_fm_3'].values, df_3firms['bin_num'].values)
    Qn_array_4 = vec_Qn_4firms(df_4firms['X_m'].values, df_4firms['Z_fm_1'].values, df_4firms['Z_fm_2'].values, df_4firms['Z_fm_3'].values, df_4firms['Z_fm_4'].values, df_4firms['bin_num'].values)

    Qn_sum = Qn_array_2.sum() + Qn_array_3.sum() + Qn_array_4.sum()
    Qn_mean = Qn_sum/market_number
    
    return Qn_mean

# objective function parallelized
@ray.remote
def min_obj_ray(theta, df_2firms, df_3firms, df_4firms, freq_2firms_dict, freq_3firms_dict, freq_4firms_dict):
    delta, mu, sigma = theta
    #sigma = 1
    market_number = df_2firms.shape[0] + df_3firms.shape[0] + df_4firms.shape[0]

    #calculate 2 firm case
    def Qn_2firms(X_m, Z_1m, Z_2m, bin_num):
        Freq_Est = freq_2firms_dict[bin_num]
        #Freq_Est = kde_freq_est_2firms(X_m, Z_1m, Z_2m, df_2firms)
        H_LB = np.array(H_2firms(X_m, Z_1m, Z_2m, delta, mu, sigma))[:4]
        H_UB = np.array(H_2firms(X_m, Z_1m, Z_2m, delta, mu, sigma))[4:]

        #lower bound
        LB = Freq_Est - H_LB
        LB_bool = LB < 0
        LB_temp = LB * LB_bool.astype(int)
        LB_dist = np.sqrt(LB_temp@LB_temp)

        #upper bound
        UB = Freq_Est - H_UB
        UB_bool = UB > 0
        UB_temp = UB * UB_bool.astype(int)
        UB_dist = np.sqrt(UB_temp@UB_temp)

        return LB_dist + UB_dist
   
    def Qn_3firms(X_m, Z_1m, Z_2m, Z_3m, bin_num):
        Freq_Est = freq_3firms_dict[bin_num]
        #Freq_Est = kde_freq_est_3firms(X_m, Z_1m, Z_2m, Z_3m, df_3firms)  
        H_LB = np.array(H_3firms(X_m, Z_1m, Z_2m, Z_3m, delta, mu, sigma))[:8]
        H_UB = np.array(H_3firms(X_m, Z_1m, Z_2m, Z_3m, delta, mu, sigma))[8:]

        #lower bound
        LB = Freq_Est - H_LB
        LB_bool = LB < 0
        LB_temp = LB * LB_bool.astype(int)
        LB_dist = np.sqrt(LB_temp@LB_temp)

        #upper bound
        UB = Freq_Est - H_UB
        UB_bool = UB > 0
        UB_temp = UB * UB_bool.astype(int)
        UB_dist = np.sqrt(UB_temp@UB_temp)

        return LB_dist + UB_dist

    def Qn_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, bin_num):
        Freq_Est = freq_4firms_dict[bin_num]
        #Freq_Est = kde_freq_est_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, df_4firms)
        H_LB = np.array(H_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, delta, mu, sigma))[:16]
        H_UB = np.array(H_4firms(X_m, Z_1m, Z_2m, Z_3m, Z_4m, delta, mu, sigma))[16:]

        #lower bound
        LB = Freq_Est - H_LB
        LB_bool = LB < 0
        LB_temp = LB * LB_bool.astype(int)
        LB_dist = np.sqrt(LB_temp@LB_temp)

        #upper bound
        UB = Freq_Est - H_UB
        UB_bool = UB > 0
        UB_temp = UB * UB_bool.astype(int)
        UB_dist = np.sqrt(UB_temp@UB_temp)

        return LB_dist + UB_dist


    vec_Qn_2firms = np.vectorize(Qn_2firms)
    vec_Qn_3firms = np.vectorize(Qn_3firms)
    vec_Qn_4firms = np.vectorize(Qn_4firms)



    Qn_array_2 = vec_Qn_2firms(df_2firms['X_m'].values, df_2firms['Z_fm_1'].values, df_2firms['Z_fm_2'].values, df_2firms['bin_num'].values)
    Qn_array_3 = vec_Qn_3firms(df_3firms['X_m'].values, df_3firms['Z_fm_1'].values, df_3firms['Z_fm_2'].values, df_3firms['Z_fm_3'].values, df_3firms['bin_num'].values)
    if df_4firms.shape[0] == 0:
        Qn_array_4 = 0
    else:
        Qn_array_4 = vec_Qn_4firms(df_4firms['X_m'].values, df_4firms['Z_fm_1'].values, df_4firms['Z_fm_2'].values, df_4firms['Z_fm_3'].values, df_4firms['Z_fm_4'].values, df_4firms['bin_num'].values)

    Qn_sum = Qn_array_2.sum() + Qn_array_3.sum() + Qn_array_4.sum()
    Qn_mean = Qn_sum/market_number
    
    return Qn_mean



# bootstrap to update cut-off level
@ray.remote
def mieq_bootstrap(j, b, df_master, parameter_space_c1, market_id_list):

    pd.set_option('mode.chained_assignment',  None)
    market_to_sample = random.sample(market_id_list, b)
    data_bootstrap = df_master.loc[df_master['market_id'].isin(market_to_sample)]
    
    df_2firms = df_reshape(data_bootstrap, 2)
    df_3firms = df_reshape(data_bootstrap, 3)
    df_4firms = df_reshape(data_bootstrap, 4)

    df_2firms = bin_number_2firms(df_2firms)
    df_3firms = bin_number_3firms(df_3firms)
    df_4firms = bin_number_4firms(df_4firms)

    freq_2firms_dict = {}
    for bin_num in df_2firms['bin_num'].unique():
        freq_2firms_dict[bin_num] = Freq_Est_2firms(df_2firms, bin_num)

    freq_3firms_dict = {}
    for bin_num in df_3firms['bin_num'].unique():
        freq_3firms_dict[bin_num] = Freq_Est_3firms(df_3firms, bin_num)

    freq_4firms_dict = {}
    for bin_num in df_4firms['bin_num'].unique():
        freq_4firms_dict[bin_num] = Freq_Est_4firms(df_4firms, bin_num)
    
    # calculate global minimum
    bnds = ((0, None), (0, None), (0, None))
    Qn_result = minimize(min_obj, [0.8, 1.2, 0.8], args = (df_2firms, df_3firms, df_4firms, freq_2firms_dict, freq_3firms_dict, freq_4firms_dict), bounds =bnds, method='Nelder-Mead', options={'maxiter':300})
    Q_bootstrap_min = Qn_result.fun
    
    # calculate Qn values on the parameter space
    Q_n_values = [min_obj(parameter, df_2firms, df_3firms, df_4firms, freq_2firms_dict, freq_3firms_dict, freq_4firms_dict)for parameter in parameter_space_c1]
    Q_n_values = np.array(Q_n_values)
    
    b_max = (Q_n_values - Q_bootstrap_min).max()
    return b*b_max

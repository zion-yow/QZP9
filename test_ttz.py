import double_check
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
import engine
from scipy.stats import expon
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import time
import random
   
# N = 11
BOUNDS = [
    (1, 100000),  # 0: monthly_total_bet
    (1, 100000),  # 1: monthly_total_pay
    (1, 100000),     # 2: cycle_total_bet
    (1, 100000),     # 3: cycle_total_pay
    (-3, 3),         # 4: is_Rookie_table
    (-3, 3),         # 5: monthly_RTP_cap_enabled
    (1, 100000),     # 6: monthly_RTP_cap
    (-3, 3),         # 7: monthly_loss_cap_enabled
    (1, 100000),     # 8: monthly_loss_cap
    (-3, 3),         # 9: monthly_player_profit_cap_enabled
    (1, 100000),     # 10: monthly_player_profit_cap
]

CONS_NUM = 6
SAMPLE_NUM = 1

def objective(x):
    # 任意常数目标函数
    return 0

def round_discrete_params(x):
    """將特定參數四捨五入到整數"""
    x_rounded = x.copy()

    # monthly/cycle/daily total bet/pay 四捨五入到整數
    x_rounded[0:4] = np.round(x_rounded[0:4])
    
    # RTP相關參數四捨五入到整數
    x_rounded[5] = np.round(x_rounded[5])
    x_rounded[6] = np.round(x_rounded[6])
    x_rounded[8] = np.round(x_rounded[8])
    x_rounded[10] = np.round(x_rounded[10])
    
    # 開關參數強制為-1或1
    x_rounded[[7,9,11]] = np.round(x_rounded[[7,9,11]])

    return x_rounded

def objective_discrete(x):
    """包裝目標函數，處理離散參數"""
    x_rounded = round_discrete_params(x)
    return objective(x_rounded)

# 修改constraint_wrapper为一个类
class ConstraintWrapper:
    def __init__(self, constraint_func):
        self.constraint_func = constraint_func
    
    def __call__(self, x):
        x_rounded = round_discrete_params(x)
        return self.constraint_func(x_rounded)


#--------------------------------
# 中間變量計算模塊
#--------------------------------


# 周期系統RTP
def constraint_monthly_rtp(x):

    # 避免除以零
    denominator = x[0] + x[2]
    monthly_cycle_RTP = (x[1] + x[3]) / denominator * 10000 if denominator > 0 else 0
    return monthly_cycle_RTP

# 當月系統虧損
def constraint_monthly_loss(x):
    monthly_total_pay = x[1] + x[3]
    monthly_total_bet = x[0] + x[2]
    
    monthly_loss = monthly_total_pay - monthly_total_bet 
    return monthly_loss

# 當月個人盈利
def constraint_monthly_player_profit(x):
    monthly_total_pay = x[1] + x[3]
    monthly_total_bet = x[0] + x[2]
    
    monthly_player_profit = -monthly_total_pay + monthly_total_bet 
    return monthly_player_profit


#--------------------------------
# 判斷模塊
#--------------------------------

# 當月系統RTP 與 當月RTP上限比較 
def monthly_rtp_cap_comparation(x):
    monthly_RTP = constraint_monthly_rtp(x)
    monthly_RTP_cap = x[6]
    
    return monthly_RTP < monthly_RTP_cap

# 當月系統虧損 與 當月虧損上限比較
def monthly_loss_cap_comparation(x):
    monthly_loss = constraint_monthly_loss(x)
    monthly_loss_cap = x[8]
    
    return monthly_loss < monthly_loss_cap

# 當月個人盈利 與 當月個人盈利上限比較
def monthly_player_profit_cap_comparation(x):
    monthly_player_profit = constraint_monthly_player_profit(x)
    monthly_player_profit_cap = x[10]
    
    return monthly_player_profit < monthly_player_profit_cap
    

def zero_judge(c):
    if c == 0:
        return 1
    else:
        return -1
    
def full_judge(c):

    if c == 8:
        return 1
    else:
        return -1


def sign_judge(c1, c2):
    if c1 == 1 and c2 == 1:
        return 1
    else:
        return -1


def generate_exponential_sum_integers(n_elements=8, target_sum=REVIEWS, lambda_param=REVIEWS/8):
    """
    生成n個元素的列表，元素和為target_sum，且符合指數分佈

    參數:
    lambda_param: 指數分佈參數，越大則小值出現機率越高
    """
    # 生成原始指數分佈樣本
    raw_samples = expon.rvs(scale=1/lambda_param, size=n_elements)
    
    # 確保非負
    raw_samples = np.maximum(raw_samples, 0)
    
    scaled_samples = (raw_samples / np.sum(raw_samples)) * target_sum
    # 向下取整
    integer_samples = np.floor(scaled_samples).astype(int)

    # 根據小數部分大小排序，將剩餘值分配給小數部分最大的元素
    remaining = target_sum - np.sum(integer_samples)
    if remaining > 0:
        decimal_parts = scaled_samples - integer_samples
        indices = np.argsort(decimal_parts)[::-1]
        for i in range(int(remaining)):
            integer_samples[indices[i]] += 1

    return integer_samples


# 0. 當月RTP上限功能開啟
def constraint0(x):
    return x[5]

# 1. 當月RTP上限功能未開啟
def constraint1(x):
    return -x[5]

#2.  當前周期RTP低於當月系統RTP
def constraint2(x):
    return constraint_rtp(x)

#3.  當前周期RTP高於等於當月系統RTP
def constraint3(x):
    return -constraint_rtp(x)

#4. 當月系統虧損上限開啓
def constraint4(x):
    return x[7]

#5. 當月系統虧損上限未開啓
def constraint5(x):
    return -x[7]

#6. 當月系統虧損低於當月虧損上限
def constraint6(x):
    return constraint_monthly_loss(x)

#7. 當月系統虧損高於等於當月虧損上限
def constraint7(x):
    return -constraint_monthly_loss(x)

#8. 當月個人盈利上限開啓
def constraint8(x):
    return x[9]

#9. 當月個人盈利上限未開啓
def constraint9(x):
    return -x[9]

#10. 當月個人盈利低於當月個人盈利上限
def constraint10(x):
    return constraint_monthly_player_profit(x)

#11. 當月個人盈利高於等於當月個人盈利上限
def constraint11(x):
    return -constraint_monthly_player_profit(x)



CON_DCT = {
    '當月RTP上限功能有開啓': constraint0,
    '當月RTP上限功能未開啓': constraint1,
    '當前周期RTP低於當月系統RTP': constraint2,
    '當前周期RTP高於等於當月系統RTP': constraint3,
    '當月系統虧損上限開啓': constraint4,
    '當月系統虧損上限未開啓': constraint5,
    '當月系統虧損低於當月虧損上限': constraint6,
    '當月系統虧損高於等於當月虧損上限': constraint7,
    '當月個人盈利上限開啓': constraint8,
    '當月個人盈利上限未開啓': constraint9,
    '當月個人盈利低於當月個人盈利上限': constraint10,
    '當月個人盈利高於等於當月個人盈利上限': constraint11,
}


def make_constraint(fun):
    """将函数转换为NonlinearConstraint格式"""
    return NonlinearConstraint(ConstraintWrapper(fun), 0.001, np.inf)

# 保存字典
def save_dict_numpy(dictionary, filename):
    np.save(filename, dictionary)

# 讀取字典
def load_dict_numpy(filename):
    return np.load(filename, allow_pickle=True).item()

LAYER1 = (make_constraint(constraint0), make_constraint(constraint1))
LAYER2 = (make_constraint(constraint2), make_constraint(constraint3))
LAYER3 = (make_constraint(constraint4), make_constraint(constraint5))
LAYER4 = (make_constraint(constraint6), make_constraint(constraint7))
LAYER5 = (make_constraint(constraint8), make_constraint(constraint9))
LAYER6 = (make_constraint(constraint10), make_constraint(constraint11))

TYPE_LAYER1 = ('當月RTP上限功能有開啓', '當月RTP上限功能未開啓')
TYPE_LAYER2 = ('當前周期RTP低於當月系統RTP', '當前周期RTP高於等於當月系統RTP')
TYPE_LAYER3 = ('當月系統虧損上限開啓', '當月系統虧損上限未開啓')
TYPE_LAYER4 = ('當月系統虧損低於當月虧損上限', '當月系統虧損高於等於當月虧損上限')
TYPE_LAYER5 = ('當月個人盈利上限開啓', '當月個人盈利上限未開啓')
TYPE_LAYER6 = ('當月個人盈利低於當月個人盈利上限', '當月個人盈利高於等於當月個人盈利上限')


CONSTRAINT_MAP = {
        TYPE_LAYER1[0]: LAYER1[0], TYPE_LAYER1[1]: LAYER1[1],
        TYPE_LAYER2[0]: LAYER2[0], TYPE_LAYER2[1]: LAYER2[1],
        TYPE_LAYER3[0]: LAYER3[0], TYPE_LAYER3[1]: LAYER3[1],
        TYPE_LAYER4[0]: LAYER4[0], TYPE_LAYER4[1]: LAYER4[1],
        TYPE_LAYER5[0]: LAYER5[0], TYPE_LAYER5[1]: LAYER5[1],
        TYPE_LAYER6[0]: LAYER6[0], TYPE_LAYER6[1]: LAYER6[1],
    }


# 一組條件約束下生成參數
def process_single_config(args):
    config_counts, _type = args
    _con = [CONSTRAINT_MAP[t] for t in _type[:CONS_NUM]]

    para_dct = {}
    para_dct[_type[:CONS_NUM]] = {}
    
    for _spl in range(SAMPLE_NUM):
        _spl += 0
        result = differential_evolution(
            func=objective_discrete,
            bounds=BOUNDS,
            constraints=_con[:CONS_NUM],
            maxiter=1000,
            popsize=15,
            mutation=(0.5, 1.9),
            recombination=0.9,
            seed=1,
            strategy='best1bin',
            updating='deferred',
            tol=1e-5,
            polish=False,
            disp=False
        )

        final_solution = round_discrete_params(result.x)


        parameters = { 
            "monthly_total_bet": final_solution[0],            
            "monthly_total_pay": final_solution[1],     
            "cycle_total_bet": final_solution[2],            
            "cycle_total_pay": final_solution[3],            
            "is_Rookie_table": True if final_solution[4] > 0 else False,                
            "monthly_RTP_cap_enabled": True if final_solution[5] > 0 else False,     
            "monthly_RTP_cap": final_solution[6],                            
            "monthly_loss_cap_enabled": True if final_solution[7] > 0 else False,     
            "monthly_loss_cap": final_solution[8],                 
            "monthly_player_profit_cap_enabled": True if final_solution[9] > 0 else False,     
            "monthly_player_profit_cap": final_solution[10],                 
        }
        
        double_check.single_double_check(r'F:/projects/QZP9/Validation/log/', f'config_{config_counts}_{_spl}.log', parameters, config_counts, _spl+1, final_solution, str(_type)[1:-1])
    
    return para_dct


# 主函數
def main():
    # 生成类型组合
    type_combinations = list(itertools.product(
        TYPE_LAYER1, TYPE_LAYER2, TYPE_LAYER3, TYPE_LAYER4, 
        TYPE_LAYER5, TYPE_LAYER6
    ))
    
    # 准备并发处理的参数
    args_list = [(i+1, type_) for i, type_ in enumerate(type_combinations)]

    # 是否并發執行
    if_parallel = input('是否并發執行？(y/n)')
    if if_parallel == 'y':
        # 使用ProcessPoolExecutor进行并发处理
        max_workers = 4  # 预留一些CPU核心
        final_para_dct = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single_config, args_list[:]), total=len(args_list[:])))
        
         # 合并所有结果
        for result in results:
            final_para_dct.update(result)
    else:
        for i in tqdm(range(len(type_combinations[0:]))):
            final_para_dct = process_single_config((i+1, type_combinations[0:][i]))
            
    return final_para_dct




if __name__ == "__main__":
    main()

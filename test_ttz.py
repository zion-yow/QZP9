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

# 8种情况
BANKER_PAY_MATRIX = np.array([
            [-1.95,-1.95,-1.95,-1.95,-1.95,-1.95],
            [-1.95,-1.95,0,-1.95,-1,-1],
            [-1.95,0,-1.95,-1,-1.95,-1],
            [-1.95,0,0,-1,-1,0],
            [0,-1.95,-1.95,-1,-1,-1.95],
            [0,-1.95,0,-1,0,-1],
            [0,0,-1.95,0,-1,-1],
            [0,0,0,0,0,0],
        ])

REVIEWS = 17       
# N = 39
BOUNDS = [
    (0, 10),    # 0-3: region_bet
    (0, 10),
    (0, 10),
    (0, 10),
    (0, 10),
    (0, 10),
    (1, 100000),  # 6: monthly_total_bet
    (1, 100000),  # 7: monthly_total_pay
    (1, 100000),     # 8: cycle_total_bet
    (1, 100000),     # 9: cycle_total_pay
    (1, 100000),     # 10: daily_total_bet
    (1, 100000),     # 11: daily_total_pay
    (1, 100000),     # 12: expected_RTP
    (-3, 3),            # 13: monthly_RTP_cap_enabled
    (1, 100000),     # 14: monthly_RTP_cap
    (-3, 3),            # 15: monthly_loss_cap_enabled
    (1, 100000),# 16: monthly_loss_cap
    (-3, 3),            # 17: daily_loss_cap_enabled
    (1, 100000), # 18: daily_loss_cap
    (-10, 10),         # 19-22: consecutive_counts
    (-10, 10),
    (-10, 10),
    # 理論上是（0，17），但求解用時太長
    (0, REVIEWS),      # 22-29: pass_n_turns_distribution
    (0, REVIEWS),
    (0, REVIEWS),
    (0, REVIEWS),
    (0, REVIEWS),
    (0, REVIEWS),
    (0, REVIEWS),
    (0, REVIEWS),
    # (1, 100000),    # 30: current_cycle_RTP
]

CONS_NUM = 8
SAMPLE_NUM = 1

def objective(x):
    # 任意常数目标函数
    return 0

def round_discrete_params(x):
    """將特定參數四捨五入到整數"""
    x_rounded = x.copy()
    
    # region_bet 四捨五入到整數
    x_rounded[0:6] = np.round(x_rounded[0:6])
    
    # monthly/cycle/daily total bet/pay 四捨五入到整數
    x_rounded[6:12] = np.round(x_rounded[6:12])
    
    # RTP相關參數四捨五入到整數
    x_rounded[12] = np.round(x_rounded[12])
    x_rounded[14] = np.round(x_rounded[14])
    
    # 開關參數強制為-1或1
    x_rounded[[13,15,17]] = np.round(x_rounded[[13,15,17]])
    
    # loss cap 四捨五入到整數
    x_rounded[16] = np.round(x_rounded[16])
    x_rounded[18] = np.round(x_rounded[18])
    
    # consecutive_counts 四捨五入到整數，不能取零
    x_rounded[19:22] = np.round(x_rounded[19:22])
    x_rounded[19:22] = np.where(x_rounded[19:22] == 0, 1, x_rounded[19:22])

    # pass_n_turns_distribution 四捨五入到整數
    x_rounded[22:30] = np.round(x_rounded[22:30])
    
    # current_cycle_RTP 四捨五入到整數
    # x_rounded[30] = np.round(x_rounded[30])
    
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
def constraint_rtp(x):
    banker_pay_clm = np.tile(np.array([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1)) * BANKER_PAY_MATRIX
    banker_pay_array = -np.sum(banker_pay_clm,axis=1).T.reshape(8,1)

    cycle_total_pay_array = x[9]* np.ones((8,1))
    cycle_total_bet_array = x[8]* np.ones((8,1))
    total_bet_array = np.sum(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) * np.ones((8,1))
    
    # 避免除以零
    denominator = cycle_total_bet_array + total_bet_array
    cycle_RTP_array = np.where(
        denominator > 0,
        (cycle_total_pay_array + banker_pay_array) / denominator * 10000,
        0
    )
    return cycle_RTP_array

# 當月系統RTP
def constraint_monthly_rtp(x):
    banker_pay_clm = np.tile(np.array([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1)) * BANKER_PAY_MATRIX
    banker_pay_array = -np.sum(banker_pay_clm,axis=1).T.reshape(8,1)

    monthly_total_pay_array = x[7]* np.ones((8,1))
    monthly_total_bet_array = x[6]* np.ones((8,1))
    total_bet_array = np.sum(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) * np.ones((8,1))
    
    # 避免除以零
    denominator = monthly_total_bet_array + total_bet_array
    monthly_cycle_RTP_array = np.where(
        denominator > 0,
        (monthly_total_pay_array + banker_pay_array) / denominator * 10000,
        0
    )
    return monthly_cycle_RTP_array

# 當月系統虧損
def constraint_monthly_loss(x):
    banker_pay_clm = np.tile(np.array([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1)) * BANKER_PAY_MATRIX
    banker_pay_array = -np.sum(banker_pay_clm,axis=1).T.reshape(8,1)

    banker_profit_array = banker_pay_array + (np.sum(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) * np.ones((8,1)))

    monthly_total_pay_array = x[7]* np.ones((8,1))
    monthly_total_bet_array = x[6]* np.ones((8,1))
    
    monthly_loss_array = monthly_total_pay_array - monthly_total_bet_array - banker_profit_array
    # display(monthly_loss_array)
    return monthly_loss_array

# 當日系統虧損
def constraint_daily_loss(x):
    banker_pay_clm = np.tile(np.array([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1)) * BANKER_PAY_MATRIX
    banker_pay_array = -np.sum(banker_pay_clm,axis=1).T.reshape(8,1)

    banker_profit_array = banker_pay_array + (np.sum(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) * np.ones((8,1)))

    daily_total_pay_array = x[11]* np.ones((8,1))
    daily_total_bet_array = x[10]* np.ones((8,1))
    
    daily_loss_array = daily_total_pay_array - daily_total_bet_array - banker_profit_array
    # display(daily_loss_array)
    return daily_loss_array

# 系統派彩
def constraint_banker_pay(x):
    banker_pay_clm = np.tile(np.array([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1)) * BANKER_PAY_MATRIX
    return banker_pay_clm

# 系統收益
def constraint_banker_profit(x):
    banker_pay_clm = np.tile(np.array([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1)) * BANKER_PAY_MATRIX
    banker_pay_array = np.sum(banker_pay_clm,axis=1).T.reshape(8,1)

    banker_profit_array = banker_pay_array + np.tile(np.sum([x[0],x[1],x[2],x[3],x[4],x[5]]) ,(8,1))
    return banker_profit_array


#--------------------------------
# 判斷模塊
#--------------------------------

# 當周期系統RTP 與 當前周期RTP比較
def count_cycle_rtp_current_comparation(x):
    cycle_RTP_array = constraint_rtp(x)
    _cpr = int(10000 * x[9]/x[8])
    current_cycle_RTP_array = np.tile(_cpr,(8,1))
    
    counter = np.sum(np.where(cycle_RTP_array < current_cycle_RTP_array,1,0))
    return counter

# 當月系統RTP 與 期望RTP比較
def count_monthly_rtp_exp_comparation(x):
    monthly_RTP_array = constraint_monthly_rtp(x)
    expected_RTP_array = np.tile(x[12],(8,1))
    
    counter = np.sum(np.where(monthly_RTP_array < expected_RTP_array,1,0))
    return counter

# 當月系統RTP 與 當月RTP上限比較 
def count_monthly_rtp_cap_comparation(x):
    monthly_RTP_array = constraint_monthly_rtp(x)
    monthly_RTP_cap_array = np.tile(x[14],(8,1))
    
    counter = np.sum(np.where(monthly_RTP_array < monthly_RTP_cap_array,1,0))
    return counter

# 當月系統虧損 與 當月虧損上限比較
def count_monthly_loss_cap_comparation(x):
    monthly_loss_array = constraint_monthly_loss(x)
    monthly_loss_cap_array = np.tile(x[16],(8,1))
    
    counter = np.sum(np.where(monthly_loss_array < monthly_loss_cap_array,1,0))
    return counter

# 當日系統虧損 與 當日系統虧損上限比較
def count_daily_loss_cap_comparation(x):
    daily_loss_array = constraint_daily_loss(x)
    daily_loss_cap_array = np.tile(x[18],(8,1))
    
    counter = np.sum(np.where(daily_loss_array < daily_loss_cap_array,1,0))
    return counter

# 系統收益全相等
def count_banker_profit_equal(x):

    banker_profit_array = constraint_banker_profit(x)
    # display(banker_profit_array)
    counter = np.sum(np.where(banker_profit_array == banker_profit_array[0],1,0))
    return counter

# 系統收益全為正
def count_banker_profit_positive(x):
    banker_profit_array = constraint_banker_profit(x)
    # display(banker_profit_array)
    counter = np.sum(np.where(banker_profit_array >= 0,1,0))
    return counter

# 系統派彩全部相等
def count_banker_pay_equal(x):
    banker_pay_array = constraint_banker_pay(x)

    counter = np.sum(np.where(banker_pay_array == banker_pay_array[0],1,0))
    return counter

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
    return x[13]

# 1. 當月RTP上限功能為開啟
def constraint1(x):
    return -x[13]

#2.  全部周期RTP高於等於當前周期RTP
def constraint2(x):
    return zero_judge(count_cycle_rtp_current_comparation(x)) 

#3.  全部周期RTP低於等於當前周期RTP
def constraint3(x):
    return full_judge(count_cycle_rtp_current_comparation(x)) 
    
#4.  -#0 & -#1
def constraint4(x):
    return sign_judge(-zero_judge(count_cycle_rtp_current_comparation(x)),-full_judge(count_cycle_rtp_current_comparation(x)))

#5. 全部當月系統RTP低於系統期望RTP
def constraint5(x):
    return full_judge(count_monthly_rtp_exp_comparation(x)) 

#6. 非全部當月系統RTP低於系統期望RTP
def constraint6(x):
    return -full_judge(count_monthly_rtp_exp_comparation(x)) 

#7. 全部當月系統RTP低於當月系統RTP上限
def constraint7(x):
    return full_judge(count_monthly_rtp_cap_comparation(x))

#8. 非全部當月系統RTP低於當月系統RTP上限
def constraint8(x):
    return -full_judge(count_monthly_rtp_cap_comparation(x))

#9. 當月系統虧損上限開啓
def constraint9(x):
    return x[15]

#10. 當月系統虧損上限未開啓
def constraint10(x):
    return -x[15]

#11. 全部當月系統虧損低於當月系統虧損上限
def constraint11(x):
    return full_judge(count_monthly_loss_cap_comparation(x))

#12. 非全部當月系統虧損低於當月系統虧損上限
def constraint12(x):
    return -full_judge(count_monthly_loss_cap_comparation(x))

#13. 單日系統虧損上限開啓
def constraint13(x):
    return x[17]

#14. 單日系統虧損上限未開啓
def constraint14(x):
    return -x[17]

#15. 全部單日系統虧損低於單日系統虧損上限
def constraint15(x):
    return full_judge(count_daily_loss_cap_comparation(x))  

#16. 非全部單日系統虧損低於單日系統虧損上限
def constraint16(x):
    return -full_judge(count_daily_loss_cap_comparation(x))

#17. 系統所有收益相等
def constraint17(x):
    return full_judge(count_banker_profit_equal(x))

#18. 系統非所有收益相等
def constraint18(x):
    return -full_judge(count_banker_profit_equal(x))

#19. 系統收益全爲正
def constraint19(x):
    return full_judge(count_banker_profit_positive(x))

#20. 系統收益全爲負
def constraint20(x):
    return zero_judge(count_banker_profit_positive(x))

#21. -#19 & -#20
def constraint21(x):
    return sign_judge(-full_judge(count_banker_profit_positive(x)),-zero_judge(count_banker_profit_positive(x)))

#22. 本局所有派彩相等
def constraint22(x):
    return full_judge(count_banker_pay_equal(x))

#23. 本局非所有派彩相等
def constraint23(x):
    return -full_judge(count_banker_pay_equal(x))


CON_DCT = {
    '當月RTP上限功能有開啓': constraint0,
    '當月RTP上限功能未開啓': constraint1,
    '全部週期RTP高於當前週期RTP': constraint2,
    '全部週期RTP低於當前週期RTP': constraint3,
    '非全部週期RTP高於當前週期RTP & 非全部週期RTP低於當前週期RTP': constraint4,
    '全部當月RTP低於系統期望RTP': constraint5,
    '非全部當月RTP低於系統期望RTP': constraint6,
    '全部當月RTP低於當月RTP上限': constraint7,
    '非全部當月RTP低於當月RTP上限': constraint8,
    '當月虧損上限開啓': constraint9,
    '當月虧損上限未開啓': constraint10,
    '全部當月虧損低於或等於當月虧損上限': constraint11,
    '非全部當月虧損低於或等於當月虧損上限': constraint12,
    '單日虧損上限開啓': constraint13,
    '單日虧損上限未開啓': constraint14,
    '全部單日虧損低於或等於單日虧損上限': constraint15,
    '非全部單日虧損低於或等於單日虧損上限': constraint16,
    '系統所有收益相等': constraint17,
    '系統非所有收益相等': constraint18,
    '系統收益全爲正': constraint19,
    '系統收益全爲負': constraint20,
    '非系統收益全爲正 & 非系統收益全爲負': constraint21,
    '全部派彩相等': constraint22,
    '非全部派彩相等': constraint23
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
LAYER2 = (make_constraint(constraint2), make_constraint(constraint3), make_constraint(constraint4))
LAYER3 = (make_constraint(constraint5), make_constraint(constraint6))
LAYER4 = (make_constraint(constraint7), make_constraint(constraint8))
LAYER5 = (make_constraint(constraint9), make_constraint(constraint10))
LAYER6 = (make_constraint(constraint11), make_constraint(constraint12))
LAYER7 = (make_constraint(constraint13), make_constraint(constraint14))
LAYER8 = (make_constraint(constraint15), make_constraint(constraint16))
# layer9 = (make_constraint(constraint17), make_constraint(constraint18))
# layer10 = (make_constraint(constraint19), make_constraint(constraint20), make_constraint(constraint21))
# layer11 = (make_constraint(constraint22), make_constraint(constraint23))

TYPE_LAYER1 = ('當月RTP上限功能有開啓', '當月RTP上限功能未開啓')
TYPE_LAYER2 = ('全部週期RTP高於當前週期RTP', '全部週期RTP低於當前週期RTP', '非全部週期RTP高於當前週期RTP & 非全部週期RTP低於當前週期RTP')
TYPE_LAYER3 = ('全部當月RTP低於系統期望RTP', '非全部當月RTP低於系統期望RTP')
TYPE_LAYER4 = ('全部當月RTP低於當月RTP上限', '非全部當月RTP低於當月RTP上限')
TYPE_LAYER5 = ('當月虧損上限開啓', '當月虧損上限未開啓')
TYPE_LAYER6 = ('全部當月虧損低於或等於當月虧損上限', '非全部當月虧損低於或等於當月虧損上限')
TYPE_LAYER7 = ('單日虧損上限開啓', '單日虧損上限未開啓')
TYPE_LAYER8 = ('全部單日虧損低於或等於單日虧損上限', '非全部單日虧損低於或等於單日虧損上限')
# TYPE_LAYER9 = ('系統所有收益相等', '系統非所有收益相等')
# type_layer10 = ('系統收益全爲正', '系統收益全爲負', '非系統收益全爲正 & 非系統收益全爲負')
# type_layer11 = ('全部派彩相等', '非全部派彩相等')

CONSTRAINT_MAP = {
        TYPE_LAYER1[0]: LAYER1[0], TYPE_LAYER1[1]: LAYER1[1],
        TYPE_LAYER2[0]: LAYER2[0], TYPE_LAYER2[1]: LAYER2[1], TYPE_LAYER2[2]: LAYER2[2],
        TYPE_LAYER3[0]: LAYER3[0], TYPE_LAYER3[1]: LAYER3[1],
        TYPE_LAYER4[0]: LAYER4[0], TYPE_LAYER4[1]: LAYER4[1],
        TYPE_LAYER5[0]: LAYER5[0], TYPE_LAYER5[1]: LAYER5[1],
        TYPE_LAYER6[0]: LAYER6[0], TYPE_LAYER6[1]: LAYER6[1],
        TYPE_LAYER7[0]: LAYER7[0], TYPE_LAYER7[1]: LAYER7[1],
        TYPE_LAYER8[0]: LAYER8[0], TYPE_LAYER8[1]: LAYER8[1],
    }


# 一組條件約束下生成參數
def process_single_config(args):
    config_counts, _type = args
    _con = [CONSTRAINT_MAP[t] for t in _type[:CONS_NUM]]

    para_dct = {}
    para_dct[_type[:CONS_NUM]] = {}
    
    for _spl in range(SAMPLE_NUM):
        _spl += 15
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

        # 處理n_turns_distribution成爲指數分佈
        final_solution[22:30] = generate_exponential_sum_integers()
        para_dct[_type[:CONS_NUM]][_spl] = final_solution

        parameters = { 
            "region_bet": final_solution[:6], 
            "monthly_total_bet": final_solution[6],            
            "monthly_total_pay": final_solution[7],     
            "cycle_total_bet": final_solution[8],            
            "cycle_total_pay": final_solution[9],            
            "daily_total_bet": final_solution[10],            
            "daily_total_pay": final_solution[11],           
            "process_type": 'Random',          
            "RTPfactor_square":0.06,           
            # "current_cycle_RTP": final_solution[-1],         
            "monthly_RTP_cap_enabled": True if final_solution[13] > 0 else False,     
            "monthly_RTP_cap": final_solution[14],             
            "expected_RTP": final_solution[12],                 
            "monthly_loss_cap_enabled": True if final_solution[15] > 0 else False,     
            "monthly_loss_cap": final_solution[16],         
            "daily_loss_cap_enabled": True if final_solution[17] > 0 else False,        
            "daily_loss_cap": final_solution[18],               
            "consecutive_above_door": final_solution[19],            
            "consecutive_heaven_door": final_solution[20],            
            "consecutive_below_door": final_solution[21],                      
            "pass_n_turns_distribution": final_solution[22:30]
        }
        
        double_check.single_double_check(r'F:/projects/TTZ/Validation/log/', f'config_{config_counts}_{_spl}.log', parameters, config_counts, _spl+1, final_solution, str(_type)[1:-1])
    
    return para_dct


# 主函數
def main():
    # 生成类型组合
    type_combinations = list(itertools.product(
        TYPE_LAYER1, TYPE_LAYER2, TYPE_LAYER3, TYPE_LAYER4, 
        TYPE_LAYER5, TYPE_LAYER6, TYPE_LAYER7, TYPE_LAYER8
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

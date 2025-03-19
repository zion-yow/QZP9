import random
import numpy as np
from dataclasses import dataclass
from IPython.display import display
from cond_judge import cond_judge
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
from datetime import datetime

np.set_printoptions(precision=5, suppress=True)
warnings.filterwarnings("ignore")
# 設置中文字體支持
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# -----------------------------
# 定義常數輸入參數的數據類
# -----------------------------
@dataclass
class ConstantInputs:
    # ------------
    # 中間結果變量所需的常數參數
    # ------------

    # 各區投注
    region_bet: list                 
    # 當月系統總投注
    monthly_total_bet: Decimal
    # 當月系統總派彩
    monthly_total_pay: Decimal
    # 週期系統總投注
    cycle_total_bet: Decimal
    # 週期系統總派彩
    cycle_total_pay: Decimal
    # 當日系統總投注
    daily_total_bet: Decimal
    # 當日系統總派彩
    daily_total_pay: Decimal
    # 當前週期系統RTP
    current_cycle_RTP: Decimal     

    # ------------
    # 其他常數參數
    # ------------     
    process_type: str                    # 流程類型
    RTPfactor_square: Decimal              # RTP因子平方

    monthly_RTP_cap_enabled: bool         # 當月系統RTP上限是否開啓
    monthly_RTP_cap: Decimal                # 當月系統RTP上限
    expected_RTP: Decimal                   # 系統期望RTP

    monthly_loss_cap_enabled: bool        # 當月系統虧損上限是否開啓
    monthly_loss_cap: Decimal               # 當月系統虧損上限
    daily_loss_cap_enabled: bool          # 單日系統虧損上限是否開啓
    daily_loss_cap: Decimal                 # 單日系統虧損上限

    # 正數為勝，負數為負， 不取0
    consecutive_above_door: int                 # 連續次數（上門）
    consecutive_heaven_door: int               # 連續次數（天門）
    consecutive_below_door: int                 # 連續次數（下門）

    pass_n_turns_distribution: list       # 過 N 局分布的情況分佈
    

    def __post_init__(self):
        # 結果矩陣
        self.banker_result_matrix = np.array([
            [-1.95,-1.95,-1.95,-1.95,-1.95,-1.95],
            [-1.95,-1.95,0,-1.95,-1,-1],
            [-1.95,0,-1.95,-1,-1.95,-1],
            [-1.95,0,0,-1,-1,0],
            [0,-1.95,-1.95,-1,-1,-1.95],
            [0,-1.95,0,-1,0,-1],
            [0,0,-1.95,0,-1,-1],
            [0,0,0,0,0,0],
        ])
        self.consecutive_matrix = np.tile([
            self.consecutive_above_door,
            self.consecutive_heaven_door,
            self.consecutive_below_door,                    
        ],(8,1))

        side_after = np.where(self.banker_result_matrix[:,:3] < 0 ,-1 ,1)

        side_before = np.where(self.consecutive_matrix < 0 ,-1 ,1)
        self.consecutive_matrix = np.where(
            side_after == side_before, 
            self.consecutive_matrix,
            0
            )

        self.mapping = {
            0: 1, 
            1: 0.98,
            2: 0.97,
            3: 0.95, 
            4: 0.9, 
            5: 0.8, 
            6: 0.7, 
            7: 0.5, 
            8: 0.1, 
            9: 0.01,
            10: 0.001
            }
        
        self.dis_mapping = {
            0: 1.1, 
            1: 1,
            2: 1,
            3: 0.8, 
            4: 0.6, 
            5: 0.4, 
            }
        
        
        self.bet_metrix = np.tile(np.array(self.region_bet),(8,1))                 # 各區投注
        self.total_bet = np.sum(np.array(self.region_bet))                          # 縂投注
        self.total_bet_array = self.total_bet * np.ones((8,1))                     # 縂投注向量
        self.cycle_total_bet_array = self.cycle_total_bet * np.ones((8,1))         # 週期系統總投注向量
        self.cycle_total_pay_array = self.cycle_total_pay * np.ones((8,1))   # 週期系統總派彩向量
        self.monthly_total_bet_array = self.monthly_total_bet * np.ones((8,1))         # 當月系統總投注向量
        self.monthly_total_pay_array = self.monthly_total_pay * np.ones((8,1))   # 當月系統總派彩向量
        self.daily_total_bet_array = self.daily_total_bet * np.ones((8,1))         # 當日系統總投注向量
        self.daily_total_pay_array = self.daily_total_pay * np.ones((8,1))   # 當日系統總派彩向量
        
    
# -----------------------------
# 定義中間變量的數據類
# -----------------------------
@dataclass
class IntermediateVariables:
    current_pay: np.array       # 本局派彩
    banker_profit: np.array     # 莊家收益
    cycle_RTP: np.array         # 週期系統RTP
    monthly_RTP: np.array       # 當月系統RTP
    monthly_loss: np.array      # 當月系統虧損
    daily_loss: np.array        # 單日系統虧損
    
    def __post_init__(self):
        self.inter_vars_df = np.array([self.current_pay.reshape(-1), self.banker_profit.reshape(-1), self.cycle_RTP.reshape(-1), self.monthly_RTP.reshape(-1), self.monthly_loss.reshape(-1), self.daily_loss.reshape(-1)])
# -----------------------------
# 定義最終輸出因子值的數據類
# -----------------------------
@dataclass
class OutputFactors:
    RTP_factor: Decimal                   # RTP因子
    win_loss_factor: Decimal              # 輸贏因子
    monthly_RTP_cap_factor: Decimal       # 當月RTP上限因子
    monthly_loss_cap_factor: Decimal      # 當月虧損上限因子
    daily_loss_cap_factor: Decimal        # 單日虧損上限因子

    situation_adjustment_factor: Decimal  # 情況調整因子

# -----------------------------
# 計算中間變量（示例公式，根據實際需求替換）
# -----------------------------
def calculate_intermediate_vars(inputs: ConstantInputs) -> IntermediateVariables:
    # 本局派彩向量
    banker_pay_clm = inputs.bet_metrix * inputs.banker_result_matrix
    banker_pay_array = -np.sum(banker_pay_clm,axis=1).T.reshape(8,1)
    
    # 莊家收益向量
    banker_profit_array = inputs.total_bet_array - banker_pay_array

    # 週期系統RTP向量
    cycle_RTP_array = (inputs.cycle_total_pay_array + banker_pay_array) / (inputs.cycle_total_bet_array + inputs.total_bet_array) * 10000
    display(cycle_RTP_array)

    # 當月系統RTP向量
    monthly_RTP_array = (inputs.monthly_total_pay_array + banker_pay_array) / (inputs.monthly_total_bet_array + inputs.total_bet_array) * 10000

    # 當月系統虧損向量
    monthly_loss_array = -inputs.monthly_total_bet_array + inputs.monthly_total_pay_array - banker_profit_array

    # 單日系統虧損向量
    daily_loss_array = -inputs.daily_total_bet_array + inputs.daily_total_pay_array - banker_profit_array

    return IntermediateVariables(
        current_pay=np.where(banker_pay_array > 0, np.round(banker_pay_array), np.floor(banker_pay_array)),
        banker_profit=np.where(banker_profit_array > 0, np.round(banker_profit_array), np.floor(banker_profit_array)),
        cycle_RTP=np.where(cycle_RTP_array > 0, np.round(cycle_RTP_array), np.floor(cycle_RTP_array)),
        monthly_RTP=np.where(monthly_RTP_array > 0, np.round(monthly_RTP_array), np.floor(monthly_RTP_array)),
        monthly_loss=np.where(monthly_loss_array > 0, np.round(monthly_loss_array), np.floor(monthly_loss_array)),
        daily_loss=np.where(daily_loss_array > 0, np.round(daily_loss_array), np.floor(daily_loss_array))
    )

# -----------------------------
# 條件樹的判斷函數：根據中間變量和常數輸入計算各因子值
# -----------------------------
def evaluate_condition_tree(inputs: ConstantInputs, inter_vars: IntermediateVariables) -> OutputFactors:
    # 功能開啓標誌（布爾值，這裡直接為標量）

    # 0. 莊家收益是否為最高：判斷每個位置是否等於整個向量的最大值
    cond0 = (inter_vars.banker_profit == np.max(inter_vars.banker_profit, axis=0))
    # 1. 莊家收益是否為正
    cond1 = (inter_vars.banker_profit > 0)
    # 2. 本局派彩是否為最低：判斷每個位置是否等於整個向量的最小值
    cond2 = (inter_vars.current_pay == np.min(inter_vars.current_pay, axis=0))
    # 3. 周期RTP低於等於當前周期RTP的情況數量是否為0
    count_cycle = np.sum(inter_vars.cycle_RTP <= inputs.current_cycle_RTP)
    cond3 = np.full((8, 1), count_cycle == 0)
    # 4. 周期RTP低於等於當前周期RTP的情況數量是否為8
    cond4 = np.full((8, 1), count_cycle == 8) 
    # 5. 周期RTP是否低於當前周期RTP：逐元素比較
    cond5 = (inter_vars.cycle_RTP < inputs.current_cycle_RTP)
    # 6. 當月RTP上限功能是否開啟：
    cond6 = np.full((8, 1), inputs.monthly_RTP_cap_enabled)
    # 7. 當月RTP是否低於系統期望RTP：逐元素比較
    cond7 = (inter_vars.monthly_RTP < inputs.expected_RTP)
    # 8. 當月RTP是否低於當月系統RTP上限：逐元素比較
    cond8 = (inter_vars.monthly_RTP < inputs.monthly_RTP_cap)
    # 9. 當月虧損上限功能是否開啟
    cond9 = np.full((8, 1), inputs.monthly_loss_cap_enabled)
    # 10. 當月虧損是否低於當月系統虧損上限：逐元素比較
    cond10 = (inter_vars.monthly_loss < inputs.monthly_loss_cap)
    # 11. 當日虧損上限功能是否開啟
    cond11 = np.full((8, 1), inputs.daily_loss_cap_enabled)
    # 12. 當日虧損是否低於當日系統虧損上限：逐元素比較
    cond12 = (inter_vars.daily_loss < inputs.daily_loss_cap)

    # 將所有條件存入一個列表中，便於檢查
    conditions = [cond0, cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond10, cond11, cond12]
    
    # 計算非常數因子值
    monthly_RTP_cap_array = np.full((8,1), inputs.monthly_RTP_cap)
    expected_RTP_array = np.full((8,1), inputs.expected_RTP)
    
    factor_value_m = np.array(np.full((8,1), np.mean(inter_vars.cycle_RTP[~cond5])),dtype=object)
    _controller = ((monthly_RTP_cap_array - inter_vars.monthly_RTP) /(monthly_RTP_cap_array - expected_RTP_array))**inputs.RTPfactor_square
    _controller = np.where(np.isnan(_controller), 0, _controller)   
    # print('xxxxxxxxxxxxxxxxxxxxxxx',_controller,'xxxxxxxxxxxxxxxxxxxxxxxxx')     
    factor_value_j = np.array(np.round(np.max(np.array([_controller, np.full((8,1),0.0001)]).reshape(2,8).T,axis=1),4),dtype=object)
    
    cond_matrix = np.array(conditions)
    cond_matrix = np.squeeze(cond_matrix).T
    
    # 條件樹判斷
    Output_Factors = cond_judge(
        cond_matrix, 
        inputs.process_type,
        factor_value_m,
        factor_value_j,
        inputs
        )

    # 设置全局打印选项
    # np.set_printoptions(
    #     suppress=True,   # 禁用科学计数法
    #     # precision=4,     # 保留小数点后6位（可调整）
    #     # floatmode="fixed"  # 固定小数位数（可选）
    # )


    Output_Factors = pd.DataFrame(Output_Factors).applymap(lambda x: Decimal(str(x)))
    Output_Factors.columns = [
        "RTPFactor",
        "WinLoseFactor", 
        "MonthlyRTPFactor", 
        "MonthlyLossFactor", 
        "DailyLossFactor", 
        "AboveContinueFactor", 
        "HeavenContinueFactor", 
        "BelowContinueFactor", 
        "AdjustFactor"
        ]
    
    return Output_Factors

X = [
    1120,                        #0
    500,                         #1 
    210,                         #2
    800,                         #3 region_bet: random.randint(100, 1000) for _ in range(4)],
    1000000,                     #4 monthly_total_bet: random.randint(10000, 100000),
    983000,                      #5 monthly_total_pay: random.randint(10000, 100000),
    10000,                       #6 cycle_total_bet: random.randint(1000, 10000),
    9740,                        #7 cycle_total_pay: random.randint(1000, 10000),
    5000,                        #8 daily_total_bet: random.randint(3000, 20000),
    5200,                        #9 daily_total_pay: random.randint(3000, 20000),
    9750,                        #10 expected_RTP: random.randint(9000, 10000),
    1,                           #11 monthly_RTP_cap_enabled: random.choice([1, 0]),
    10000,                       #12 monthly_RTP_cap: random.randint(9000, 10000),
    1,                           #13 monthly_loss_cap_enabled: random.choice([1, 0]),
    2500000,                     #14 monthly_loss_cap: random.randint(1000000, 10000000),
    1,                           #15 daily_loss_cap_enabled: random.choice([1, 0]),
    2500000,                     #8 daily_loss_cap: random.randint(100000, 10000),
    9,                           #17 consecutive_heaven: random.randint(-10, 10),
    -3,                          #18 consecutive_earth: random.randint(-10, 10),
    -4,                          #19 consecutive_mystic: random.randint(-10, 10),
    5,                           #20 consecutive_yellow: random.randint(-10, 10),
    1,                           #21 
    1,                           #22
    0,                           #23
    1,                           #24
    0,                           #25
    3,                           #26
    1,                           #27
    2,                           #28
    1,                           #29
    0,                           #30
    1,                           #31
    2,                           #32
    1,                           #33
    0,                           #34
    1,                           #35
    2,                           #36 pass_n_turns_distribution: random.randint(0, 10) for _ in range(8)
    9740,                        #37 current_cycle_RTP: random.randint(9000, 10000),

]


# -----------------------------
# 測試用例執行函數
# -----------------------------
def generate_test_case(
        region_bet=[100, 500, 15100, 520, 3000, 110], 
        monthly_total_bet=1000000,            
        monthly_total_pay=983000,     
        cycle_total_bet=10000,            
        cycle_total_pay=9740,            
        daily_total_bet=5000,            
        daily_total_pay=5200,           
        process_type='Random',          
        RTPfactor_square=0.06,         
        monthly_RTP_cap_enabled=True,     
        monthly_RTP_cap=10000,             
        expected_RTP=9750,                 
        monthly_loss_cap_enabled=True,     
        monthly_loss_cap=2500000,         
        daily_loss_cap_enabled=True,        
        daily_loss_cap=2500000,               
        consecutive_above_door=9,            
        consecutive_heaven_door=-3,            
        consecutive_below_door=5,                            
        pass_n_turns_distribution=[1,0,3,4,6,2,0,1]
    ) -> None:
    
    INTT_WEIGHT = np.array([2436,834,834,837,834,837,837,2551])

    current_cycle_RTP=int(10000*cycle_total_pay/(cycle_total_bet if cycle_total_bet != 0 else 1))

    inputs = ConstantInputs(
        region_bet=region_bet,
        monthly_total_bet=monthly_total_bet,
        monthly_total_pay=monthly_total_pay,
        cycle_total_bet=cycle_total_bet,         
        cycle_total_pay=cycle_total_pay,        
        daily_total_bet=daily_total_bet,         
        daily_total_pay=daily_total_pay,          
        process_type=process_type,          
        RTPfactor_square=RTPfactor_square,         
        current_cycle_RTP=current_cycle_RTP,           
        monthly_RTP_cap_enabled=monthly_RTP_cap_enabled,      
        monthly_RTP_cap=monthly_RTP_cap,      
        expected_RTP=expected_RTP,             
        monthly_loss_cap_enabled=monthly_loss_cap_enabled,    
        monthly_loss_cap=monthly_loss_cap,           
        daily_loss_cap_enabled=daily_loss_cap_enabled,   
        daily_loss_cap=daily_loss_cap,               
        consecutive_above_door=consecutive_above_door,          
        consecutive_heaven_door=consecutive_heaven_door,            
        consecutive_below_door=consecutive_below_door,          
        pass_n_turns_distribution=pass_n_turns_distribution
    )
    # 計算中間變量
    inter_vars = calculate_intermediate_vars(inputs)
    
    # 根據條件樹計算輸出因子
    output_factors = evaluate_condition_tree(inputs, inter_vars)
    
    inter_df = pd.DataFrame(inter_vars.inter_vars_df.T, columns=['TotalPayout', 'SystemProfit', 'CycleRTP', 'MonthlyRTP', 'MonthlyLoss', 'DailyLoss'])
    

    output_factors = output_factors.applymap(lambda x: Decimal(str(x)))
    Sys_factor = output_factors.apply(lambda row:row['RTPFactor']*row['WinLoseFactor']*row['MonthlyRTPFactor']*row['MonthlyLossFactor']*row['DailyLossFactor'], axis=1)

    output_factors['InitFactor'] = INTT_WEIGHT.astype('object')
    output_factors['SystemFactor'] = Sys_factor.astype('object')

    output_factors = output_factors.applymap(lambda x: Decimal(str(x)))
    
    Medal_factor = output_factors.apply(lambda row:row['AboveContinueFactor']*row['HeavenContinueFactor']*row['BelowContinueFactor']*row['AdjustFactor'], axis=1)

    output_factors['MedalFactor'] = Medal_factor    
    
    output_factors = output_factors.applymap(lambda x: Decimal(str(x)))
    Real_weight = output_factors.apply(lambda row:
                                       row['MedalFactor'].quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP) 
                                       * row['InitFactor'].quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP) 
                                       * row['SystemFactor'].quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
                                       , axis=1)
    # print(Real_weight)
    display(Real_weight)
    output_factors['FinalWeight'] = Real_weight.apply(lambda x: x.quantize(Decimal('1'), rounding=ROUND_CEILING))
    output_factors['CumulativeWeights'] = np.cumsum(output_factors['FinalWeight'])
    # display(output_factors)

    # 打印常數輸入參數
    print("【常數輸入參數】")
    display(pd.Series(list(inputs.__dict__.values())[:22], index=list(inputs.__dict__.keys())[:22]))


    output_factors = output_factors.applymap(lambda x: Decimal(str(x)))
    return inputs, inter_df, output_factors.applymap(lambda x: x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))




# -----------------------------
# 多次累計模擬功能
# -----------------------------
def simulate_multiple_rounds(
        rounds,
        region_bet, 
        monthly_total_bet,            
        monthly_total_pay,     
        cycle_total_bet,            
        cycle_total_pay,            
        daily_total_bet,            
        daily_total_pay,           
        process_type,          
        RTPfactor_square,         
        monthly_RTP_cap_enabled,     
        monthly_RTP_cap,             
        expected_RTP,                 
        monthly_loss_cap_enabled,     
        monthly_loss_cap,         
        daily_loss_cap_enabled,        
        daily_loss_cap,               
        consecutive_above_door,            
        consecutive_heaven_door,            
        consecutive_below_door,                            
        pass_n_turns_distribution,
        review_round # 回顧局數限制
    ):
    """
    多次累計模擬功能：模擬連續多局遊戲，每局基於前一局累計參數進行更新
    
    參數:
    -----
    rounds : int
        模擬局數，預設為10局
    region_bet : list
        各區投注額，預設為[100, 500, 15100, 520, 3000, 110]
    monthly_total_bet : Decimal
        當月系統總投注，預設為1000000
    monthly_total_pay : Decimal
        當月系統總派彩，預設為983000
    cycle_total_bet : Decimal
        週期系統總投注，預設為10000
    cycle_total_pay : Decimal
        週期系統總派彩，預設為9740
    daily_total_bet : Decimal
        當日系統總投注，預設為5000
    daily_total_pay : Decimal
        當日系統總派彩，預設為5200
    process_type : str
        流程類型，預設為'Random'，可選值包括'SystemWin'、'SystemLose'等
    RTPfactor_square : Decimal
        RTP因子平方，預設為0.06
    monthly_RTP_cap_enabled : bool
        當月系統RTP上限是否開啓，預設為True
    monthly_RTP_cap : Decimal
        當月系統RTP上限，預設為10000
    expected_RTP : Decimal
        系統期望RTP，預設為9750
    monthly_loss_cap_enabled : bool
        當月系統虧損上限是否開啓，預設為True
    monthly_loss_cap : Decimal
        當月系統虧損上限，預設為2500000
    daily_loss_cap_enabled : bool
        單日系統虧損上限是否開啓，預設為True
    daily_loss_cap : Decimal
        單日系統虧損上限，預設為2500000
    consecutive_above_door : int
        連續上門次數，預設為0，正數表示莊家連贏，負數表示連輸
    consecutive_heaven_door : int
        連續天門次數，預設為0，正數表示莊家連贏，負數表示連輸
    consecutive_below_door : int
        連續下門次數，預設為0，正數表示莊家連贏，負數表示連輸
    pass_n_turns_distribution : list
        過 N 局分布的情況分佈，預設為[0,0,0,0,0,0,0,0]
    review_round : int
        回顧局數限制，預設為17，決定情況分布的最大累計局數
    
    更新規則:
    -------
    1. 投注和派彩：每局的投注和派彩會累加到系統總投注和總派彩
    2. 連續次數：根據結果矩陣更新連續次數，正數增加表示連贏，負數增加表示連輸
    3. 情況分布：增加當前獲勝情況的計數，如果超過回顧局數則移除最早的情況
    
    返回:
    -----
    simulation_results : list
        包含每一局模擬結果的列表，每局結果包含當局狀態和累計結果
    summary : dict
        模擬結果統計，包含最終投注總額、派彩總額、RTP等信息
    """
    # 創建基於時間戳的日誌目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"f:/projects/TTZ/log_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"創建日誌目錄: {log_dir}")

    # 初始化結果列表
    simulation_results = []
    
    # 當前參數狀態
    current_params = {
        'region_bet': region_bet,
        'monthly_total_bet': monthly_total_bet,
        'monthly_total_pay': monthly_total_pay,
        'cycle_total_bet': cycle_total_bet,
        'cycle_total_pay': cycle_total_pay,
        'daily_total_bet': daily_total_bet,
        'daily_total_pay': daily_total_pay,
        'process_type': process_type,
        'RTPfactor_square': RTPfactor_square,
        'monthly_RTP_cap_enabled': monthly_RTP_cap_enabled,
        'monthly_RTP_cap': monthly_RTP_cap,
        'expected_RTP': expected_RTP,
        'monthly_loss_cap_enabled': monthly_loss_cap_enabled,
        'monthly_loss_cap': monthly_loss_cap,
        'daily_loss_cap_enabled': daily_loss_cap_enabled,
        'daily_loss_cap': daily_loss_cap,
        'consecutive_above_door': consecutive_above_door,
        'consecutive_heaven_door': consecutive_heaven_door,
        'consecutive_below_door': consecutive_below_door,
        'pass_n_turns_distribution': pass_n_turns_distribution.copy()
    }

    # 新增一個队列來追踪情況發生的時間順序
    situation_history = []

    print(f"開始進行{rounds}局模擬...")
    
    for round_num in range(1, rounds + 1):
        print(f"\n模擬第 {round_num} 局...")
        
        # 使用當前參數進行模擬
        inputs, inter_df, output_factors = generate_test_case(
            region_bet=current_params['region_bet'],
            monthly_total_bet=current_params['monthly_total_bet'],
            monthly_total_pay=current_params['monthly_total_pay'],
            cycle_total_bet=current_params['cycle_total_bet'],
            cycle_total_pay=current_params['cycle_total_pay'],
            daily_total_bet=current_params['daily_total_bet'],
            daily_total_pay=current_params['daily_total_pay'],
            process_type=current_params['process_type'],
            RTPfactor_square=current_params['RTPfactor_square'],
            monthly_RTP_cap_enabled=current_params['monthly_RTP_cap_enabled'],
            monthly_RTP_cap=current_params['monthly_RTP_cap'],
            expected_RTP=current_params['expected_RTP'],
            monthly_loss_cap_enabled=current_params['monthly_loss_cap_enabled'],
            monthly_loss_cap=current_params['monthly_loss_cap'],
            daily_loss_cap_enabled=current_params['daily_loss_cap_enabled'],
            daily_loss_cap=current_params['daily_loss_cap'],
            consecutive_above_door=current_params['consecutive_above_door'],
            consecutive_heaven_door=current_params['consecutive_heaven_door'],
            consecutive_below_door=current_params['consecutive_below_door'],
            pass_n_turns_distribution=current_params['pass_n_turns_distribution']
        )
        
        # 獲取模擬結果
        # 計算FinalWeight之間的間距
        final_weights = output_factors['FinalWeight'].to_numpy()
        weight_gaps = np.zeros(len(final_weights) - 1)
        for i in range(len(final_weights) - 1):
            weight_gaps[i] = float(final_weights[i+1]) - float(final_weights[i])
        
        # 按照個情況間距加權后隨機選取
        probability = [p if p > 0 else 0.0000000001 for p in final_weights / final_weights.sum()]
        winning_situation_idx = np.random.choice(range(len(final_weights)), p=probability)
        winning_situation = winning_situation_idx + 1  # 情況從1開始編號

        # 識別間距最大處右側邊界
        max_gap_index = np.argmax(weight_gaps)
        
        # 獲取當前局的派彩和投注
        current_bet = sum(current_params['region_bet'])
        current_pay = inter_df.iloc[winning_situation_idx]['TotalPayout']
        banker_profit = inter_df.iloc[winning_situation_idx]['SystemProfit']
        
        # 更新累計參數
        # 1. 更新投注和派彩
        current_params['monthly_total_bet'] += current_bet
        current_params['monthly_total_pay'] += current_pay
        current_params['cycle_total_bet'] += current_bet
        current_params['cycle_total_pay'] += current_pay
        current_params['daily_total_bet'] += current_bet
        current_params['daily_total_pay'] += current_pay
        
        # 2. 更新連續次數
        # 獲取獲勝情況的結果矩陣
        result_matrix = inputs.banker_result_matrix[winning_situation_idx]
        
        # 上門
        if result_matrix[0] < 0:  # 莊家輸
            if current_params['consecutive_above_door'] <= 0:
                current_params['consecutive_above_door'] -= 1
            else:
                current_params['consecutive_above_door'] = -1

        elif result_matrix[0] >= 0:  # 莊家贏
            if current_params['consecutive_above_door'] >= 0:
                current_params['consecutive_above_door'] += 1
            else:
                current_params['consecutive_above_door'] = 1
        
        # 天門
        if result_matrix[1] < 0:  # 莊家輸
            if current_params['consecutive_heaven_door'] <= 0:
                current_params['consecutive_heaven_door'] -= 1
            else:
                current_params['consecutive_heaven_door'] = -1  

        elif result_matrix[1] >= 0:  # 莊家贏
            if current_params['consecutive_heaven_door'] >= 0:
                current_params['consecutive_heaven_door'] += 1
            else:
                current_params['consecutive_heaven_door'] = 1

        # 下門
        if result_matrix[2] < 0:  # 莊家輸
            if current_params['consecutive_below_door'] <= 0:
                current_params['consecutive_below_door'] -= 1
            else:
                current_params['consecutive_below_door'] = -1

        elif result_matrix[2] >= 0:  # 莊家贏
            if current_params['consecutive_below_door'] >= 0:
                current_params['consecutive_below_door'] += 1
            else:
                current_params['consecutive_below_door'] = 1
                
                
            
        # 3. 更新情況分布
        # 添加當前獲勝情況到歷史記錄
        situation_history.append(winning_situation_idx)
        
        # 增加當前獲勝情況的計數
        current_params['pass_n_turns_distribution'][winning_situation_idx] += 1

        # 檢查是否需要移除最早的情況（如果有回顧局數限制）
        total_situations = len(situation_history)
        if total_situations > review_round:
            # 獲取需要移除的最早情況
            oldest_situation_idx = situation_history.pop(0)
            # 減少對應情況的計數
            current_params['pass_n_turns_distribution'][oldest_situation_idx] -= 1
        
        
        # 計算當前RTP
        current_cycle_RTP = int(10000 * current_params['cycle_total_pay'] / current_params['cycle_total_bet'])
        current_monthly_RTP = int(10000 * current_params['monthly_total_pay'] / current_params['monthly_total_bet'])
        current_daily_RTP = int(10000 * current_params['daily_total_pay'] / current_params['daily_total_bet'])
        
        # 保存本局結果
        round_result = {
            'round': round_num,
            'winning_situation': winning_situation,
            'max_gap_index': max_gap_index + 1,  # 最大间距的左边界情况（从1开始编号）
            'max_gap_value': float(weight_gaps[max_gap_index]),  # 最大间距值
            'current_bet': current_bet,
            'current_pay': current_pay,
            'banker_profit': banker_profit,
            'monthly_total_bet': current_params['monthly_total_bet'],
            'monthly_total_pay': current_params['monthly_total_pay'],
            'cycle_total_bet': current_params['cycle_total_bet'],
            'cycle_total_pay': current_params['cycle_total_pay'],
            'daily_total_bet': current_params['daily_total_bet'],
            'daily_total_pay': current_params['daily_total_pay'],
            'cycle_RTP': current_cycle_RTP,
            'monthly_RTP': current_monthly_RTP,
            'daily_RTP': current_daily_RTP,
            'consecutive_above_door': current_params['consecutive_above_door'],
            'consecutive_heaven_door': current_params['consecutive_heaven_door'],
            'situation_history': situation_history.copy(),
            'consecutive_below_door': current_params['consecutive_below_door'],
            'pass_n_turns_distribution': current_params['pass_n_turns_distribution'].copy(),
            'weights': output_factors['FinalWeight'].tolist(),
            'cumulative_weights': output_factors['CumulativeWeights'].tolist()
        }
        
        simulation_results.append(round_result)
        
        print(f"第 {round_num} 局模擬完成")
        print(f"獲勝情況: {winning_situation} (按照個情況間距加權后隨機選取)")
        print(f"最大間距位置: {max_gap_index + 1}~{max_gap_index + 2}, 間距值: {weight_gaps[max_gap_index]}")
        print(f"所有權重: {[float(w) for w in output_factors['FinalWeight']]}")
        print(f"當前週期RTP: {current_cycle_RTP}")
        print(f"當前月RTP: {current_monthly_RTP}")
        print(f"當前日RTP: {current_daily_RTP}")
        print(f"連續上門次數: {current_params['consecutive_above_door']}")
        print(f"連續天門次數: {current_params['consecutive_heaven_door']}")
        print(f"連續下門次數: {current_params['consecutive_below_door']}")
        print(f"情況分布: {current_params['pass_n_turns_distribution']}")
    
    print("\n所有模擬完成!")
    
    # 生成統計結果
    summary = {
        'total_rounds': rounds,
        'final_monthly_bet': current_params['monthly_total_bet'],
        'final_monthly_pay': current_params['monthly_total_pay'],
        'final_monthly_RTP': current_monthly_RTP,
        'final_cycle_bet': current_params['cycle_total_bet'],
        'final_cycle_pay': current_params['cycle_total_pay'],
        'final_cycle_RTP': current_cycle_RTP,
        'final_daily_bet': current_params['daily_total_bet'],
        'final_daily_pay': current_params['daily_total_pay'],
        'final_daily_RTP': current_daily_RTP,
        'situation_distribution': current_params['pass_n_turns_distribution'],
        'situation_history': situation_history,
        'situation_counts': {i+1: sum(1 for idx in situation_history if idx == i) for i in range(8)}
    }
    
    # 打印統計結果
    print("\n模擬統計結果:")
    print(f"總局數: {summary['total_rounds']}")
    print(f"最終月投注總額: {summary['final_monthly_bet']}")
    print(f"最終月派彩總額: {summary['final_monthly_pay']}")
    print(f"最終月RTP: {summary['final_monthly_RTP']}")
    print(f"最終週期投注總額: {summary['final_cycle_bet']}")
    print(f"最終週期派彩總額: {summary['final_cycle_pay']}")
    print(f"最終週期RTP: {summary['final_cycle_RTP']}")
    print(f"最終日投注總額: {summary['final_daily_bet']}")
    print(f"最終日派彩總額: {summary['final_daily_pay']}")
    print(f"最終日RTP: {summary['final_daily_RTP']}")
    
    # 計算並顯示情況分布百分比
    if len(situation_history) > 0:
        recent_history = situation_history[-min(20, len(situation_history)):]

    total_situations = sum(summary['situation_distribution'])
    print(f"\n(最近{len(recent_history)}局情況分布 (百分比):")
    for i, count in enumerate(summary['situation_distribution']):
        percentage = (count / total_situations * 100) if total_situations > 0 else 0
        print(f"情況 {i+1}: {percentage:.2f}%")

    # 將結果轉換為DataFrame以便分析
    df_results = pd.DataFrame(simulation_results)
    
    # 繪製RTP變化圖
    plt.figure(figsize=(12, 8))
    
    # plt.subplot(3, 1, 1)
    plt.plot(df_results['round'], df_results['monthly_RTP'])
    plt.title("月RTP變化")
    plt.grid(True)
    
    # plt.subplot(3, 1, 2)
    # plt.plot(df_results['round'], df_results['cycle_RTP'], marker='o')
    # plt.title("週期RTP變化")
    # plt.grid(True)
    
    # plt.subplot(3, 1, 3)
    # plt.plot(df_results['round'], df_results['daily_RTP'], marker='o')
    # plt.title("日RTP變化")
    # plt.grid(True)
    
    # plt.tight_layout()
    plt.savefig(f"{log_dir}/simulation_rtp_{rounds}_rounds.png")
    plt.show()
    
    # 繪製連續次數變化圖
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(df_results['round'], df_results['consecutive_above_door'], marker='o')
    plt.title("連續上門次數變化")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_results['round'], df_results['consecutive_heaven_door'], marker='o')
    plt.title("連續天門次數變化")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df_results['round'], df_results['consecutive_below_door'], marker='o')
    plt.title("連續下門次數變化")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/simulation_consecutive_{rounds}_rounds.png")
    plt.show()
    
    # 繪製最大間距變化圖
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_results['round'], df_results['max_gap_value'], marker='o', color='red')
    plt.title("最大間距值變化")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df_results['round'], df_results['max_gap_index'], marker='o', color='green')
    plt.title("最大間距位置變化 (左邊界)")
    plt.yticks(range(1, 9))
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/simulation_gaps_{rounds}_rounds.png")
    plt.show()
    
    # 繪製情況分布圖
    plt.figure(figsize=(10, 6))
    situation_counts = [sum(1 for r in simulation_results if r['winning_situation'] == i+1) for i in range(8)]
    total_situations = sum(situation_counts)
    situation_percentages = [(count / total_situations * 100) if total_situations > 0 else 0 for count in situation_counts]
    plt.bar(range(1, 9), situation_percentages)
    plt.xlabel("情況")
    plt.ylabel("出現百分比 (%)")
    plt.title("總計各情況出現百分比分布")
    plt.xticks(range(1, 9))
    plt.grid(True, axis='y')
    plt.savefig(f"{log_dir}/simulation_distribution_{rounds}_rounds.png")
    plt.show()
    
    # 繪製情況歷史趨勢圖
    if len(situation_history) > 0:
        plt.figure(figsize=(12, 6))
        # 將情況索引轉換為從1開始的情況編號
        situation_history_1based = [idx + 1 for idx in situation_history]
        plt.plot(range(1, len(situation_history_1based) + 1), situation_history_1based, marker='o', linestyle='-')
        plt.xlabel("局數")
        plt.ylabel("情況")
        plt.title("情況變化趨勢")
        plt.yticks(range(1, 9))
        plt.grid(True)
        plt.savefig(f"{log_dir}/simulation_history_trend_{rounds}_rounds.png")
        plt.show()

    # 保存模擬結果到CSV文件
    df_results.to_csv(f"{log_dir}/simulation_results_{rounds}_rounds.csv", index=False)
    print(f"模擬結果已保存到 {log_dir}/simulation_results_{rounds}_rounds.csv")
    
    # 保存模擬參數和統計結果到文本文件
    with open(f"{log_dir}/simulation_summary.txt", "w", encoding="utf-8") as f:
        f.write("模擬參數:\n")
        f.write(f"總局數: {rounds}\n")
        f.write(f"流程類型: {process_type}\n")
        f.write(f"RTP因子平方: {RTPfactor_square}\n")
        f.write(f"系統期望RTP: {expected_RTP}\n")
        f.write(f"各區投注: {region_bet}\n")
        f.write(f"回顧局數: {review_round}\n\n")
        
        f.write("最終統計結果:\n")
        for key, value in summary.items():
            if key not in ['situation_counts', 'situation_distribution', 'situation_history']:
                f.write(f"{key}: {value}\n")
        
        f.write("\n情況分布 (百分比):\n")
        for i, ptg in enumerate(situation_percentages):
            f.write(f"情況 {i+1}: {ptg:.2f}%\n")
    
    print(f"模擬參數和統計結果已保存到 {log_dir}/simulation_summary.txt")
    
    return simulation_results, summary, situation_percentages

if __name__ == '__main__':
    print("=" * 60)
    print("TTZ模擬系統 - 多輪累計模擬工具")
    print("=" * 60)
    print("本工具可以進行單次模擬或多次累計模擬:")
    print("1. 單次模擬: 模擬單局遊戲結果，顯示中間變量和輸出因子")
    print("2. 多次累計模擬: 模擬多局遊戲結果，參數會按規則累計變化")
    print("   - 可自定義高級參數（RTP因子、流程類型等）")
    print("   - 生成RTP變化、連續次數變化、情況分布等統計圖表")
    print("   - 所有數據保存至CSV文件便於後續分析")
    print("-" * 60)
    
    # 選擇模式
    mode = input("選擇模式 (1: 單次模擬, 2: 多次累計模擬): ")
    
    if mode == "1":
        inputs, inter_df, output_factors = generate_test_case()
        print("【常數輸入參數】")
        display(pd.Series(list(inputs.__dict__.values())[:22], index=list(inputs.__dict__.keys())[:22]))

        # 打印中間變量
        print("\n【中間變量】")
        display(inter_df)

        # 打印輸出因子
        print("\n【輸出因子】")
        display(output_factors)
    
    elif mode == "2":
        rounds = int(input("輸入模擬局數: "))
        advanced_setting = input("是否需要設置高級参數? (y/n): ").lower()
        
        # 默認參數
        params = {
            'rounds': rounds,
            'region_bet': [0, 0, 3000, 0, 0, 0],
            'monthly_total_bet': 0,
            'monthly_total_pay': 0,
            'cycle_total_bet': 0,
            'cycle_total_pay': 0,
            'daily_total_bet': 0,
            'daily_total_pay': 0,
            'process_type': 'Random',
            'RTPfactor_square': 0.12,
            'monthly_RTP_cap_enabled': True,
            'monthly_RTP_cap': 10000,
            'expected_RTP': 9700,
            'monthly_loss_cap_enabled': True,
            'monthly_loss_cap': 3000000,
            'daily_loss_cap_enabled': True,
            'daily_loss_cap': 3000000,
            'consecutive_above_door': 0,
            'consecutive_heaven_door': 0,
            'consecutive_below_door': 0,
            'pass_n_turns_distribution': [0,0,0,0,0,0,0,0],
            'review_round': 17
        }
        
        if advanced_setting == 'y':
            print("\n進行高級設定...")
            
            # 設置RTP因子平方
            rtp_factor = input(f"RTP因子平方 (default: {params['RTPfactor_square']}): ")
            if rtp_factor:
                params['RTPfactor_square'] = float(rtp_factor)
                
            # 設置流程類型
            process_type = input(f"流程類型 (Random/SystemWin/SystemLose/Break_Monthly_RTP_CAP/Break_Monthly_Loss_CAP/Break_Daily_Loss_CAP) (default: {params['process_type']}): ")
            if process_type:
                params['process_type'] = process_type
                
            # 設置系統期望RTP
            expected_rtp = input(f"系統期望RTP (default: {params['expected_RTP']}): ")
            if expected_rtp:
                params['expected_RTP'] = int(expected_rtp)
                
            # 設置各區投注
            region_bet = input(f"各區投注額 (6個值，用逗號分隔) (default: {params['region_bet']}): ")
            if region_bet:
                params['region_bet'] = [int(x.strip()) for x in region_bet.split(',')]
                
            # 設置連續次數
            consecutive = input(f"連續次數(上門,天門,下門) (default: {params['consecutive_above_door']},{params['consecutive_heaven_door']},{params['consecutive_below_door']}): ")
            if consecutive:
                consecutives = [int(x.strip()) for x in consecutive.split(',')]
                if len(consecutives) >= 3:
                    params['consecutive_above_door'] = consecutives[0]
                    params['consecutive_heaven_door'] = consecutives[1]
                    params['consecutive_below_door'] = consecutives[2]
                    
            # 設置回顧局數
            review_round = input(f"回顧局數 (default: {params['review_round']}): ")
            if review_round:
                params['review_round'] = int(review_round)
        
        # 執行模擬
        print("\n使用以下參數進行模擬:")
        for key, value in params.items():
            if key != 'rounds':
                print(f"{key}: {value}")
                
        simulation_results, summary, situation_percentages = simulate_multiple_rounds(
            rounds=params['rounds'],
            region_bet=params['region_bet'],
            monthly_total_bet=params['monthly_total_bet'],
            monthly_total_pay=params['monthly_total_pay'],
            cycle_total_bet=params['cycle_total_bet'],
            cycle_total_pay=params['cycle_total_pay'],
            daily_total_bet=params['daily_total_bet'],
            daily_total_pay=params['daily_total_pay'],
            process_type=params['process_type'],
            RTPfactor_square=params['RTPfactor_square'],
            monthly_RTP_cap_enabled=params['monthly_RTP_cap_enabled'],
            monthly_RTP_cap=params['monthly_RTP_cap'],
            expected_RTP=params['expected_RTP'],
            monthly_loss_cap_enabled=params['monthly_loss_cap_enabled'],
            monthly_loss_cap=params['monthly_loss_cap'],
            daily_loss_cap_enabled=params['daily_loss_cap_enabled'],
            daily_loss_cap=params['daily_loss_cap'],
            consecutive_above_door=params['consecutive_above_door'],
            consecutive_heaven_door=params['consecutive_heaven_door'],
            consecutive_below_door=params['consecutive_below_door'],
            pass_n_turns_distribution=params['pass_n_turns_distribution'],
            review_round=params['review_round']
        )
        
        # 打印最終統計結果
        print("\n最終統計結果:")
        for key, value in summary.items():
            if key not in ['situation_counts', 'situation_distribution', 'situation_history']:
                print(f"{key}: {value}")
        
        # 計算並顯示情況分布百分比
        print("\n情況分布 (百分比):")
        for i, ptg in enumerate(situation_percentages):
            print(f"情況 {i+1}: {ptg:.2f}%")
            
    

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
    
    # 是否使用新手表
    is_Rookie_table: bool
    
    # 當月系統總投注
    monthly_total_bet: Decimal
    # 當月系統總派彩
    monthly_total_pay: Decimal
    # 週期系統總投注
    cycle_total_bet: Decimal
    # 週期系統總派彩
    cycle_total_pay: Decimal


    # ------------
    # 其他常數參數
    # ------------     
    monthly_RTP_cap_enabled: bool         # 當月系統RTP上限是否開啓
    monthly_RTP_cap: Decimal                # 當月系統RTP上限

    monthly_loss_cap_enabled: bool        # 當月系統虧損上限是否開啓
    monthly_loss_cap: Decimal               # 當月系統虧損上限

    monthly_player_profit_cap_enabled: bool          # 當月個人盈利上限是否開啓
    monthly_player_profit_cap: Decimal                 # 當月個人盈利上限

# -----------------------------
# 定義中間變量的數據類
# -----------------------------
@dataclass
class IntermediateVariables:
    cycle_RTP: Decimal        # 週期系統RTP
    monthly_loss: Decimal      # 當月系統虧損
    monthly_player_profit: Decimal        # 當月個人盈利
    
    def __post_init__(self):
        self.inter_vars_df = np.array([
            self.cycle_RTP,
            self.monthly_loss,
            self.monthly_player_profit
                ])
# -----------------------------
# 定義最終輸出因子值的數據類
# -----------------------------
@dataclass
class OutputFactors:
    Cap_break_factor: Decimal                   # 上限突破因子

# -----------------------------
# 計算中間變量（示例公式，根據實際需求替換）
# -----------------------------
def calculate_intermediate_vars(inputs: ConstantInputs) -> IntermediateVariables:
    # 週期系統RTP向量
    cycle_RTP = (inputs.cycle_total_pay + inputs.monthly_total_pay) / (inputs.cycle_total_bet + inputs.monthly_total_bet) * 10000

    # 當月系統虧損向量
    monthly_loss = -inputs.monthly_total_bet + inputs.monthly_total_pay

    # 當月個人盈利
    monthly_player_profit = inputs.monthly_total_pay - inputs.monthly_total_bet

    return IntermediateVariables(
        cycle_RTP=cycle_RTP,
        monthly_loss=monthly_loss,
        monthly_player_profit=monthly_player_profit
    )

# -----------------------------
# 條件樹的判斷函數：根據中間變量和常數輸入計算各因子值
# -----------------------------
def evaluate_condition_tree(inputs: ConstantInputs, inter_vars: IntermediateVariables) -> OutputFactors:
    # 功能開啓標誌（布爾值，這裡直接為標量）

    # 0. 是否使用新手表
    cond0 = (inputs.is_Rookie_table)
    # 1. 當月RTP上限功能是否開啟：
    cond1 = (inputs.monthly_RTP_cap_enabled)
    # 2. 周期RTP是否低於當月系統RTP上限
    cond2 = (inter_vars.cycle_RTP < inputs.monthly_RTP_cap)
    # 3. 當月系統虧損上限功能是否開啟：
    cond3 = (inputs.monthly_loss_cap_enabled)
    # 4. 當月系統虧損是否低於當月系統虧損上限
    cond4 = (inter_vars.monthly_loss < inputs.monthly_loss_cap)
    # 5. 當月個人盈利上限功能是否開啟
    cond5 = (inputs.monthly_player_profit_cap_enabled)
    # 6. 當月個人盈利是否低於當月個人盈利上限
    cond6 = (inter_vars.monthly_player_profit < inputs.monthly_player_profit_cap)

    # 將所有條件存入一個列表中，便於檢查
    conditions = [cond0, cond1, cond2, cond3, cond4, cond5, cond6]
    
    # 計算非常數因子值
    
    cond_matrix = np.array(conditions)
    cond_matrix = np.squeeze(cond_matrix).T
    
    # 條件樹判斷
    Output_Factors = cond_judge(
        cond_matrix, 
        inputs.is_Rookie_table,
        inputs
        )

    Output_Factors = pd.Series([
        inter_vars.cycle_RTP, inter_vars.monthly_loss, inter_vars.monthly_player_profit, Output_Factors],
        index=['CycleRTP', 'MonthlyLoss', 'MonthlyPlayerProfit', 'CapBreakFactor']
        )
    return Output_Factors

# -----------------------------
# 測試用例執行函數
# -----------------------------
def generate_test_case( 
        monthly_total_bet=1000000,            
        monthly_total_pay=983000,     
        cycle_total_bet=10000,            
        cycle_total_pay=9740,
        is_Rookie_table=True,                      
        monthly_player_profit=9740,     
        monthly_RTP_cap_enabled=True,     
        monthly_RTP_cap=10000,             
        monthly_loss_cap_enabled=True,     
        monthly_loss_cap=2500000,         
        monthly_player_profit_cap_enabled=True,
        monthly_player_profit_cap=2500000,
    ) -> None:
    
    inputs = ConstantInputs(
        is_Rookie_table=is_Rookie_table,
        monthly_total_bet=monthly_total_bet,
        monthly_total_pay=monthly_total_pay,
        cycle_total_bet=cycle_total_bet,         
        cycle_total_pay=cycle_total_pay,                       
        monthly_RTP_cap_enabled=monthly_RTP_cap_enabled,      
        monthly_RTP_cap=monthly_RTP_cap,      
        monthly_loss_cap_enabled=monthly_loss_cap_enabled,    
        monthly_loss_cap=monthly_loss_cap,           
        monthly_player_profit_cap_enabled=monthly_player_profit_cap_enabled,   
        monthly_player_profit_cap=monthly_player_profit_cap,               
    )
    # 計算中間變量
    inter_vars = calculate_intermediate_vars(inputs)
    
    # 根據條件樹計算輸出因子
    output_factors = evaluate_condition_tree(inputs, inter_vars)
    inter_df = pd.Series(inter_vars.inter_vars_df.T, index=['CycleRTP', 'MonthlyLoss', 'MonthlyPlayerProfit'])
    output_factors = output_factors.apply(lambda x: Decimal(str(x)))

    # 打印常數輸入參數
    print("【常數輸入參數】")
    display(pd.Series(list(inputs.__dict__.values())[:22], index=list(inputs.__dict__.keys())[:22]))

    return inputs, inter_df, output_factors.apply(lambda x: x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))




# -----------------------------
# 多次累計模擬功能
# -----------------------------
def simulate_multiple_rounds(
        rounds,
        monthly_total_bet,            
        monthly_total_pay,     
        cycle_total_bet,            
        cycle_total_pay,
        is_Rookie_table,
        monthly_RTP_cap_enabled,     
        monthly_RTP_cap,             
        monthly_loss_cap_enabled,     
        monthly_loss_cap,         
        monthly_player_profit_cap_enabled,
        monthly_player_profit_cap,
    ):
    """
    多次累計模擬功能：模擬連續多局遊戲，每局基於前一局累計參數進行更新
    
    參數:
    -----
    rounds : int
        模擬局數，預設為10局
    monthly_total_bet : Decimal
        當月系統總投注，預設為1000000
    monthly_total_pay : Decimal
        當月系統總派彩，預設為983000
    cycle_total_bet : Decimal
        週期系統總投注，預設為10000
    cycle_total_pay : Decimal
        週期系統總派彩，預設為9740
    is_Rookie_table : bool
        是否使用新手表，預設為True
    monthly_RTP_cap_enabled : bool
        當月系統RTP上限是否開啓，預設為True
    monthly_RTP_cap : Decimal
        當月系統RTP上限，預設為10000
    monthly_loss_cap_enabled : bool
        當月系統虧損上限是否開啓，預設為True
    monthly_loss_cap : Decimal
        當月系統虧損上限，預設為2500000
    monthly_player_profit_cap_enabled : bool
        當月個人盈利上限是否開啓，預設為True
    monthly_player_profit_cap : Decimal
        當月個人盈利上限，預設為2500000
        
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
        'monthly_total_bet': monthly_total_bet,
        'monthly_total_pay': monthly_total_pay,
        'cycle_total_bet': cycle_total_bet,
        'cycle_total_pay': cycle_total_pay,
        'is_Rookie_table': is_Rookie_table,
        'monthly_RTP_cap_enabled': monthly_RTP_cap_enabled,
        'monthly_RTP_cap': monthly_RTP_cap,
        'monthly_loss_cap_enabled': monthly_loss_cap_enabled,
        'monthly_loss_cap': monthly_loss_cap,
        'monthly_player_profit_cap_enabled': monthly_player_profit_cap_enabled,
        'monthly_player_profit_cap': monthly_player_profit_cap,
    }

    # 新增一個队列來追踪情況發生的時間順序
    situation_history = []

    print(f"開始進行{rounds}局模擬...")
    
    for round_num in range(1, rounds + 1):
        print(f"\n模擬第 {round_num} 局...")
        
        # 使用當前參數進行模擬
        inputs, inter_df, output_factors = generate_test_case(
            monthly_total_bet=current_params['monthly_total_bet'],
            monthly_total_pay=current_params['monthly_total_pay'],
            cycle_total_bet=current_params['cycle_total_bet'],
            cycle_total_pay=current_params['cycle_total_pay'],
            is_Rookie_table=current_params['is_Rookie_table'],
            monthly_player_profit=current_params['monthly_player_profit'],
            monthly_RTP_cap_enabled=current_params['monthly_RTP_cap_enabled'],
            monthly_RTP_cap=current_params['monthly_RTP_cap'],
            monthly_loss_cap_enabled=current_params['monthly_loss_cap_enabled'],
            monthly_loss_cap=current_params['monthly_loss_cap'],
            monthly_player_profit_cap_enabled=current_params['monthly_player_profit_cap_enabled'],
            monthly_player_profit_cap=current_params['monthly_player_profit_cap'],
        )

        
        # 獲取當前局的派彩和投注
        current_bet = sum(current_params['monthly_total_bet'])
        current_pay = inter_df['TotalPayout']
        banker_profit = inter_df['SystemProfit']
        
        # 更新累計參數
        # 1. 更新投注和派彩和玩家收益
        current_params['monthly_total_bet'] += current_bet
        current_params['monthly_total_pay'] += current_pay
        current_params['cycle_total_bet'] = current_bet
        current_params['cycle_total_pay'] = current_pay
        current_params['monthly_player_profit'] += banker_profit
        current_params['monthly_loss'] += -current_bet + current_pay

        # 計算當前RTP
        current_cycle_RTP = int(10000 * current_params['cycle_total_pay'] / current_params['cycle_total_bet'])
        current_monthly_RTP = int(10000 * current_params['monthly_total_pay'] / current_params['monthly_total_bet'])
        current_daily_RTP = int(10000 * current_params['daily_total_pay'] / current_params['daily_total_bet'])
        
        # 保存本局結果
        round_result = {
            'round': round_num,
            'current_bet': current_bet,
            'current_pay': current_pay,
            'banker_profit': banker_profit,
            'monthly_total_bet': current_params['monthly_total_bet'],
            'monthly_total_pay': current_params['monthly_total_pay'],
            'cycle_total_bet': current_params['cycle_total_bet'],
            'cycle_total_pay': current_params['cycle_total_pay'],
            'monthly_RTP': current_monthly_RTP,
            'monthly_loss': current_params['monthly_loss'],
            'monthly_player_profit': current_params['monthly_player_profit'],
        }
        
        simulation_results.append(round_result)
        
        print(f"第 {round_num} 局模擬完成")
        print(f"當前月RTP: {current_monthly_RTP}")
        print(f"當前週期RTP: {current_cycle_RTP}")
        print(f"當前月虧損: {current_params['monthly_loss']}")
        print(f"當前月玩家盈利: {current_params['monthly_player_profit']}")

    print("\n所有模擬完成!")
    
    # 生成統計結果
    summary = {
        'total_rounds': rounds,
        'final_monthly_bet': current_params['monthly_total_bet'],
        'final_monthly_pay': current_params['monthly_total_pay'],
        'final_monthly_RTP': current_monthly_RTP,
        'final_monthly_loss': current_params['monthly_loss'],
        'final_monthly_player_profit': current_params['monthly_player_profit'],
    }
    
    # 打印統計結果
    print("\n模擬統計結果:")
    print(f"總局數: {summary['total_rounds']}")
    print(f"最終月投注總額: {summary['final_monthly_bet']}")
    print(f"最終月派彩總額: {summary['final_monthly_pay']}")
    print(f"最終月RTP: {summary['final_monthly_RTP']}")
    print(f"最終月虧損: {summary['final_monthly_loss']}")
    print(f"最終月玩家盈利: {summary['final_monthly_player_profit']}")
    

    # 將結果轉換為DataFrame以便分析
    df_results = pd.DataFrame(simulation_results)
    
    # 繪製RTP變化圖
    plt.figure(figsize=(12, 8))
    
    plt.plot(df_results['round'], df_results['monthly_RTP'])
    plt.title("月RTP變化")
    plt.grid(True)
    
    plt.savefig(f"{log_dir}/simulation_rtp_{rounds}_rounds.png")
    plt.show()
    
    # 繪製連續次數變化圖
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(df_results['round'], df_results['monthly_loss'], marker='o')
    plt.title("月虧損變化")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_results['round'], df_results['monthly_player_profit'], marker='o')
    plt.title("月玩家盈利變化")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df_results['round'], df_results['cycle_RTP'], marker='o')
    plt.title("週期RTP變化")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/simulation_consecutive_{rounds}_rounds.png")
    plt.show()
    
    # 繪製最大間距變化圖
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_results['round'], df_results['monthly_loss'], marker='o', color='red')
    plt.title("月虧損變化")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df_results['round'], df_results['monthly_player_profit'], marker='o', color='green')
    plt.title("月玩家盈利變化")
    plt.yticks(range(1, 9))
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/simulation_gaps_{rounds}_rounds.png")
    plt.show()
    
    

    # 保存模擬結果到CSV文件
    df_results.to_csv(f"{log_dir}/simulation_results_{rounds}_rounds.csv", index=False)
    print(f"模擬結果已保存到 {log_dir}/simulation_results_{rounds}_rounds.csv")
    
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
            'monthly_total_bet': 0,
            'monthly_total_pay': 0,
            'cycle_total_bet': 0,
            'cycle_total_pay': 0,
            'is_Rookie_table': True,
            'monthly_RTP_cap_enabled': True,
            'monthly_RTP_cap': 10000,
            'monthly_loss_cap_enabled': True,
            'monthly_loss_cap': 3000000,
            'monthly_player_profit_cap_enabled': True,
            'monthly_player_profit_cap': 2500000,
        }
        
        if advanced_setting == 'y':
            print("\n進行高級設定...")
            
            # 設置Rookie表
            is_Rookie_table = input(f"是否使用新手表 (default: {params['is_Rookie_table']}): ")
            if is_Rookie_table:
                params['is_Rookie_table'] = is_Rookie_table
                
            # 設置投注
            monthly_total_bet = input(f"當月投注額 (default: {params['monthly_total_bet']}): ")
            if monthly_total_bet:
                params['monthly_total_bet'] = int(monthly_total_bet)
                
            # 設置派彩
            monthly_total_pay = input(f"當月派彩額 (default: {params['monthly_total_pay']}): ")
            if monthly_total_pay:
                params['monthly_total_pay'] = int(monthly_total_pay)
                
        
        # 執行模擬
        print("\n使用以下參數進行模擬:")
        for key, value in params.items():
            if key != 'rounds':
                print(f"{key}: {value}")
                
        simulation_results, summary, situation_percentages = simulate_multiple_rounds(
            rounds=params['rounds'],
            monthly_total_bet=params['monthly_total_bet'],
            monthly_total_pay=params['monthly_total_pay'],
            cycle_total_bet=params['cycle_total_bet'],
            cycle_total_pay=params['cycle_total_pay'],
            is_Rookie_table=params['is_Rookie_table'],
            monthly_RTP_cap_enabled=params['monthly_RTP_cap_enabled'],
            monthly_RTP_cap=params['monthly_RTP_cap'],
            monthly_loss_cap_enabled=params['monthly_loss_cap_enabled'],
            monthly_loss_cap=params['monthly_loss_cap'],
            monthly_player_profit_cap_enabled=params['monthly_player_profit_cap_enabled'],
            monthly_player_profit_cap=params['monthly_player_profit_cap'],
        )
        
        # 打印最終統計結果
        print("\n最終統計結果:")
        for key, value in summary.items():
            print(f"{key}: {value}")

            
    

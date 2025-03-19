import json
from tabulate import tabulate
from datetime import datetime
import pandas as pd
import numpy as np
from IPython.display import display
from engine import generate_test_case
import yaml
import subprocess
import os
import shutil
import contextlib
from filelock import FileLock, Timeout
import time
import random

FLOW_TYPE_DCT = {
    'Random': 5,
    'SystemWin': 10,
    'SystemLose': 11,
    'Break_Monthly_RTP_CAP': 1,
    'Break_Monthly_Loss_CAP': 12,
    'Break_Daily_Loss_CAP': 3
}

RANDOM_RTP_RANGE_DCT = {
    0.02:10,
    0.03:9,
    0.04:8,
    0.05:7,
    0.06:6,
    0.07:5,
    0.08:4,
    0.09:3,
    0.10:2,
    0.11:1,
    0.12:0,
    0.13:-1,
    0.14:-2,
    0.15:-3,
    0.16:-4,
    0.17:-5,
    0.18:-6,
    0.19:-7,
    0.20:-8,
    0.21:-9,
    0.22:-10
}

SYS_LOSE_RTP_RANGE_DCT = {
    0.22:10,
    0.23:9,
    0.24:8,
    0.25:7,
    0.26:6,
    0.27:5,
    0.28:4,
    0.29:3,
    0.30:2,
    0.31:1,
    0.32:0,
    0.33:-1,
    0.34:-2,
    0.35:-3,
    0.36:-4,
    0.37:-5,
    0.38:-6,
    0.39:-7,
    0.40:-8,
    0.41:-9,
    0.42:-10
}

def format_number(value):
    """数字格式化处理器"""
    if isinstance(value, (int, float)):
        if value >= 10000:
            return f"{value:,.0f}"
        if isinstance(value, float) and 0 < value < 1:
            return f"{value:.0}"
        return str(value)
    return value

def generate_table(data, headers, title=None):
    """通用表格生成器"""
    formatted_data = []
    for row in data:    
        formatted_row = [format_number(cell) for cell in row]
        formatted_data.append(formatted_row)
    
    table = tabulate(
        formatted_data, 
        headers=headers,
        tablefmt="grid",
        stralign="center",
        numalign="center"
    )
    
    if title:
        title_line = f"### {title}"
        return f"{title_line}\n{table}\n"
    return f"{table}\n"

def process_log_data(log_data):
    """主处理函数"""

    '''
    --------------------------------
    需根據實際數據類型和鍵名調整
    --------------------------------
    '''

    # 輸入參數
    parameters = {}
    parameters["region_bet"] = [log_data["log"]["TopDoorBet"],log_data["log"]["SkyDoorBet"],log_data["log"]["BottomDoorBet"],log_data["log"]["TopCornerBet"],log_data["log"]["BridgeBet"],log_data["log"]["BottomCornerBet"]]

    parameters["monthly_total_bet"] = log_data["log"]["Results"]["RTPResult"]["SysRecord"]["MonthlyBet"]
    parameters["monthly_total_pay"] = log_data["log"]["Results"]["RTPResult"]["SysRecord"]["MonthlyPay"]
    parameters["cycle_total_bet"] = log_data["log"]["Results"]["RTPResult"]["SysRecord"]["CycleBet"]
    parameters["cycle_total_pay"] = log_data["log"]["Results"]["RTPResult"]["SysRecord"]["CyclePay"]
    parameters["daily_total_bet"] = log_data["log"]["Results"]["RTPResult"]["SysRecord"]["DailyBet"]
    parameters["daily_total_pay"] = log_data["log"]["Results"]["RTPResult"]["SysRecord"]["DailyPay"]
    parameters["process_type"] = log_data["log"]["Results"]["RTPResult"]["RTPFlow"]
    parameters["RTPfactor_square"] = log_data["log"]["CurrentRTPFactorExp"]
    parameters["current_cycle_RTP"] = log_data["log"]["Results"]["RTPRecord"]["Sys"]["FinalCycleRTP"]
    parameters["monthly_RTP_cap_ena bled"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyRTPLimitEnabled"]
    parameters["monthly_RTP_cap"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyRTPLimit"]
    parameters["expected_RTP"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["ExpectedRTP"]
    parameters["monthly_loss_cap_enabled"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyLossLimitEnabled"]
    parameters["monthly_loss_limit"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyLossLimit"]
    parameters["daily_loss_limit_enabled"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["DailyLossLimitEnabled"]
    parameters["daily_loss_limit"] = log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["DailyLossLimit"]
    parameters["consecutive_above_door"] = log_data["log"]['TopDoorContinueCount'] * (1 if log_data["log"]['TopDoorContinueResult'][-1] == "贏" else -1)
    parameters["consecutive_heaven_door"] = log_data["log"]['SkyDoorContinueCount'] * (1 if log_data["log"]['SkyDoorContinueResult'][-1] == "贏" else -1)
    parameters["consecutive_below_door"] = log_data["log"]['BottomDoorContinueCount'] * (1 if log_data["log"]['BottomDoorContinueResult'][-1] == "贏" else -1)
    parameters["pass_n_turns_distribution"] = [
            item["CardCount"]
        for item in log_data["log"]["RTPTable"][:]
    ]
    para_series = pd.Series(parameters)

    sys_records = [
        ["系統記錄", "月投注總額", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["MonthlyBet"]],
        ["", "初始月RTP", log_data["log"]["Results"]["RTPRecord"]["Sys"]["InitialMonthlyRTP"]],
        ["", "最終月RTP", log_data["log"]["Results"]["RTPRecord"]["Sys"]["FinalMonthlyRTP"]],
        ["", "月支付總額", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["MonthlyPay"]],
        ["", "日投注總額", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["DailyBet"]],
        ["", "日支付總額", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["DailyPay"]],
        ["", "周期注總額", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["CycleBet"]],
        ["", "周期付總額", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["CyclePay"]],
        ["", "初始周期RTP", log_data["log"]["Results"]["RTPRecord"]["Sys"]["InitialCycleRTP"]],
        ["", "最終周期RTP", log_data["log"]["Results"]["RTPRecord"]["Sys"]["FinalCycleRTP"]],
        ["", "周期縂數", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["CycleNo"]],
        ["", "周期輪次", log_data["log"]["Results"]["RTPResult"]["SysRecord"]["CycleRounds"]],
        ["RTP配置", "月預期RTP", log_data["log"]["Results"]["RTPResult"]["SysConfig"]["ExpectedRTP"]],
        ["", "月RTP上限啟用", "是" if log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyRTPLimitEnabled"] else "否"],
        ["", "月RTP上限值", f"{log_data['log']['Results']['RTPResult']['SysConfig']['RTPLimit']['MonthlyRTPLimit']}"],
        ["", "月虧損上限啟用", "是" if log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyLossLimitEnabled"] else "否"],
        ["", "月虧損上限", log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["MonthlyLossLimit"]],
        ["", "日虧損上限啟用", "是" if log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["DailyLossLimitEnabled"] else "否"],
        ["", "日虧損上限", log_data["log"]["Results"]["RTPResult"]["SysConfig"]["RTPLimit"]["DailyLossLimit"]],
        ["", "連續上門", log_data["log"]['TopDoorContinueCount'] * (1 if log_data["log"]['TopDoorContinueResult'][-1] == "贏" else -1)],
        ["", "連續天門", log_data["log"]['SkyDoorContinueCount'] * (1 if log_data["log"]['SkyDoorContinueResult'][-1] == "贏" else -1)],
        ["", "連續下門", log_data["log"]['BottomDoorContinueCount'] * (1 if log_data["log"]['BottomDoorContinueResult'][-1] == "贏" else -1)],
    ]

    _df_records = pd.DataFrame(sys_records)
    _df_records.columns = ["類別", "子項", "值"]
    # display(_df_records)
    
    # 表3: 當前投注匯總表
    bet_zones = [
        ["上門投注", log_data["log"]["TopDoorBet"]],
        ["天門投注", log_data["log"]["SkyDoorBet"]],
        ["下門投注", log_data["log"]["BottomDoorBet"]],
        ["上角投注", log_data["log"]["TopCornerBet"]],
        ["橋投注", log_data["log"]["BridgeBet"]],
        ["下角投注", log_data["log"]["BottomCornerBet"]],
    ]

    _df_bet = pd.DataFrame(bet_zones)
    _df_bet.columns = ["欄位", "值"]
    # display(_df_bet)


    # 表4: RTP結果分析表
    rtp_data = [
        [
            item["Situation"],
            item["TotalPayout"],
            item["SystemProfit"],
            item["CycleRTP"],
            item["MonthlyRTP"],
            item["MonthlyLoss"],
            item["DailyLoss"],
            item["CardCount"]
        ]
        for item in log_data["log"]["RTPTable"][:]
    ]
    df_rtp = pd.DataFrame(rtp_data)
    df_rtp.columns = ["情境", "current_pay", "banker_profit", "cycle_RTP", "monthly_RTP", "monthly_loss", "daily_loss", "pass_n_turns"]
    # display(df_rtp)


    # 表6: 權重配置表
    weight_data = [
        [
            item["Situation"],
            item["OriginalWeights"],
            item["SystemFactor"],
            item["RTPFactor"],
            item["WinLoseFactor"],
            item["MonthlyRTPFactor"],
            item["MonthlyLossFactor"],
            item["DailyLossFactor"],
            item["AwardCardFactor"],
            item["TopDoorContinueFactor"],
            item["SkyDoorContinueFactor"],
            item["BottomDoorContinueFactor"],
            item["AdjustFactor"],
            item["FinalWeight"],
            item["CumulativeWeights"]
        ]
        for item in log_data["log"]["GameWeightTable"][:]
    ]

    df_weight_data = pd.DataFrame(weight_data)
    df_weight_data.columns = [
            "情境",
            "OriginalWeights",
            "SystemFactor",
            "RTPFactor",
            "WinLoseFactor",
            "MonthlyRTPFactor",
            "MonthlyLossFactor",
            "DailyLossFactor",
            "AwardCardFactor",
            "TopDoorContinueFactor",
            "SkyDoorContinueFactor",
            "BottomDoorContinueFactor",
            "AdjustFactor",
            "FinalWeight",
            "CumulativeWeights"
            ]
    # display(df_weight_data)

    return para_series, df_rtp, df_weight_data


def log_loader(log_file_path:str, log_name:str):
    with open(log_file_path + log_name, 'r', encoding='utf-8') as file:
        formatted_data = []
        for line in file:
            try:
                if line.strip():
                    json_data = json.loads(line)
                    # 格式化输出到新文件
                    formatted_data.append(json.dumps(json_data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"解析错误在行: {line}")
                print(f"错误信息: {e}")
                continue

    # 将格式化的数据写入新文件
    with open(log_file_path + "formatted_" + log_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_data))

    return formatted_data


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def safe_load_yaml(path, timeout=10):
    lock_path = path + ".lock"
    lock = FileLock(lock_path, timeout=timeout)
    with lock:
        return load_yaml(path)


def represent_list_flow(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def save_yaml(data, path):
    class MyDumper(yaml.SafeDumper):
        pass
    # 将自定义的 representer 应用于所有 list 类型数据
    MyDumper.add_representer(list, represent_list_flow)

    # 保存 YAML 文件
    with open(path, 'w', encoding='utf-8') as file:
        yaml.dump(
            data,
            file,
            Dumper=MyDumper,           # 使用自定义 Dumper
            default_flow_style=False,  # 映射使用块状格式输出
            sort_keys=False,           # 保持字典的键顺序
            allow_unicode=True
        )
    print(f'{path} is saved.')


def safe_save_yaml(data, path, timeout=10):
    lock_path = path + ".lock"
    lock = FileLock(lock_path, timeout=timeout)
    try:
        with lock:
            save_yaml(data, path)
    except Timeout:
        print(f"獲取鎖超時，無法寫入 {path}")


def highlight_diff(data):
    # 創建一個與數據框相同形狀的 DataFrame，用於存儲樣式
    return pd.DataFrame('', index=data.index, columns=data.columns).where(
        pd.isna(data), 'background-color: red')


def wait_for_file(filepath, timeout=10):
    start_time = time.time()
    while not os.path.exists(filepath):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"等待文件 {filepath} 超时")
        time.sleep(0.1)


def silent_copy(src, dst):
    # 避免OI擁堵
    # 等待源文件存在
    wait_for_file(src, timeout=10)

    # 使用共享锁文件，确保同一时刻只有一个进程操作
    lock = FileLock(dst + ".lock", timeout=10)
    try:
        with lock:
            # 锁定期间，执行读写操作
            with open(src, 'r', encoding='utf-8') as source:
                data = source.read()
            with open(dst, 'w', encoding='utf-8') as target:
                target.write(data)
    except Timeout:
        print("获取锁超时，可能其他进程长时间占用锁")


def single_double_check(log_file_path:str, log_name:str, parameters:dict, order:int, _spl:int, final_solution:dict, para_type:str):
    
    # -----------------step 1: 填寫測試工具參數------------------------
    
    os.makedirs(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/config_game', exist_ok=True)

    lock_path = log_file_path + ".lock"
    lock = FileLock(lock_path, timeout=15)
    try:
        with lock:
            # 設置bet_strategy參數
            bet_strategy_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/bet_strategy.yaml')
            bet_strategy_parameters["fixed_top_door"] = int(parameters["region_bet"][0])
            bet_strategy_parameters["fixed_sky_door"] = int(parameters["region_bet"][1])
            bet_strategy_parameters["fixed_bottom_door"] = int(parameters["region_bet"][2])
            bet_strategy_parameters["fixed_top_corner"] = int(parameters["region_bet"][3])
            bet_strategy_parameters["fixed_bridge"] = int(parameters["region_bet"][4])
            bet_strategy_parameters["fixed_bottom_corner"] = int(parameters["region_bet"][5])
            save_yaml(bet_strategy_parameters, f'F:/projects/TTZ/Validation/' + 'config_game/bet_strategy.yaml')
            save_yaml(bet_strategy_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/bet_strategy.yaml')

            # 設置game_init參數（回顧過去對局）
            game_init_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/game_init.yaml')
            game_init_parameters["top_door_continue_count"] = abs(int(parameters["consecutive_above_door"]))
            game_init_parameters["top_door_continue_type"] = 1 if int(parameters["consecutive_above_door"]) > 0 else 2
            game_init_parameters["sky_door_continue_count"] = abs(int(parameters["consecutive_heaven_door"]))
            game_init_parameters["sky_door_continue_type"] = 1 if int(parameters["consecutive_heaven_door"]) > 0 else 2
            game_init_parameters["bottom_door_continue_count"] = abs(int(parameters["consecutive_below_door"]))
            game_init_parameters["bottom_door_continue_type"] = 1 if int(parameters["consecutive_below_door"]) > 0 else 2
            # 回顧局數出現次數
            game_init_parameters["review_round_count_1"] = int(parameters["pass_n_turns_distribution"][0])
            game_init_parameters["review_round_count_2"] = int(parameters["pass_n_turns_distribution"][1])
            game_init_parameters["review_round_count_3"] = int(parameters["pass_n_turns_distribution"][2])
            game_init_parameters["review_round_count_4"] = int(parameters["pass_n_turns_distribution"][3])
            game_init_parameters["review_round_count_5"] = int(parameters["pass_n_turns_distribution"][4])
            game_init_parameters["review_round_count_6"] = int(parameters["pass_n_turns_distribution"][5])
            game_init_parameters["review_round_count_7"] = int(parameters["pass_n_turns_distribution"][6])
            game_init_parameters["review_round_count_8"] = int(parameters["pass_n_turns_distribution"][7])
            save_yaml(game_init_parameters, f'F:/projects/TTZ/Validation/' + 'config_game/game_init.yaml')
            save_yaml(game_init_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/game_init.yaml')

            # 設置rtp_valid_tool_setting參數
            rtp_valid_tool_setting_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/rtp_valid_tool_setting.yaml')
            rtp_valid_tool_setting_parameters["flow_type"] = FLOW_TYPE_DCT[parameters["process_type"]]
            rtp_valid_tool_setting_parameters["monthly_bet_amount"] = int(parameters["monthly_total_bet"])
            rtp_valid_tool_setting_parameters["monthly_pay_amount"] = int(parameters["monthly_total_pay"])
            rtp_valid_tool_setting_parameters["cycle_bet_amount"] = int(parameters["cycle_total_bet"])
            rtp_valid_tool_setting_parameters["cycle_pay_amount"] = int(parameters["cycle_total_pay"])
            rtp_valid_tool_setting_parameters["daily_bet_amount"] = int(parameters["daily_total_bet"])
            rtp_valid_tool_setting_parameters["daily_pay_amount"] = int(parameters["daily_total_pay"])
            rtp_valid_tool_setting_parameters["monthly_rtp_limit_enabled"] = parameters["monthly_RTP_cap_enabled"]
            rtp_valid_tool_setting_parameters["monthly_rtp_limit"] = int(parameters["monthly_RTP_cap"])
            rtp_valid_tool_setting_parameters["monthly_loss_limit_enabled"] = parameters["monthly_loss_cap_enabled"]
            rtp_valid_tool_setting_parameters["monthly_loss_limit"] = int(parameters["monthly_loss_cap"])
            rtp_valid_tool_setting_parameters["daily_loss_limit_enabled"] = parameters["daily_loss_cap_enabled"]
            rtp_valid_tool_setting_parameters["daily_loss_limit"] = int(parameters["daily_loss_cap"])
            rtp_valid_tool_setting_parameters["system_expect_rtp"] = int(parameters["expected_RTP"])
            rtp_valid_tool_setting_parameters["system_rtp_range_no"] = RANDOM_RTP_RANGE_DCT[parameters["RTPfactor_square"]] if rtp_valid_tool_setting_parameters["flow_type"] == 5 else SYS_LOSE_RTP_RANGE_DCT[parameters["RTPfactor_square"]]
            save_yaml(rtp_valid_tool_setting_parameters, f'F:/projects/TTZ/Validation/' + 'config_game/rtp_valid_tool_setting.yaml')
            save_yaml(rtp_valid_tool_setting_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/rtp_valid_tool_setting.yaml')

            # 設置tool_setting參數
            tool_setting_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/tool_setting.yaml')
            tool_setting_parameters["group_id"] = log_name[:-4]
            tool_setting_parameters["round_count"] = 3
            save_yaml(tool_setting_parameters, f'F:/projects/TTZ/Validation/' + 'config_game/tool_setting.yaml')
            save_yaml(tool_setting_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/tool_setting.yaml')

            # 複製game_setting參數
            game_setting_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/game_setting.yaml')
            save_yaml(game_setting_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/game_setting.yaml')

            # 複製room_setting參數
            room_setting_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/room_setting.yaml')
            save_yaml(room_setting_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/room_setting.yaml')

            # 複製tool_setting參數
            tool_setting_parameters = load_yaml('F:/projects/TTZ/Validation/' + 'config_game/tool_setting.yaml')
            save_yaml(tool_setting_parameters, f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/' + 'config_game/tool_setting.yaml')
            
            # -----------------step 2: 運行驗證工具------------------------ 
            exe_path = r"F:/projects\TTZ\Validation\validation_tool"
            result = subprocess.run([exe_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print("驗證工具運行成功")
            else:
                print("驗證工具運行失敗")

            # 複製log
            # os.makedirs(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/log', exist_ok=True)
            # silent_copy(
            #     log_file_path + log_name, 
            #     f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/log/' + log_name
            #     )
            
            # 記錄參數類型
            with open(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/para_type.txt', 'w', encoding='utf-8') as file:
                file.write(para_type)

            # -----------------step 2: 讀取驗證工具執行結果日志------------------------
            result_json_data_list = log_loader(log_file_path, log_name)


            # -----------------step 3: 讀取驗證工具執行結果日志（只執行一輪for循環只跑一次，多次執行則牌型結果不同）------------------------
            result_json_data_list = log_loader(log_file_path, log_name)
            for i, log_data in enumerate(result_json_data_list):
                # 將log_data轉換為json格式
                print(f'+++++++++++++++++++++++++ Round {i} ++++++++++++++++++++++++++')
                log_data = json.loads(log_data)
                para_series, rtp_data, df_weight_data = process_log_data(log_data)


                # -----------------step 4: 執行 engine------------------------
                inputs, inter_df, output_factors = generate_test_case(
                    region_bet=final_solution[:6]*1000, 
                    monthly_total_bet=final_solution[6],            
                    monthly_total_pay=final_solution[7],     
                    cycle_total_bet=final_solution[8],            
                    cycle_total_pay=final_solution[9],            
                    daily_total_bet=final_solution[10],            
                    daily_total_pay=final_solution[11],           
                    process_type='Random',          
                    RTPfactor_square=0.06,                 
                    monthly_RTP_cap_enabled=True if final_solution[13] > 0 else False,     
                    monthly_RTP_cap=final_solution[14],             
                    expected_RTP=final_solution[12],                 
                    monthly_loss_cap_enabled=True if final_solution[15] > 0 else False,     
                    monthly_loss_cap=final_solution[16],         
                    daily_loss_cap_enabled=True if final_solution[17] > 0 else False,        
                    daily_loss_cap=final_solution[18],               
                    consecutive_above_door=final_solution[19],            
                    consecutive_heaven_door=final_solution[20],            
                    consecutive_below_door=final_solution[21],                      
                    pass_n_turns_distribution=final_solution[22:30]
                )

                # -----------------step 5: 比較驗證工具和engine的結果------------------------

                # 比較中間因子
                rtp_data.columns = ['Situation','TotalPayout','SystemProfit','CycleRTP','MonthlyRTP','MonthlyLoss','DailyLoss','CardCount']
                comparison_inter = inter_df.astype('float').round(0).compare(rtp_data[inter_df.columns].round(0))

                # 比較權重表
                df_weight_data.columns = ['Situation','InitFactor','SystemFactor','RTPFactor','WinLoseFactor','MonthlyRTPFactor','MonthlyLossFactor','DailyLossFactor','MedalFactor','AboveContinueFactor','HeavenContinueFactor','BelowContinueFactor','AdjustFactor','FinalWeight','CumulativeWeights']
                comparison_weight = output_factors.astype('float').round(4).compare(df_weight_data[output_factors.columns].round(4))

                # 如果不完全一致，打印中間因子與權重表
                if not comparison_inter.empty:
                    print('【中間因子】')
                    display(inter_df, rtp_data[inter_df.columns])
                    pd.Series(list(inputs.__dict__.values())[:22], index=list(inputs.__dict__.keys())[:22]).to_csv(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/inputs.csv')
                    comparison_inter.to_csv(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/comparison_inter.csv')
                
                # 如果權重表不為空，打印權重表
                if not comparison_weight.empty:
                    print('【權重表】')
                    display(inter_df)
                    display(output_factors, output_factors.astype('float'), df_weight_data[output_factors.columns])
                    pd.Series(list(inputs.__dict__.values())[:22], index=list(inputs.__dict__.keys())[:22]).to_csv(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/inputs.csv')
                    comparison_weight.to_csv(f'F:/projects/TTZ/Validation/config_list/config_{str(order)}_{str(_spl)}/comparison_weight.csv')
                    
                # 顯示比較結果並標紅不同的單元格
                inter_styled_comparison = comparison_inter.style.apply(highlight_diff, axis=None)
                weight_styled_comparison = comparison_weight.style.apply(highlight_diff, axis=None)

                # 顯示比較結果
                display(inter_styled_comparison)
                display(weight_styled_comparison)

    except Timeout:
        print(f"獲取鎖超時，無法寫入 {log_file_path}")


# if __name__ == "__main__":
#     single_double_check(r'F:/projects/TTZ/Validation/', 'rtp_test_001.txt')
import numpy as np
from IPython.display import display
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP

np.set_printoptions(precision=5, suppress=True)

to_decimal = np.vectorize(lambda x: Decimal(str(x)))  # 避免浮點數誤差
quantize_decimal = np.vectorize(lambda x: x.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP))
quantize_decimal4 = np.vectorize(lambda x: x.quantize(Decimal('0.0001'), rounding=ROUND_CEILING))


def cond_judge(
        conditions_matrix:np.array, 
        process_type:str, 
        factor_value_m, 
        factor_value_j,
        inputs
        ):
    
    if process_type == 'Random':
        RTP_factor, WIN_factor = Random_judge(
            conditions_matrix,
            process_type,
            factor_value_m, 
            factor_value_j
            )
        
    elif process_type == 'Sys_Lose':
        RTP_factor, WIN_factor = Sys_Lose_judge(
            conditions_matrix,
            process_type,
            factor_value_m, 
            factor_value_j
            )
    
    elif process_type == 'Sys_Win':
        RTP_factor, WIN_factor = Sys_Lose_judge(
            conditions_matrix,
            process_type,
            factor_value_m, 
            factor_value_j
            )

    # 爆當月系統RTP上限流程 \    \ 爆單日系統虧損上限流程
    elif process_type == 'Break_Monthly_RTP_CAP' or process_type == 'Break_Monthly_Loss_CAP' or process_type == 'Break_Daily_Loss_CAP':
        WIN_factor = np.zeros(conditions_matrix.shape[0], dtype=int)
        WIN_factor[
        conditions_matrix[:,0]
        ] = 1

        RTP_factor = np.zeros(conditions_matrix.shape[0], dtype=int)
        RTP_factor[
        conditions_matrix[:,0]
        ] = 1
    


    # 不需要判斷流程類型的因子
    Monthly_RTP_CAP_factor = Monthly_RTP_CAP_factor_judger(conditions_matrix)
    Monthly_Loss_CAP_factor = Monthly_Loss_CAP_factor_judger(conditions_matrix)
    Daily_Loss_CAP_factor = Daily_Loss_CAP_factor_judger(conditions_matrix)
    
    # 連續因子
    Con_above_factor = [1 if x <= 0 else (inputs.mapping[x] if 0 < abs(x) <= 9 else 0.001) for x in np.abs(inputs.consecutive_matrix[:,0])]
    Con_heaven_factor = [1 if x <= 0 else (inputs.mapping[x] if 0 < abs(x) <= 9 else 0.001) for x in np.abs(inputs.consecutive_matrix[:,1])]
    Con_below_factor = [1 if x <= 0 else (inputs.mapping[x] if 0 < abs(x) <= 9 else 0.001) for x in np.abs(inputs.consecutive_matrix[:,2])]
    
    Sit_Judger_factor = [0.4 if x > 5 else str(inputs.dis_mapping[x]) for x in inputs.pass_n_turns_distribution]
    Output_Factors_lst = [
        RTP_factor, 
        WIN_factor, 
        Monthly_RTP_CAP_factor, 
        Monthly_Loss_CAP_factor, 
        Daily_Loss_CAP_factor, 
        Con_above_factor, 
        Con_heaven_factor, 
        Con_below_factor, 
        Sit_Judger_factor
        ]
    # display(Output_Factors_lst)

    Output_Factors =  np.array(Output_Factors_lst).reshape(-1,8).T

    return Output_Factors

# Monthly_RTP_CAP_factor_judger
def Monthly_RTP_CAP_factor_judger(conditions_matrix:np.array):
    # 預設RTP因子值全部為 1
    Monthly_RTP_CAP_factor = np.ones(conditions_matrix.shape[0], dtype=int)
    
    Monthly_RTP_CAP_factor = np.where(
        (conditions_matrix[:,6])
        &(conditions_matrix[:,8]),
        1,
        Monthly_RTP_CAP_factor
    )
    
    Monthly_RTP_CAP_factor = np.where(
        (conditions_matrix[:,6])
        &(~conditions_matrix[:,8]) 
        ,
        0,
        Monthly_RTP_CAP_factor
    )
    
    Monthly_RTP_CAP_factor = np.where(
        (conditions_matrix[:,6])
        &(conditions_matrix[:,2]),
        1,
        Monthly_RTP_CAP_factor
    )

    return Monthly_RTP_CAP_factor

# Monthly_Loss_CAP_factor_judger
def Monthly_Loss_CAP_factor_judger(conditions_matrix:np.array):
    # 預設RTP因子值全部為 1
    Monthly_Loss_CAP_factor = np.ones(conditions_matrix.shape[0], dtype=int)
    
    
    Monthly_Loss_CAP_factor = np.where(
        (conditions_matrix[:,9])
        &(conditions_matrix[:,10]),
        1,
        Monthly_Loss_CAP_factor
    )
    
    Monthly_Loss_CAP_factor = np.where(
        (conditions_matrix[:,9])
        &(~conditions_matrix[:,10]),
        0,
        Monthly_Loss_CAP_factor
    )
    
    Monthly_Loss_CAP_factor = np.where(
        (conditions_matrix[:,9])
        &(conditions_matrix[:,2]),
        1,
        Monthly_Loss_CAP_factor
    )

    return Monthly_Loss_CAP_factor

# Daily_Loss_CAP_factor_judger
def Daily_Loss_CAP_factor_judger(conditions_matrix:np.array):
    # 預設RTP因子值全部為 1
    Daily_Loss_CAP_factor = np.ones(conditions_matrix.shape[0], dtype=int)
    
    Daily_Loss_CAP_factor = np.where(
        (conditions_matrix[:,11])
        &(conditions_matrix[:,12]),
        1,
        Daily_Loss_CAP_factor
    )
    
    Daily_Loss_CAP_factor = np.where(
        (conditions_matrix[:,11])
        &(~conditions_matrix[:,12]),
        0,
        Daily_Loss_CAP_factor
    )
    
    Daily_Loss_CAP_factor = np.where(
        (conditions_matrix[:,11])
        &(conditions_matrix[:,2]),
        1,
        Daily_Loss_CAP_factor
    )

    return Daily_Loss_CAP_factor


# Sys_Win
def Sys_Win_judge(conditions_matrix:np.array, process_type:str, factor_value_m, factor_value_j):
    WIN_factor = np.zeros(conditions_matrix.shape[0], dtype=int)
    # 莊家收益是為正
    WIN_factor[
        conditions_matrix[:,1]
        ] = 1
    
    # 預設RTP因子值全部為 0
    RTP_factor = np.zeros(conditions_matrix.shape[0], dtype=int)

    RTP_factor[
        (conditions_matrix[:,3]) 
        ] = 1
    
    RTP_factor[
        (~conditions_matrix[:,3])
        &(conditions_matrix[:,5]) 
        ] = 1
    
    RTP_factor[
        (~conditions_matrix[:,3])
        &(~conditions_matrix[:,5]) 
        ] = 0
    
    return RTP_factor, WIN_factor


# Sys_Lose
def Sys_Lose_judge(conditions_matrix:np.array, process_type:str, factor_value_m, factor_value_j):
    WIN_factor = np.ones(conditions_matrix.shape[0], dtype=int)
    # 莊家收益是為正
    WIN_factor = np.where(
        conditions_matrix[:,1],
        0,
        WIN_factor
    )

    if conditions_matrix[:,4].all():
        RTP_factor = np.ones(conditions_matrix.shape[0], dtype=int)
    else:
        # 預設RTP因子值全部為 0
        RTP_factor = np.zeros(conditions_matrix.shape[0], dtype=int)

        RTP_factor = np.where(
            (~conditions_matrix[:,6]) 
            &(~conditions_matrix[:,5]),
            1,
            RTP_factor
        )

        RTP_factor = np.where(
            (~conditions_matrix[:,6]) 
            &(conditions_matrix[:,5]),
            0,
            RTP_factor
        )

        if np.any((conditions_matrix[:,6]) & (~conditions_matrix[:,7])):
            RTP_factor = np.where(
                (conditions_matrix[:,6])    
                &(~conditions_matrix[:,7]),
                np.round(factor_value_j[
                (conditions_matrix[:,6]) 
                &(~conditions_matrix[:,7])
                ],5),
                RTP_factor
            )
        else:
            pass
        
        RTP_factor = np.where(
            (conditions_matrix[:,6])    
            &(~conditions_matrix[:,7])
            &(~conditions_matrix[:,8]),
            0.0001,
            RTP_factor
        )

        RTP_factor = np.where(
            (conditions_matrix[:,6])    
            &(conditions_matrix[:,7]),
            1,
            RTP_factor
        )


        if np.any((conditions_matrix[:,6]) & (conditions_matrix[:,5])):
            RTP_factor = np.where(
                (conditions_matrix[:,6]) 
                &(conditions_matrix[:,5]),
                0,
                RTP_factor
            )
        else:
            pass

        if conditions_matrix[:,5].all():
            RTP_factor = np.ones(conditions_matrix.shape[0], dtype=int)
        else:
            pass


    return RTP_factor, WIN_factor


# Random
def Random_judge(conditions_matrix:np.array, process_type:str, factor_value_m, factor_value_j):
    WIN_factor = np.ones(conditions_matrix.shape[0], dtype=object)
    
    # 預設RTP因子值全部為 0
    RTP_factor = np.zeros(conditions_matrix.shape[0], dtype=object)

    # 月總RTP上限功能未開啓
    RTP_factor = np.where(
        ~conditions_matrix[:,6],
        Decimal(str(1)),
        RTP_factor
    )

    display(to_decimal(factor_value_j))
    if np.any((conditions_matrix[:,6]) & (~conditions_matrix[:,7])):
        RTP_factor = np.where(
            (conditions_matrix[:,6])    
            &(~conditions_matrix[:,7]),
            # 添加条件检查，如果索引结果为空则使用默认值
            to_decimal(factor_value_j),
            RTP_factor
        )
    else:
        pass

    RTP_factor = np.where(
        (conditions_matrix[:,6])    
        &(~conditions_matrix[:,7])
        &(~conditions_matrix[:,8]),
        Decimal(str(0.0001)),
        RTP_factor
    )

    RTP_factor = np.where(
        (conditions_matrix[:,6])    
        &(conditions_matrix[:,7]),
        Decimal(str(1)),
        RTP_factor
    )

    # 帶平均值的項，最後賦值

    # print('=======================',
    #       RTP_factor[
    #         (conditions_matrix[:,6])
    #         &(~conditions_matrix[:,5]) 
    #         ],
    #         '==============================')
    
    # print('************************',
    #       sum(RTP_factor[
    #         (conditions_matrix[:,6])
    #         &(~conditions_matrix[:,5]) 
    #         ]) / Decimal(str(len(RTP_factor[
    #         (conditions_matrix[:,6])
    #         &(~conditions_matrix[:,5]) 
    #         ]))),
    #         '****************************')

    if np.any((conditions_matrix[:,6]) & (~conditions_matrix[:,5])):
        RTP_factor = np.where(
            (conditions_matrix[:,6]) 
            &(conditions_matrix[:,5]),
            Decimal(str(sum(RTP_factor[
            (conditions_matrix[:,6])
            &(~conditions_matrix[:,5]) 
            ]) / Decimal(len(RTP_factor[
            (conditions_matrix[:,6])
            &(~conditions_matrix[:,5]) 
            ])))).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP) ,
            RTP_factor
        )
    else:
        pass
    
    if conditions_matrix[:,4].all():
        RTP_factor = np.ones(conditions_matrix.shape[0], dtype=int)
    else:
        pass


    return RTP_factor, WIN_factor

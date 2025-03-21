import numpy as np
from IPython.display import display
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP

np.set_printoptions(precision=5, suppress=True)

to_decimal = np.vectorize(lambda x: Decimal(str(x)))  # 避免浮點數誤差
quantize_decimal = np.vectorize(lambda x: x.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP))
quantize_decimal4 = np.vectorize(lambda x: x.quantize(Decimal('0.0001'), rounding=ROUND_CEILING))


def cond_judge(
        conditions_matrix:np.array, 
        is_Rookie_table:bool, 
        inputs
        ):
    
    if is_Rookie_table:
        Cap_break_factor = 1
        
    else:
        Cap_break_factor = Non_Rookie_judge(
            conditions_matrix,
            )

    Output_Factors = Cap_break_factor

    return Output_Factors


# Random
def Non_Rookie_judge(conditions_matrix:np.array):
    
    # 預設因子值全部為 0
    Cap_break_factor = Decimal(str(0))

    # 若使用新手表，則因子值為0
    Cap_break_factor = Cap_break_factor if ~conditions_matrix[0] else Decimal(str(0))
    
    # 當月RTP上限開啓,如果爆當月RTP上限則因子值為1
    Cap_break_factor = Decimal(str(1)) if conditions_matrix[1] and ~conditions_matrix[2] else Cap_break_factor

    # 當月系統虧損上限開啓,如果爆當月系統虧損上限則因子值為1
    Cap_break_factor = Decimal(str(1)) if conditions_matrix[3] and ~conditions_matrix[4] else Cap_break_factor

    # 當月個人盈利上限開啓,如果爆當月個人盈利上限則因子值為1
    Cap_break_factor = Decimal(str(1)) if conditions_matrix[5] and ~conditions_matrix[6] else Cap_break_factor
    
    return Cap_break_factor

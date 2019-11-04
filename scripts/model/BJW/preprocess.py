import pandas as pd
import numpy as np
import os
import sys

# data load
path = 'drive/My Drive/bigcontest2019/data/'
print(os.listdir(path + 'train'),'\n', 
      os.listdir(path + 'test'),'\n', 
      os.listdir(path + '../metrics'),'\n', 
      os.listdir(path + '../scripts'))

train_label = pd.read_csv(path + 'train/train_label.csv')
train_activity = pd.read_csv(path + 'train/train_activity.csv')
train_payment = pd.read_csv(path + 'train/train_payment.csv')
train_trade = pd.read_csv(path + 'train/train_trade.csv')
train_pledge = pd.read_csv(path + 'train/train_pledge.csv')
train_combat = pd.read_csv(path + 'train/train_combat.csv')


# activity
train_activity['game_money_change'] = np.abs(train_activity['game_money_change'])
train_activity_group = train_activity.groupby('acc_id').agg({'day':'nunique', 
                                                             'char_id':'nunique', 
                                                             'server':'nunique', 
                                                             'playtime':['sum','mean'], 
                                                             'npc_kill':['sum','mean'], 
                                                             'solo_exp':'sum',
                                                             'party_exp':'sum', 
                                                             'quest_exp':'sum',
                                                             'rich_monster':'sum', 
                                                             'death':'sum', 
                                                             'revive':'sum',
                                                             'exp_recovery':'sum',
                                                             'fishing':'sum',
                                                             'private_shop':'sum',
                                                             'game_money_change':'sum',
                                                             'enchant_count':'sum'})
train_activity_group.columns = ['activity_'+'_'.join(x) for x in train_activity_group.columns.ravel()]


# payment
train_payment_group = train_payment.groupby('acc_id').agg({'day':'nunique',
                                                           'amount_spent':'sum'})
train_payment_group.columns = ['payment_'+i for i in train_payment_group.columns]


# trade
## 판매자 테이블 정의
grouped_trade_seller = train_trade.groupby('source_acc_id').agg({'day':'count', 
                                         'type':['nunique','count','sum'], # 거래의 종류들을 파악하기 위해서 -> nuique = 2이면 두 종류의 거래 모두 진행 / count 
                                         'server':'nunique', 
                                         'source_char_id':'nunique', # 몇개의 캐릭터 운용하는지
                                         'target_acc_id':'nunique', 
                                         'target_char_id':'nunique',
                                         'item_type':'nunique', 
                                         'item_amount':'sum',
                                         'item_price':'sum'})

grouped_trade_seller.columns = ['trade_seller_'+'_'.join(x) for x in grouped_trade_seller.columns.ravel()]
grouped_trade_seller['trade_seller_type_count'] -= grouped_trade_seller['trade_seller_type_sum']
grouped_trade_seller = grouped_trade_seller.rename(columns = {'trade_seller_type_count':'trade_seller_type_personal','trade_seller_type_sum':'trade_seller_type_exchange'})


## 구매자 테이블 정의
grouped_trade_buyer = train_trade.groupby('target_acc_id').agg({'day':'count', 
                                         'type':['nunique','count','sum'], # 거래의 종류들을 파악하기 위해서 -> nuique = 2이면 두 종류의 거래 모두 진행 / count 
                                         'server':'nunique', 
                                        'target_char_id':'nunique',
                                         'source_acc_id':'nunique', 
                                         'source_char_id':'nunique', # 몇개의 캐릭터 운용하는지
                                         'item_type':'nunique', 
                                         'item_amount':'sum',
                                          'item_price':'sum'})
grouped_trade_buyer.columns = ['trade_buyer_'+'_'.join(x) for x in grouped_trade_buyer.columns.ravel()]
grouped_trade_buyer['trade_buyer_type_count'] -= grouped_trade_buyer['trade_buyer_type_sum']
grouped_trade_buyer = grouped_trade_buyer.rename(columns = {'trade_buyer_type_count':'trade_buyer_type_personal','trade_buyer_type_sum':'trade_buyer_type_exchange'})

## trade merge
train_trade_group = pd.concat([grouped_trade_seller, grouped_trade_buyer], axis=1)


# pledge
train_pledge_group = train_pledge.groupby('acc_id').agg(
    {
        'day': 'nunique',                   # 날짜
        'char_id': 'nunique',               # 캐릭터 아이디
        'server' : 'nunique',               # 캐릭터 서버
        'pledge_id' : 'nunique',            # 혈맹 아이디
        'play_char_cnt' : 'sum',        # 게임에 접속한 혈맹원 수
        'combat_char_cnt' : 'sum',      # 전투에 참여한 혈맹원 수
        'pledge_combat_cnt': 'sum',     # 혈맹 간 전투 횟수의 합
        'random_attacker_cnt' : 'sum',  # 혈맹원 중 막피 전투를 행한 횟수의 합
        'random_defender_cnt': 'sum',   # 혈맹원 중 막피로부터 피해를 받은 횟수의 합
        'same_pledge_cnt': 'sum',       # 동일 혈맹원 간 전투 횟수의 합
        'temp_cnt' : 'sum',             # 혈맹원들의 단발성 전투 횟수의 합
        'etc_cnt' : 'sum',              # 혈맹원들의 기타 전투 횟수의 합
        'combat_play_time': 'sum',      # 혈맹의 전투 캐릭터들의 플레이 시간의 합
        'non_combat_play_time' : 'sum' # 혈맹의 非전투 캐릭터 플레이 시간의 합
    })

train_pledge_group.columns = ['pledge_'+i for i in train_pledge_group.columns]


# combat
train_combat_group = train_combat.groupby('acc_id').agg(
    {
        'day': 'nunique',              # 날짜
        'char_id': 'nunique',          # 캐릭터 아이디
        'server' : 'nunique',          # 캐릭터 서버
        'class' : 'nunique',           # 직업
        'pledge_cnt' : 'sum',          # 혈맹간 전투에 참여한 횟수
        'random_attacker_cnt' : 'sum', # 본인이 막피 공격을 행한 횟수
        'random_defender_cnt' : 'sum', # 막피 공격자로부터 공격을 받은 횟수
        'temp_cnt' : 'sum',            # 단발성 전투 횟수
        'same_pledge_cnt' : 'sum',     # 동일 혈맹원 간의 전투 횟수
        'etc_cnt' : 'sum',             # 기타 전투 횟수
        'num_opponent' : 'sum'         # 전투 상대 캐릭터수
    })
train_combat_group.columns = ['combat_'+i for i in train_combat_group.columns]

def merge_all_df(df, by ,*args):
    '''
    df : 기준이 되는 데이터 프레임(left join)
    by : 기준이 되는 열 (df 외 타 args(df) 에도 해당 열 존재하게 생성할 것)
    *args : 가변인자 (데이터 프레임을 원하는 만큼 집어넣으면 된다.)
    '''    
    for arg in args:
        if by not in arg.columns:
            arg[by] = arg.index
            del arg.index.name
        df = pd.merge(df, arg, how='left',on= by)
    return df

train = merge_all_df(train_label,'acc_id',train_trade_group,
                     train_payment_group,train_activity_group,
                     train_combat_group,train_pledge_group).fillna(0)

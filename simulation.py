import random
from enum import Enum
from typing import List, Tuple, Dict

# 牌型定義
class CardType(Enum):
    HEAVEN = 0       # 天牌
    EARTH = 1        # 地牌
    HUMAN = 2        # 人牌
    GOOSE = 3        # 鵝牌
    PLUM = 4         # 梅牌
    LONG3 = 5        # 長三
    BENCH = 6        # 板凳
    AXE = 7          # 斧頭
    RED10 = 8        # 紅頭十
    HIGH7 = 9        # 高腳七
    ZERO6 = 10       # 零霸六
    MISC9_54 = 11       # 雜九
    MISC9_63 = 12       # 雜九
    MISC8_53 = 13       # 雜八
    MISC8_62 = 14       # 雜八
    MISC7_43 = 15       # 雜七
    MISC7_52 = 16       # 雜七
    MISC5_41 = 17       # 雜五
    MISC5_32 = 18       # 雜五
    D24 = 19         # 二四
    D13 = 20         # 丁三

# 牌型點數和數量
CARD_POINTS = {
    CardType.HEAVEN: 12,
    CardType.EARTH: 2,
    CardType.HUMAN: 8,
    CardType.GOOSE: 4,
    CardType.PLUM: 10,
    CardType.LONG3: 6,
    CardType.BENCH: 4,
    CardType.AXE: 11,
    CardType.RED10: 10,
    CardType.HIGH7: 7,
    CardType.ZERO6: 6,
    CardType.MISC9_54: 9,
    CardType.MISC9_63: 9,
    CardType.MISC8_53: 8,
    CardType.MISC8_62: 8,
    CardType.MISC7_43: 7,
    CardType.MISC7_52: 7,
    CardType.MISC5_41: 5,
    CardType.MISC5_32: 5,
    CardType.D24: 6,
    CardType.D13: 3,
}

CARD_COUNTS = {
    CardType.HEAVEN: 2,
    CardType.EARTH: 2,
    CardType.HUMAN: 2,
    CardType.GOOSE: 2,
    CardType.PLUM: 2,
    CardType.LONG3: 2,
    CardType.BENCH: 2,
    CardType.AXE: 2,
    CardType.RED10: 2,
    CardType.HIGH7: 2,
    CardType.ZERO6: 2,
    CardType.MISC9_54: 1,
    CardType.MISC9_63: 1,
    CardType.MISC8_53: 1,
    CardType.MISC8_62: 1,
    CardType.MISC7_43: 1,
    CardType.MISC7_52: 1,
    CardType.MISC5_41: 1,
    CardType.MISC5_32: 1,
    CardType.D24: 1,
    CardType.D13: 1,
}

class Card:
    def __init__(self, card_type: CardType):
        self.card_type = card_type
        self.points = CARD_POINTS[card_type]
    
    def __str__(self):
        return f"{self.card_type.name}({self.points})"

class Hand:
    def __init__(self, cards: List[Card]):
        self.cards = cards
        self.is_pair = self._check_pair()
        self.pair_name = self._get_pair_name()
        self.total_points = self._calculate_points()
        self.max_card = max(cards, key=lambda x: x.points)
    
    def _check_pair(self) -> bool:
        # 檢查是否為對子
        if len(self.cards) != 2:
            return False
            
        # 同牌對子
        if self.cards[0].card_type == self.cards[1].card_type:
            return True
        
        # 雜五對子
        if self.cards[0].card_type == CardType.MISC5_32 and self.cards[1].card_type == CardType.MISC5_41:
            return True
        
        # 雜七對子
        if self.cards[0].card_type == CardType.MISC7_52 and self.cards[1].card_type == CardType.MISC7_43:
            return True
        
        # 雜八對子
        if self.cards[0].card_type == CardType.MISC8_62 and self.cards[1].card_type == CardType.MISC8_53:
            return True
        
        # 雜九對子
        if self.cards[0].card_type == CardType.MISC9_63 and self.cards[1].card_type == CardType.MISC9_54:
            return True
        
        # 特殊組合對子
        card_types = {card.card_type for card in self.cards}
        
        # 至尊: 丁三+二四
        if CardType.D13 in card_types and CardType.D24 in card_types:
            return True

        # 天皇: 天牌+雜九
        if CardType.HEAVEN in card_types and (CardType.MISC9_54 in card_types or CardType.MISC9_63 in card_types):
            return True
            
        # 地皇: 地牌+雜九
        if CardType.EARTH in card_types and (CardType.MISC9_54 in card_types or CardType.MISC9_63 in card_types):
            return True
            
        # 天杠: 天牌+雜八或人牌
        if CardType.HEAVEN in card_types and (CardType.MISC8_53 in card_types or CardType.MISC8_62 in card_types or CardType.HUMAN in card_types):
            return True
            
        # 地杠: 地牌+雜八或人牌
        if CardType.EARTH in card_types and (CardType.MISC8_53 in card_types or CardType.MISC8_62 in card_types or CardType.HUMAN in card_types):
            return True
            
        # 天高九: 天牌+二五牌
        if CardType.HEAVEN in card_types and CardType.MISC7_52 in card_types:
            return True
            
        # 地高九: 地牌+高腳七
        if CardType.EARTH in card_types and CardType.HIGH7 in card_types:
            return True
            
        return False
    
    def _get_pair_name(self) -> str:
        # 如果不是對子，返回空
        if not self.is_pair:
            return "散牌"
            
        # 同牌對子
        if self.cards[0].card_type == self.cards[1].card_type:
            return f"{self.cards[0].card_type.name}對"
        
        # 雜五對子
        if self.cards[0].card_type == CardType.MISC5_32 and self.cards[1].card_type == CardType.MISC5_41:
            return "雜五對"
        
        # 雜七對子
        if self.cards[0].card_type == CardType.MISC7_52 and self.cards[1].card_type == CardType.MISC7_43:
            return "雜七對"     

        # 雜八對子
        if self.cards[0].card_type == CardType.MISC8_62 and self.cards[1].card_type == CardType.MISC8_53:
            return "雜八對"
        
        # 雜九對子
        if self.cards[0].card_type == CardType.MISC9_63 and self.cards[1].card_type == CardType.MISC9_54:
            return "雜九對"

        # 特殊組合對子
        card_types = {card.card_type for card in self.cards}
        
        if CardType.D13 in card_types and CardType.D24 in card_types:
            return "至尊"
        if CardType.HEAVEN in card_types and CardType.MISC9_54 in card_types:
            return "天皇"
        if CardType.EARTH in card_types and CardType.MISC9_54 in card_types:
            return "地皇"
        if CardType.HEAVEN in card_types and (CardType.MISC8_53 in card_types or CardType.MISC8_62 in card_types or CardType.HUMAN in card_types):
            return "天杠"
        if CardType.EARTH in card_types and (CardType.MISC8_53 in card_types or CardType.MISC8_62 in card_types or CardType.HUMAN in card_types):
            return "地杠"
        if CardType.HEAVEN in card_types and CardType.MISC7_52 in card_types:
            return "天高九"
        if CardType.EARTH in card_types and CardType.HIGH7 in card_types:
            return "地高九"
            
        return "散牌"  # 不應該到達這裡
    
    def _calculate_points(self) -> int:
        # 對子有特定點數
        if self.is_pair:
            pair_name = self._get_pair_name()
            if pair_name == "至尊":
                return 99  
            elif pair_name == "HEAVEN對":
                return 98   
            elif pair_name == "EARTH對":
                return 97  
            elif pair_name == "HUMAN對":
                return 96  
            elif pair_name == "GOOSE對":
                return 95  
            elif pair_name == "PLUM對":
                return 94  
            elif pair_name == "LONG3對":
                return 93  
            elif pair_name == "BENCH對":
                return 92  
            elif pair_name == "AXE對":
                return 91  
            elif pair_name == "RED10對":
                return 90  
            elif pair_name == "HIGH7對":
                return 89     
            elif pair_name == "ZERO6對":
                return 88  
            elif pair_name == "雜九對":
                return 87  
            elif pair_name == "雜八對":
                return 86  
            elif pair_name == "雜七對":
                return 85  
            elif pair_name == "雜五對":
                return 84  
            elif pair_name == "天皇":
                return 83  
            elif pair_name == "地皇":
                return 82  
            elif pair_name == "天杠":
                return 81  
            elif pair_name == "地杠":
                return 80  
            elif pair_name == "天高九":
                return 79  
            elif pair_name == "地高九":
                return 78  

        # 普通牌型或普通對子，計算總點數取個位數
        total = sum(card.points for card in self.cards)
        return total % 10  # 只取個位數
    
class Player:
    def __init__(self, name: str, balance: int):
        self.name = name
        self.balance = balance
        self.hand = None
        self.bet = 0
        self.banker_multiple = 0  # 搶莊倍率
    
    def set_hand(self, hand: Hand):
        self.hand = hand
    
    def place_bet(self, amount: int):
        self.bet = amount
        return amount
    
    def bid_for_banker(self, multiple: int):
        self.banker_multiple = multiple
        return multiple
    
    def win(self, amount: int):
        self.balance += amount
    
    def lose(self, amount: int):
        self.balance -= amount
    
    def __str__(self):
        return f"{self.name} (餘額: {self.balance})"

class Game:
    def __init__(self, player_names: List[str], initial_balance: int = 1000):
        self.players = [Player(name, initial_balance) for name in player_names]
        self.deck = self._create_deck()
        self.banker = None
        self.banker_multiple = 0
        self.round = 0
    
    def _create_deck(self) -> List[Card]:
        # 創建一副牌
        deck = []
        for card_type in CardType:
            for _ in range(CARD_COUNTS[card_type]):
                deck.append(Card(card_type))
        return deck
    
    def shuffle(self):
        random.shuffle(self.deck)
    
    def deal_cards(self):
        self.shuffle()
        for player in self.players:
            cards = [self.deck.pop(), self.deck.pop()]
            player.set_hand(Hand(cards))
    
    def banker_bidding(self):
        print("搶莊階段:")
        possible_multiples = [0, 1, 2, 3]  # 可用的搶莊倍率
        
        # 玩家選擇搶莊倍率
        bids = []
        for player in self.players:
            # 隨機選擇搶莊倍率 (在實際遊戲中，這應該由玩家決定)
            bid = random.choice(possible_multiples)
            player.bid_for_banker(bid)
            bids.append((player, bid))
            print(f"{player.name} 搶莊倍率: {bid}倍")
        
        # 找出最高倍率的玩家
        max_bid = max(bids, key=lambda x: x[1])[1]
        highest_bidders = [p for p, b in bids if b == max_bid]
        
        # 如果有多個玩家選擇了最高倍率，隨機選擇其中一位
        self.banker = random.choice(highest_bidders)
        self.banker_multiple = max_bid
        
        print(f"{self.banker.name} 成為莊家，倍率 {self.banker_multiple}倍")
    
    def player_betting(self):
        print("\n玩家下注階段:")
        max_bet_multiple = 3  # 最大下注是莊家倍率的3倍
        
        for player in self.players:
            if player != self.banker:
                # 隨機選擇下注倍率 (實際遊戲中由玩家決定)
                bet_multiple = random.randint(1, max_bet_multiple)
                bet_amount = bet_multiple * 10  # 假設基本注額為10
                player.place_bet(bet_amount)
                print(f"{player.name} 下注: {bet_amount}")
    
    def compare_hands(self, hand1: Hand, hand2: Hand) -> int:
        """比較兩個牌型，返回1如果hand1贏，-1如果hand2贏，0如果平局"""
        # 對子比散牌
        if hand1.is_pair and not hand2.is_pair:
            return 1
        if not hand1.is_pair and hand2.is_pair:
            return -1
        
        # 都是對子，比較點數
        if hand1.is_pair and hand2.is_pair:
            if hand1.total_points > hand2.total_points:
                return 1
            elif hand1.total_points < hand2.total_points:
                return -1
            return 0
        
        # 都是散牌，先比較點數
        if hand1.total_points > hand2.total_points:
            return 1
        elif hand1.total_points < hand2.total_points:
            return -1
        
        # 點數相同，比較最大的單牌
        if hand1.max_card.points > hand2.max_card.points:
            return 1
        elif hand1.max_card.points < hand2.max_card.points:
            return -1
        
        return 0
    
    def settlement(self):
        print("\n結算階段:")
        
        banker_hand = self.banker.hand
        print(f"莊家 {self.banker.name} 的牌: {banker_hand}")
        
        for player in self.players:
            if player != self.banker:
                print(f"閒家 {player.name} 的牌: {player.hand}")
                
                result = self.compare_hands(player.hand, banker_hand)
                win_amount = player.bet * (1 + self.banker_multiple)
                
                if result == 1:  # 閒家贏
                    player.win(win_amount)
                    self.banker.lose(win_amount)
                    print(f"{player.name} 贏得 {win_amount}，餘額: {player.balance}")
                    print(f"{self.banker.name} 輸掉 {win_amount}，餘額: {self.banker.balance}")
                elif result == -1:  # 莊家贏
                    player.lose(win_amount)
                    self.banker.win(win_amount)
                    print(f"{player.name} 輸掉 {win_amount}，餘額: {player.balance}")
                    print(f"{self.banker.name} 贏得 {win_amount}，餘額: {self.banker.balance}")
                else:  # 平局
                    print(f"{player.name} 和 {self.banker.name} 平局，不輸不贏")
    
    def play_round(self):
        self.round += 1
        print(f"\n========== 第 {self.round} 輪遊戲 ==========")
        
        # 重置遊戲狀態
        self.deck = self._create_deck()
        for player in self.players:
            player.hand = None
            player.bet = 0
            player.banker_multiple = 0
        
        # 發牌
        self.deal_cards()
        
        # 展示每個玩家的牌 (這部分在實際遊戲中應該隱藏)
        for player in self.players:
            print(f"{player.name} 的牌: {player.hand}")
        
        # 搶莊
        self.banker_bidding()
        
        # 玩家下注
        self.player_betting()
        
        # 結算
        self.settlement()
        
        # 顯示玩家餘額
        print("\n玩家餘額:")
        for player in self.players:
            print(f"{player.name}: {player.balance}")

# 主函數
def main():
    # 創建4個玩家
    player_names = ["玩家1", "玩家2", "玩家3", "玩家4"]
    game = Game(player_names)
    
    # 進行3輪遊戲
    for _ in range(3):
        game.play_round()
        input("\n按Enter繼續...")

if __name__ == "__main__":
    main()
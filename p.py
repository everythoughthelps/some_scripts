import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--echo')
args = parser.parse_args()

print(args)
print(args.echo)

def func():
    print(args.echo)    

func()

class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    @staticmethod
    def speak():
        print("说: 我 岁。") 
    @classmethod
    def run(cls):
        print(type(cls),cls) 
# 实例化类
p = people('runoob',10,30)
p.speak()

people.run()

p.run()



from collections import namedtuple

# 使用命名元组，可以简单的构建一个对象
Card = namedtuple("Card", ["rank", "suit"])

class FrenchDeck:
    # 2-A
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    # 4种花色
    suits = "spades diamonds clubs hearts".split()

    def __init__(self):
        # 构建扑克牌
        self._cards = [Card(rank, suit) for suit in self.suits 
                       for rank in self.ranks]
        print(self._cards)
    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        print(position)
        return self._cards[position]

print(FrenchDeck.ranks,FrenchDeck.suits)

deck = FrenchDeck()
print(Card)
print(len(deck))
for i in deck:
    print(i)

#print ('***获取当前目录***')
#print (os.getcwd())
#print (os.path.abspath(os.path.dirname(__file__)))
#print ('***获取上级目录***')
#print (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
#print (os.path.abspath(os.path.dirname(os.getcwd())))
#print (os.path.abspath(os.path.join(os.getcwd(), "..")))
#print ('***获取上上级目录***')
#print (os.path.abspath(os.path.join(os.getcwd(), "../..")))

def test():
    list1 = ['a','b','c','d','e']
    list1_iter = iter(list1)

if __name__ == '__main__':
    test()
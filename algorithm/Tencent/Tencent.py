'''
给定一个整数数组[a1,a2,.....aN] ，N个数,  现在从里面选择若干数使得他们的和最大，
同时满足相邻两数不能同时被选， a1和aN首尾两个也认为是相邻的。
'''
import random

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def main():
    N = 10
    A = random_int_list(-20, 20, N)
    maxsum = MAXSUBSEQSUM1(A, N)
    print(maxsum)

def MAXSUBSEQSUM1(A, N):
    ThisSum, MaxSum, MaxSum_dan,MaxSum_shuang = 0, 0, 0, 0
    for i in range(N):
        for j in range(i+1, N):
            ThisSum = 0
            for k in range(i, j + 1):
                if k // 2 == 1:
                    ThisSum += A[k]
                    if ThisSum > MaxSum:
                        MaxSum_dan = ThisSum
                if k //2 == 0:
                    ThisSum += A[k]
                    if ThisSum > MaxSum:
                        MaxSum_shuang = ThisSum
                MaxSum = max(MaxSum_shuang,MaxSum_dan)
    return MaxSum

main()
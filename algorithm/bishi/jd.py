import sys

while True:
    try:
        m,n = map(int,input().split())
        result = list()
        for i in range(m,n+1):
            a = i//100
            b = i//10-10*a
            c = i - 10 * b - 100*a
            if a ** 2 + b ** 2 + c **2 == i:
                result.append(i)
        if len(result)==0:
            sys.stdout.write('no')
        else:
            for i in result:
                sys.stdout.write(str(i)+' ')
    except:
        break
import sys

while True:
    try:
        n = int(input())
        lines = []
        for i in range(n):
            line = list(map(int,input().split()))
            lines.append(line)
        dp = []
        a = 0
        for i in range(n):
            if i == 0:
                dp[i] = lines[i][i]
                a = i
            else:
                dp[i] = dp[i-1] + max(lines[i][a],lines[i][a+1],lines[a+2])
                a = lines[i].index(max(lines[i][a],lines[i][a+1],lines[a+2]))
        sys.stdout.write(str(dp[-1]))
    except:
        break

import sys

while True:
    try:
        n = int(input())
        lines = []
        for i in range(n):
            line = list(map(int,input().split()))
            if line[0] == 1:
                a = line[1]
                b = line[2]
                lines[a] = b
            elif line[0] == 2:
                x = line[1]
                lines.remove(x)
            else:
                for i ,x in enumerate(lines):
                    sys.stdout.write(str(x)+' ')
    except:
        break
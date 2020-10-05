t = int(input())
for i in range(t):
    n = int(input())
    circles = []

    for i in range(n):
        num = n
        circle = list(map(int,input().split()))
        circle = sorted(circle)
        circles.append(circle)
        if i > 0:
            for j in range(i):
                count = 0
                for a in range(6):
                    if circle[a] != circles[j][a]:
                        break
                    else:
                        count = count + 1
                        continue
                if count == 6:
                    print('YES')
                    break
                break
            num = num -1
        if num == 1:
            print('NO')



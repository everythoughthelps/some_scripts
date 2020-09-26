t = int(input())
for i in range(t):
    lines = []
    n = int(input())
    for j in range(n):
        line = list(map(int,input().split()))
        lines.append(line)

    colums = []
    for k in range(n):
        colum = lines[n,:]
        colum_sum = sum(colum)
        colums.append(colum_sum)

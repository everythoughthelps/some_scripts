while True:
    try:
        n = int(input())
        red = []
        blue = []
        lines = []
        for i in range(n):
            line = list(map(int,input().split()))
            lines.append(line)
            if line[1] == 1:
                red.append(line)
            else:
                blue.append(line)
        blue_sort = sorted(blue)
        red_sort = sorted(red)
        max_blue = 0
        max_red = 0
        if len(blue) >= 3:
            max_blue = blue_sort[-1][0] + blue_sort[-2][0] + blue_sort[-3][0]
        if len(red) >= 3:
            max_red = red_sort[-1][0] + red_sort[-2][0] + red_sort[-3][0]

        if max_blue > max_red:
            max_result = max_blue
            candy = 2
            index1 = lines.index(blue_sort[-3]) + 1
            index2 = lines.index(blue_sort[-2]) + 1
            index3 = lines.index(blue_sort[-1]) + 1
            print(index1,index2,index3)
            print(candy)
            print(max_blue)
        elif max_blue == max_red:
            index_red = min(lines.index(red_sort[-3]),lines.index(red_sort[-2]),lines.index(red_sort[-1]))
            index_blue = min(lines.index(blue_sort[-3]),lines.index(blue_sort[-2]),lines.index(blue_sort[-1]))
            if index_red < index_blue:
                max_result = max_red
                candy = 1
                index1 = lines.index(red_sort[-3]) + 1
                index2 = lines.index(red_sort[-2]) + 1
                index3 = lines.index(red_sort[-1]) + 1
                print(index1,index2,index3)
                print(candy)
                print(max_red)
            else:
                max_result = max_blue
                candy = 2
                index1 = lines.index(blue_sort[-3]) + 1
                index2 = lines.index(blue_sort[-2]) + 1
                index3 = lines.index(blue_sort[-1]) + 1
                print(index1,index2,index3)
                print(candy)
                print(max_blue)
        else:
            max_result = max_red
            candy = 1
            index1 = lines.index(red_sort[-3]) + 1
            index2 = lines.index(red_sort[-2]) + 1
            index3 = lines.index(red_sort[-1]) + 1
            print(index1,index2,index3)

            print(candy)
            print(max_red)
    except:
        break

while True:
    try:
        k = int(input())
        n = int(input())
        w = list(map(int,input().split()))
        v = list(map(int,input().split()))
        dp = [[0 for _ in range(k+1)] for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,k+1):
                if j < w[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = max(dp[i-1][j-w[i-1]]+v[i-1],dp[i-1][j])
        print(dp[n][k])
        pass
    except:
        break


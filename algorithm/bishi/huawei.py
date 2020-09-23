while True:
    try:
        check_str , target = input().split()
        origin_str = check_str
        length = int(len(target) * 0.8)
        count = 0
        start = 0
        end = 0
        for i,x in enumerate(check_str):
            for j,y in enumerate(target):
                print(start,end)
                if x == y :
                    if count == 0:
                        start = i
                    count += 1
                    target = target[j+1:]
                    check_str = check_str[i+1:]
                    end = start + 1
                    break
        if count >= length:
            res_str = origin_str[0:start] + '*' * (end - start) + origin_str[end:]

        print(res_str)
    except:
        break

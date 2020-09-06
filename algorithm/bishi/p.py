while True:
    try:
        line = list(map(int,input().split()))
        n = line[0]
        m = line[1]
        k = line[2]

        def factorial(n):
            if  n == 0:
                return 1
            else:
                return factorial(n-1) * n
        def permutaion(n,m):
            n_fac = factorial(n)
            m_fac = factorial(m)
            diff_fac = factorial(n - m)
            return n_fac / (m_fac * diff_fac)

        sum = 0

        for i in range(3,n):
            if k - i < 2:
                break
            else:
                if k - i > m :
                    continue
                else:
                    sum = sum + permutaion(n,i) * permutaion(m,k-i)
        print(sum)
    except:
        break


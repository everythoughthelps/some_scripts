
s = 'nfpdmp'
def lengthOfLongestSubstring( s: str) -> int:
    l = []
    previous_length = 0
    for x in s:
        if x not in l:
            l.append(x)
        else:
            index = l.index(x)
            l = l[index + 1:]
            l.append(x)
        current_length = len(l)
        previous_length = max(current_length, previous_length)
    print(previous_length)

def permutation(str,start,end):

    if(start==end):
        for s in str:
            print(s,end='')
        print('')
        return

    for i in range(start,end+1):
        if not isrepeat(str,start,i):
            continue
        str[start],str[i]=str[i],str[start]
        permutation(str,start+1,end)
        str[start], str[i] = str[i], str[start]

def isrepeat(str,start,i):
    bcan=True
    #第i个字符与第j个字符交换时，要求[i,j)中没有与第j个字符相等的数
    for j in range(start, i):
        if str[start]==str[i]:
            bcan=False
            break
    return bcan

a=[1,2,3,4,4]
permutation(a,1,3)

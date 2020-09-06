
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

class permutation(object):

    def isrepeat(self,str,start,i):
        bcan=True
        #第i个字符与第j个字符交换时，要求[i,j)中没有与第j个字符相等的数
        for j in range(start, i):
            if str[start]==str[i]:
                bcan=False
                break
        return bcan

    def permutation(self,str,start,end):

        if(start==end):
            for s in str:
                print(s,end='')
            print('')
            return

        for i in range(start,end+1):
            if not self.isrepeat(str,start,i):
                continue
            str[start],str[i]=str[i],str[start]
            self.permutation(str,start+1,end)
            str[start], str[i] = str[i], str[start]


class Solution:


    def longestPalindrome(self, s: str) -> str:
        '''''''''
        longest_str = ''
        max_length = 0
        if not s:
            longest_str = ''
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                if s[i:j] == s[i:j][::-1]:
                    if len(s[i:j]) > max_length:
                        max_length = len(s[i:j])
                        longest_str = s[i:j]
        '''''
        left ,right = 0,0
        odd_length = 0
        even_length = 0
        max_length = 0
        for i in range(len(s)):
            L , R = i , i
            while  L >= 0 and R < len(s) and s[L] == s[R]:
                L = L - 1
                R = R + 1
            odd_length = R - L - 1
            l , r = i , i
            while l >= 0 and r + 1 <len(s) and s[l] == s[r + 1]:
                l = l - 1
                r = r + 1
            even_length = r - l
            max_length = max(odd_length,even_length)
            if max_length > right - left:
                left = i - (max_length-1)// 2
                right = i + max_length // 2
        print(s[left:right + 1])
        return s[left:right]

row = int(input())
lines_multrows = []
for n in range(row):    #没有n 用while true 也ok
    line = input()
    if line=='':
        break
    lines = line.split()
    lines = list(map(int,lines))
    lines_multrows.append(lines) #多行数组

new_list = dict()

for i, x in enumerate(lines_multrows):
    if x[0] in new_list:
        new_list[x[0]] = new_list[x[0]] + x[1]
    else:
        new_list[x[0]] = x[1]
for i,x in enumerate(lines_multrows):
    print(int(x),int(new_list[x]))



class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        s = ''
        a = zip(*strs)
        for i in a:
            if len(i) == 1:
                s += i[0]
            else:
                break
        return s


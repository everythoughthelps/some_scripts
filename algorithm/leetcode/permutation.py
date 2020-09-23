
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

class Solution:
	def fib(self, n: int) -> int:
		fib_result = []
		for i in range(n + 1):
			if i == 0:
				fib_result[i] = 0
			if i == 1:
				fib_result[i] = 1
			else:
				fib_result[i] = fib_result[i - 1] + fib_result[i - 2]
		return fib_result[n]


class Solution:
	def numWays(self, n: int) -> int:
		res = [0 for _ in range(n + 1)]
		for i in range(n + 1):
			if i == 0:
				res[0] = 1
			elif i == 1:
				res[1] = 1
			else:
				res[i] = res[i - 1] + res[i - 2]
			print(res)
			return res[n]

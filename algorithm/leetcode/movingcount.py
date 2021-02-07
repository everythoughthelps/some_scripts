class Solution:
	def movingCount(self, m: int, n: int, k: int) -> int:
		def sum_num(x):
			a = x // 10
			b = x % 10
			return a + b

		matrix = [[0 for _ in range(n)] for _ in range(m)]
		print(matrix)

		def access(i, j, m, n):
			if 0 <= i <= m and 0 <= j <= n:
				if sum_num(i) + sum_num(j) <= k:
					matrix[i][j] = True
					access(i - 1, j, m, n)
					access(i + 1, j, m, n)
					access(i, j - 1, m, n)
					access(i, j + 1, m, n)

		access(0, 0, m, n)
		print(matrix)
List = [[]]
class Solution:
	def exist(self, board: List[List[str]], word: str) -> bool:

		def help(board, x, y, word, i, visited):
			if i >= len(word):
				return True

			if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]):
				return False

			if visited[x][y]:
				return False

			if board[x][y] != word[i]:
				return False

			visited[x][y] = True

			# print(x, y)

			res = help(board, x-1, y, word, i+1, visited) or help(board, x+1, y, word, i+1, visited) or help(board, x, y-1, word, i+1, visited) or help(board, x, y+1, word, i+1, visited)
			if not res:
				visited[x][y] = False
			return res

		for i in range(len(board)):
			for j in range(len(board[0])):
				if board[i][j] == word[0]:
					visited = [[False for _ in range(len(board[0]))] for _ in range(len(board))]
					if help(board, i, j, word, 0, visited):
						return True
		return False

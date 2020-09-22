List = [[]]
class Solution:
	def exist(self, board: List[List[str]], word: str) -> bool:
		def find(i,j,k):
			if not 0 <= i <len(board) or 0 <= j < len(board[0]) or board[i][j] != word[k]:
				return False
			if k == len(word) - 1:
				return True
			tmp,board[i][j] = board[i][j],'/'
			result = find(i,j-1,k+1) or find(i,j+1,k+1) or find(i-1,j,k+1) or find(i+1,j,k+1)
			board[i][j] = tmp
			return result

		for i in range(board):
			for j in range(board[0]):
				if board[i][j] == word[0]:
					if find(i,j,0):
						return True
		return False



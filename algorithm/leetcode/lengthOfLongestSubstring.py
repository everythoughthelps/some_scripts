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

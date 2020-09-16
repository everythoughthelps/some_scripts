import sys
a = [1,2,3]
b = a

def test_id(x):
	print(hex(id(x)))

test_id(a)
test_id(a[1])
test_id(b)
print('modify b')

b[0] = 10
test_id(b)
print(a)



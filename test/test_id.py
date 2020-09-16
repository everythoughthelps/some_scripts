import sys
a = [1,2,3]
b = a

def test_id_changeable(x):
	x.pop()
	print(x)
	print(hex(id(x)))
	x = [1,2]
	print(hex(id(x)))


p = 1
q = p

def test_id_unchangeable(x):
	x = x + 1
	print(x)
	print(hex(id(x)))

print(hex(id(p)))
print(hex(id(q)))
test_id_unchangeable(p)
print(hex(id(q)))
print(q)


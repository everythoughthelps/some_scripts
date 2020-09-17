import sys
a = [1,2,3]
b = a

def test_id_changeable(x):
	print(hex(id(x)))
	x.append(1)
	print(sys.getsizeof(x))
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
print(sys.getsizeof(a))
print(sys.getsizeof(b))
test_id_changeable(a)


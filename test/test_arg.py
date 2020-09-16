def pet(*args):
	for i in args:
		print('I have{0}'.format(i))

pet('cat','dog','pig')


def other_per(**kwargs):
	print(kwargs.items())
	print(kwargs)
	for key,value in kwargs.items():
		print('{0} has {1}'.format(key,value))

other_per(tom = 'cat',jerry = 'mouse')
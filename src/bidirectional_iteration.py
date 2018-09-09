import os

os.system('python3 validation_bidirectional.py dx dy clf0')

for i in range(500):
	os.system('python3 validation_bidirectional.py dx dy clf'+str(i+1)+ ' clf'+str(i))

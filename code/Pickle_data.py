from preprocess import Load_data
import cPickle as pickle
import sys

width = 200
height = 300
output = sys.argv[1]

# pickle data for later training
ld = Load_data((width,height))
X,y = ld.load('all_metas.csv','images',cols=['Horror','Romance','Adventure','Crime'])

with open('./data/{}'.format(output),'wb') as f:
	pickle.dump((X,y),f)
	f.close()
print X.shape

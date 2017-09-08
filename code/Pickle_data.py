from preprocess import Load_data
import cPickle as pickle
import sys


'''
Load and pickle data for later training
Genres: Horror,Romance,Adventure,Documentary
Input: meta_data, images_folder
Output: pickle data
'''

width = 200
height = 300
meta_data = sys.argv[1]
output = sys.argv[2]

ld = Load_data((width,height),['Horror','Romance','Adventure','Documentary'])
X,y = ld.load(meta_data,'images')

with open('./data/{}'.format(output),'wb') as f:
	pickle.dump((X,y),f)
	f.close()
print X.shape

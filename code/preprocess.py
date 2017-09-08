from PIL import Image
import glob
import pandas as pd
import numpy as np
import re

# generate 2d numpy array for images and create labels
class Load_data(object):
    '''
    Convert images to 2D arrays and create labels from meta_data
    '''

    def __init__(self,size,cols=None,threshold=1000,multi_label=False):
        '''
        size : size of rescaling images
        threshold: the minimum number of movies for a genre to be included
        cols: genres to be selected from meta data
        multi_label: multiple-hot encoding
        '''

        self.size = size
        self.cols = cols
        self.threshold = threshold
        self.multi_label = multi_label

    def load(self,meta_path,img_folder):
        '''
        Vectorize genres information from movies meta data and convert corresponding images to 2D arrays
        Parameters:
        meta_path: path for meta data
        img_folder: folder for images
        '''

        self.genres = self.process_meta(meta_path)
        imgs_id =  glob.glob("{}/*.png".format(img_folder))
        return self.preprocess(imgs_id)

    def _genres(self):
        '''
        genres included
        '''

        return self.genres.columns

    def process_meta(self,path):
        '''
        Process meta data and vectorize genres
        Parameters:
        path: path for meta_data
        '''

        df = pd.read_csv(path)
        df.dropna(subset=['Genre'],inplace=True)
        df.drop_duplicates(subset=['imdbID'], keep='first', inplace=True)
        df.set_index('imdbID',inplace=True)

        target = df['Genre'].apply(lambda x: [w.strip() for w in x.split(',')])
        genres = pd.get_dummies(target.apply(pd.Series).stack()).sum(level=0)
        if not self.cols:
        # cutoff
            cols = genres.columns[genres.sum() < self.threshold]
            genres.drop(cols,axis=1,inplace =True)
        else:
            genres = genres[self.cols]
	    if not self.multi_label:
            	genres = genres[genres.sum(1) == 1]
        return genres[genres.sum(1) > 0]

    def preprocess(self,img_ids):
        '''
        Convert images to 2D numpy arrays and match them with labels
        Input: img_ids
        Output: image 2D arrays and labels
        '''
        
        p =re.compile(r'tt\w+')
        Xs,ys = [],[]
        for img in img_ids:
            id_ = p.search(img).group()
            try:
                 x,y = np.array(Image.open(img).resize(self.size,Image.NEAREST)),np.array(self.genres.loc[id_])
                 Xs.append(x)
                 ys.append(y)
            except:
                continue
        return np.array(Xs),np.array(ys)

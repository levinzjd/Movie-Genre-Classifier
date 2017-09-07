from PIL import Image
import glob
import pandas as pd
import numpy as np
import re

# generate numpy array for images and labels
class Load_data(object):
    def __init__(self,size):
        self.size = size

    def load(self,meta_path,img_folder,threshold=600,cols=None,ml=False):
        self.genres = self.process_meta(meta_path,threshold,cols,ml)
        imgs_id =  glob.glob("{}/*.png".format(img_folder))
        return self.preprocess(imgs_id,self.size)

    def _genres(self):
        return self.genres.columns

    def process_meta(self,path,threshold=1000,cols=None,multi_label=False):
        df = pd.read_csv(path)
        df.dropna(subset=['Genre'],inplace=True)
        df.drop_duplicates(subset=['imdbID'], keep='first', inplace=True)
        df.set_index('imdbID',inplace=True)

        target = df['Genre'].apply(lambda x: [w.strip() for w in x.split(',')])
        genres = pd.get_dummies(target.apply(pd.Series).stack()).sum(level=0)
        if not cols:
        # cutoff
            cols = genres.columns[genres.sum() < threshold]
            genres.drop(cols,axis=1,inplace =True)
        else:
            genres = genres[cols]
	    if not multi_label:
            	genres = genres[genres.sum(1) == 1]
        return genres[genres.sum(1) > 0]

    def preprocess(self,paths,size):
        p =re.compile(r'tt\w+')
        Xs,ys = [],[]
        for img in paths:
            id_ = p.search(img).group()
            try:
                 x,y = np.array(Image.open(img).resize(size,Image.NEAREST)),np.array(self.genres.loc[id_])
                 Xs.append(x)
                 ys.append(y)
            except:
                continue
        return np.array(Xs),np.array(ys)

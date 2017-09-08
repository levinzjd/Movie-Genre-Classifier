import urllib
import requests
import pandas as pd
import sys


# scrape movie posters and corresponding meta data
def scrape_meta_posters(mids):
    '''
    Scrape posters and metadata from IMDB using OMDB API
    Input: IMDB ids
    Output: meta data and posters
    '''
    
    meta_info = []
    for id_ in mids:
        url_meta = "http://www.omdbapi.com/?apikey={}&i={}".format(api_key,id_)
        response = requests.get(url_meta,stream=True)
        if response.status_code != 200:
            continue
        content = response.json()
        if 'Poster' not in content or content['Poster'] == 'N/A':
            continue
        else:
            info = response.json()
            url_poster = "http://img.omdbapi.com/?apikey={}&i={}".format(api_key,id_)
            urllib.urlretrieve(url_poster, "imgs/{}.png".format(id_))
            meta_info.append(info)
    return meta_info

if __name__ == '__main__':
	n1,n2 = int(sys.argv[1]),int(sys.argv[2])
	ids = pd.read_csv('movie_ids.csv')['ids'].values[n1:n2]

	df = pd.DataFrame(scrape_meta_posters(ids))
	df.to_csv('metas/{}_to_{}.csv'.format(n1,n2),encoding='utf-8')

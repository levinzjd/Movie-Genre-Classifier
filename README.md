# Movie-Genre-Classifier
Convolutional Neural Net to classify movie genres from posters. The posters are scarped from IMDB movie database using OMDB API. More than 37,000 posters of 28 genres and corresponding meta data are scraped. Due to the imbalance of different genres and high similarity/correlation of some genres (e.g. action & adventure, comedy & romance, horror & thriller), 4 out of 9 major genres (genres with at least 3,000 movies) are selected for classification.

## Data
The posters are scarped from IMDB movie database using OMDB API. More than 37,000 posters of 28 genres and corresponding meta data are scraped (each movie could have 1-3 genres).

## Scope
Due to the imbalance of different genres (e.g. half of the movies belong to drama) and high similarity/correlation of some genres (e.g. action & adventure, comedy & romance, horror & thriller), 4 out of 9 major genres (genres with at least 3,000 movies) are selected for classification.
 - Adventure
 - Documentary
 - Horror
 - Romance

## Model
The CNN model is implemented in Keras, customized image processing and other helper functions are shared in this repo

## Web app
The final product of this classifier is a web app implemented with Flask

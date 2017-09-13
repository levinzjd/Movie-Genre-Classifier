# Movie-Genre-Classifier
Convolutional Neural Net to classify movie genres from posters.

Convolutional Neural Net is widely used in image recognition problem. It is actually very good at capturing shapes and outlines of specific objects regardless of the relative locations in images. The question is how well could it capture the abstract information from the images, like genres from movie posters?

## Data
The posters are scarped from IMDB movie database using OMDB API. More than 37,000 posters of 28 genres and corresponding meta data are scraped (each movie could have 1-3 genres).

## Scope
Due to the imbalance of different genres (e.g. half of the movies belong to drama) and high similarity/correlation of some genres (e.g. action & adventure, comedy & romance, horror & thriller), 4 out of 9 major genres (genres with at least 3,000 movies) are selected for classification.
 - Adventure
 - Documentary
 - Horror
 - Romance

## Model
The final CNN model is implemented in Keras, with 5 Convolutional layers followed by maxpooling layers, 2 Dense layers followed by one dropout layer. The final model is implemented and trained in AWS EC2 instance with GPU configuration. Model, customized image processing and other helper functions are shared in this repo.

## Web app
The final product of this classifier is a web app implemented with Flask.

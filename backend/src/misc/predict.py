import os
import sys
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from keras.models import load_model
from pydub import AudioSegment

PATH_WEIGHTS_CLASSIFIER = os.path.join(os.getcwd(), "static/classifier.h5")
PATH_WEIGHTS_REGRESSOR = os.path.join(os.getcwd(), "static/regressor.h5")
PATH_CROSS_GENRES = os.path.join(os.getcwd(), "static/cross_genres.csv" )
PATH_FILES = os.path.join(os.getcwd(), "upload")

# remove debug output from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def generateJson(labels, genres):
    formatedValues = []
    for array in labels:
        for values in array:
            formatedValues.append(format(float(values), '.2f'))

    spotifyJson = {
        "target_acousticness": formatedValues[0],
        "target_danceability": formatedValues[1],
        "target_energy": formatedValues[2],
        "target_instrumentalness": formatedValues[3],
        "target_liveness": formatedValues[4],
        "target_speechiness": formatedValues[5],
        "target_valence": formatedValues[6],
        "seed_genres": genres
    }

    return json.dumps(spotifyJson)

def gen_features(wav_path):
    y, sr = librosa.load(wav_path)
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    return C

def mp3_to_wav(filename):
    sound = AudioSegment.from_mp3(PATH_FILES + '/' + filename)
    sound.export(PATH_FILES + '/wavs/' + filename + '.wav', format="wav") 
    return PATH_FILES + '/wavs/' + filename + '.wav'

def predict_mp3_regressor(filename):
    regressor = load_model(PATH_WEIGHTS_REGRESSOR)

    x_raw = []
    x_raw.append(gen_features(mp3_to_wav(filename))[:, :1291])

    x = np.array(x_raw)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

    return regressor.predict(x)

def predict_mp3_classifier(filename):
    classifier = load_model(PATH_WEIGHTS_CLASSIFIER)
    cross_genres = pd.read_csv(PATH_CROSS_GENRES)
    predictions = []

    x_raw = []
    x_raw.append(gen_features(mp3_to_wav(filename))[:, :1291])

    x = np.array(x_raw)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

    labels = classifier.predict(x)
    top_labels = np.argsort(-labels)[0,:5]
    predictions.append(cross_genres.loc[top_labels]['title'].to_numpy())

    listStrings = str(cross_genres.loc[top_labels]['title'].to_numpy()).split(' ')

    formatedList = []
    # remove non alphanumeric chars and add to list
    for s in listStrings:
        formatedList.append(''.join(e for e in s if e.isalnum()))

    # add elements to one string, separated with ','
    output = ""
    for ele in formatedList[:2]:
        output += ele + ","

    output = output[:-1]

    return output


def main(filename):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    predictedRegressor = predict_mp3_regressor(filename)
    predictedClassifier = predict_mp3_classifier(filename)

    print(generateJson(predictedRegressor, predictedClassifier))
    sys.stdout.flush()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename", help="file name of mp3")
    args = parser.parse_args()

    main(args.filename)


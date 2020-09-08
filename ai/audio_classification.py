'''
This class represents a library to analyze and genreata training data
in form of features and labels. Using these you can build, train and evaluate two types of AI.
The audio feature regressor and the genre classifier.
'''

import math
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import SGD
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# paths for loading/storing models and data
DATA_PATH = '/Users/linusstenzel/Desktop/PS_MUSICAI/data'
EXT_DATA_PATH = '/Volumes/Linus/data'
AI_PATH = '/Users/linusstenzel/Desktop/PS_MUSICAI/ai'

# time (in librosa ticks) that the tracks need to fit the model
TRACK_TIME = 1291

# amount of training data that should be loaded, None -> no restriction
FEAT_AMOUNT = 10
CLS_LAB_AMOUNT = 10
REG_LAB_AMOUNT = 10

# ai names
CLASSIFIER = 'classifier'
REGRESSOR = 'regressor'


if Path(DATA_PATH + '/features.csv').is_file():
    # load training data (features)
    features_pd = pd.read_csv(DATA_PATH + '/features.csv', index_col=0) if FEAT_AMOUNT is None else pd.read_csv(
        DATA_PATH + '/features.csv', nrows=FEAT_AMOUNT, index_col=0)
    features_pd.columns = features_pd.columns.astype(int)

if Path(DATA_PATH + '/classifier_labels_1000.csv').is_file():
    # load training data (classifier labels)
    classifier_labels_pd = pd.read_csv(DATA_PATH + '/classifier_labels_1000.csv', index_col=0) if CLS_LAB_AMOUNT is None else pd.read_csv(
        DATA_PATH + '/classifier_labels_1000.csv', nrows=CLS_LAB_AMOUNT, index_col=0)
    classifier_labels_pd.columns = classifier_labels_pd.columns.astype(int)

if Path(DATA_PATH + '/regressor_labels.csv').is_file():
    # load training data (regressor labels)
    regressor_labels_pd = pd.read_csv(DATA_PATH + '/regressor_labels.csv', index_col=0) if REG_LAB_AMOUNT is None else pd.read_csv(
        DATA_PATH + '/regressor_labels.csv', nrows=REG_LAB_AMOUNT, index_col=0)
    regressor_labels_pd.columns = regressor_labels_pd.columns.astype(int)

# load free music archive echonest. used for regressor labels
echonest = np.genfromtxt(DATA_PATH + '/fma_metadata/echonest.csv',
                         delimiter=',', usecols=list(range(7)) + list(range(8, 9)), skip_header=4)
# regressor label names
audio_features = ['acousticness', 'danceability', 'energy',
                  'instrumentalness', 'liveness', 'speechiness', 'valence']

# load free music archive tracks. used for classifier labels
genres = pd.read_csv(DATA_PATH + '/fma_metadata/tracks.csv',
                     header=0, index_col=0, usecols=[0, 40], dtype=str)
genres = genres.iloc[1:]

# load intersection between spotify and free music archive tracks genres. classifier label names
cross_genres = pd.read_csv(DATA_PATH + '/cross_genres.csv', index_col=0)


####################################################
#-----------------------DATA-----------------------#
####################################################


def save_train_data(model_type, do_features=True, count=100, start=0):
    ''' Runs over all .wav files in DATA_APTH and converts them to features and labels
        needed for training the ai. Returns features/labels in a dictionary
        but also saves the them to DATA_PATH. Return values in dictionary are
        Pandas DataFrames where each row contains the file(name) and its features/labels.

        Keyword arguments:
        model_type -- model name. 'regressor' or 'classifier'
        do_features -- True -> compute features (computationally intensive)
        count -- amount of wav files to consider
        start -- position in wav files to start working on
    '''
    # keeping track of both data and names
    x_raw, y_raw = [], []
    x_names, y_names = [], []

    num = 0
    print('convert .wav files to features and labels')
    for filename in os.listdir(EXT_DATA_PATH + '/wavs/'):
        if start <= 0:
            if filename.endswith('.wav') and num < count:

                labels_cls = get_labels(int(filename[:-4]), CLASSIFIER)
                labels_reg = get_labels(int(filename[:-4]), REGRESSOR)

                # only compute features when any labels are existing to current .wav file
                if do_features and not(labels_cls is None and labels_reg is None):
                    features = gen_features(
                        EXT_DATA_PATH + '/wavs/' + filename)
                    # tracks that are too short are not considered
                    if features.shape[1] >= TRACK_TIME:
                        x_raw.append(features)
                        x_names.append(filename[:-4])

                if model_type == CLASSIFIER and not(labels_cls is None):
                    y_raw.append(labels_cls)
                    y_names.append(filename[:-4])
                elif model_type == REGRESSOR and not(labels_reg is None):
                    y_raw.append(labels_reg)
                    y_names.append(filename[:-4])

                # keep track of progress
                num += 1
                if num > 1:
                    delete_last_lines()
                print(str(round(num/count*100, 2)) + ' percent done')
        else:
            start -= 1

    if do_features:
        # reshape/flatten feature array from 2 dimensions to 1
        x = np.zeros((len(x_raw), len(x_raw[0]), TRACK_TIME))
        i = 0
        for feature in x_raw:
            j = 0
            for semitone in feature:
                x[i][j] = semitone[:TRACK_TIME]
                j += 1
            i += 1
        x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))  # =15492

        # save to filesystem
        x = pd.DataFrame(x, index=x_names)
        x.to_csv(DATA_PATH + '/features.csv')
    else:
        x = None

    # save to filesystem
    y = pd.DataFrame(y_raw, index=y_names)
    y.to_csv(DATA_PATH + '/' + model_type + '_labels.csv')

    return {'features': x, 'labels': y}


def gen_features(wav_path):
    ''' Converts .wav file to chromagram (using the librosa library) and returns it.
        Chroma feature or chromagram relates to the twelve different pitch classes.
        It is a powerful tool for analyzing music whose pitches can be meaningfully categorized.
        One main property of chroma features is that they capture harmonic and melodic characteristics of music,
        while being robust to changes in timbre and instrumentation.

        Keyword arguments:
        wav_path -- .wav file path to compute
    '''
    y, sr = librosa.load(wav_path)
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    return C


def get_labels(track_id, model_type):
    ''' Returns labels as a numpy array corresponding to track_id.

        Keyword arguments:
        track_id -- track identifier
        model_type -- model name. 'regressor' or 'classifier'
    '''
    if model_type == CLASSIFIER:

        # get raw genre ids from free music archive genres
        top_genre_name = genres.loc[track_id, 'favorites.2']
        if isinstance(top_genre_name, str):
            top_genre_name = top_genre_name.lower()
        else:
            top_genre_name = ''

        titles = cross_genres['title']
        if top_genre_name in cross_genres['title'].array:
            # convert free music archive genre ids to cross genre ids and label
            genre_id = cross_genres.loc[cross_genres['title']
                                        == top_genre_name].index
            labels = np.zeros(7)
            labels[genre_id] = 1
            return labels

    elif model_type == REGRESSOR:
        for row in echonest:
            if row[0] == track_id:
                return row[1:]


def mp3_to_wav(mp3_path, name):
    ''' Converts .mp3 .wav file saves it to filesystem and returns filepath.

        Keyword arguments:
        mp3_path -- path of .mp3 file
        name -- .wav name
    '''
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(DATA_PATH + '/wavs/' + name + '.wav', format="wav")
    return DATA_PATH + '/wavs/' + name + '.wav'


def all_to_wav(count=100, start=0):
    ''' Converts mulitiple .mp3 files to .wav files and saves them to filesystem.

        Keyword arguments:
        count -- amount of .mp3 files to consider
        start -- position in .mp3 files to start working on
    '''
    i = 0
    print('convert .mp3 files to .wav')
    for subdir, dirs, files in os.walk(DATA_PATH + '/fma_large'):
        for file in files:
            if start <= 0:
                if file.endswith('.mp3') and i < count:
                    try:
                        mp3_to_wav(os.path.join(subdir, file), file[:-4])
                        i += 1
                    except:
                        # some files in fma_large may be damaged
                        pass

                    # keep track of progress
                    if i > 1:
                        delete_last_lines()
                    print(str(round(i/count*100, 2)) + ' percent done')

            else:
                start -= 1


def genre_occurence():
    ''' Returns numpy array containing the amounts of tracks
        each genre holds in classifier labels.
    '''
    genre_occ = np.zeros(7, dtype=int)
    for index, row in classifier_labels_pd.iterrows():
        genre_occ[np.argwhere(row.values == 1).flatten()] += 1
    return genre_occ


def genre_to_kill(min_amount=250):
    ''' Returns array containing the genres which have
        less than min_amount occurences in classifier labels.

        Keyword arguments:
        min_amount -- min occurence amount to not be killed
    '''
    to_kill = []
    for key, value in dict(zip(cross_genres['title'], genre_occurence())).items():
        if value < min_amount:
            to_kill.append(key)
    return to_kill


def pick_cls_labels(min_amount=500):
    ''' Removes tracks from classifier labels which genres
        have more occurences than min_amount.

        Keyword arguments:
        min_amount -- min occurence amount to be removed
    '''
    labels_amount = np.zeros(7)

    classifier_labels_pd = pd.read_csv(
        DATA_PATH + '/classifier_labels.csv', index_col=0)

    for index, row in classifier_labels_pd.iterrows():
        if (labels_amount < min_amount).any():
            row_amounts = labels_amount[np.argwhere(
                row.to_numpy() == 1).flatten()]

            if (row_amounts >= min_amount).all():
                classifier_labels_pd = classifier_labels_pd.drop(index)

            np.put(labels_amount, np.argwhere(row.to_numpy() == 1).flatten(),
                   labels_amount[np.argwhere(row.to_numpy() == 1).flatten()] + 1)

    classifier_labels_pd.to_csv(
        DATA_PATH + '/classifier_labels_' + str(min_amount) + '.csv')


####################################################
#---------------------REGRESSOR--------------------#
####################################################


def build_regressor():
    ''' Builds regressor keras model and returns it.
        Layer dimensions: 15492 -> 512 -> 256 -> 256 -> 128 -> 7
    '''
    regressor_input = Input(shape=(15492,))
    x = Dense(512, activation='relu')(regressor_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    regressor_output = Dense(7, activation='sigmoid')(
        x)  # seven tunable audio features
    return Model(inputs=regressor_input, outputs=regressor_output)


def train_regressor(generate_data):
    ''' Trains regressor keras model with loaded training data.
        Saves model to file system and returns it.

        Keyword arguments:
        generate_data -- True -> generate data, False -> load data
    '''
    if generate_data:
        train_data = save_train_data(REGRESSOR)
        x = train_data['features']
        y = train_data['labels']
    else:
        y = regressor_labels_pd
        x = features_pd.loc[y.index.values].dropna()
        y = y.loc[x.index.values].dropna()

    x = x.sort_index()
    y = y.sort_index()

    # training data must contain same indexes
    assert x.index.equals(y.index)

    # split data in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(), y.to_numpy(), test_size=0.02, random_state=42)

    regressor = build_regressor()
    # compile and train model
    regressor.compile(optimizer=SGD(lr=.001),
                      loss='mean_absolute_error',
                      metrics=['mse'])
    regressor.fit(x_train, y_train, epochs=250, batch_size=128)
    regressor.save(AI_PATH + "/Models/regressor.h5")
    return regressor


def evaluate_regressor(train, generate_data=False):
    ''' Evaluates regressor keras model with loaded training(test) data.
        Also makes predictions on test data and saves them to filesystem.

        Keyword arguments:
        train -- True -> train model, False -> load model
        generate_data -- True -> generate data, False -> load data
    '''
    if train:
        regressor = train_regressor(generate_data)
    else:
        regressor = load_model(AI_PATH + '/Models/regressor.h5')

    if generate_data:
        train_data = save_train_data(REGRESSOR)
        x = train_data['features']
        y = train_data['labels']
    else:
        y = regressor_labels_pd
        x = features_pd.loc[y.index.values].dropna()
        y = y.loc[x.index.values].dropna()

    x = x.sort_index()
    y = y.sort_index()

    # training data must contain same indexes
    assert x.index.equals(y.index)

    # split data in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(), y.to_numpy(), test_size=0.02, random_state=42)

    results = regressor.evaluate(x_test, y_test, batch_size=2)
    print('eva ' + str(results))

    predictions = regressor.predict(x_test)
    truth = y_test
    save_pred_truth(predictions, truth, 7, REGRESSOR)

####################################################
#--------------------CLASSIFIER--------------------#
####################################################


def build_classifier():
    ''' Builds classifier keras model and returns it.
        Layer dimensions: 15492 -> 512 -> 256 -> 128 -> 29
    '''
    classifier_input = Input(shape=(15492,))
    x = Dense(512, activation='relu')(classifier_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    classifier_output = Dense(7, activation='softmax')(x)
    # cross_genres holds 7 unique values
    return Model(inputs=classifier_input, outputs=classifier_output)


def train_classifier(generate_data):
    ''' Trains classifier keras model with loaded training data.
        Saves model to file system and returns it.

        Keyword arguments:
        generate_data -- True -> generate data, False -> load data
    '''
    if generate_data:
        train_data = save_train_data(CLASSIFIER)
        x = train_data['features']
        y = train_data['labels']
    else:
        y = classifier_labels_pd
        x = features_pd.reindex(y.index.values).dropna()
        y = y.reindex(x.index.values).dropna()

    x = x.sort_index()
    y = y.sort_index()

    # training data must contain same indexes
    assert x.index.equals(y.index)

    # split data in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(), y.to_numpy(), test_size=0.01, random_state=42)

    # compute classweights because of dataset imbalance
    y_integer = np.empty(0, dtype=int)
    for index, row in y.iterrows():  # TODO iter y not cls_labels
        y_integer = np.append(
            y_integer, np.argwhere(row.values == 1).flatten())

    keys = np.sort(np.unique(y_integer))
    values = compute_class_weight('balanced', np.unique(y_integer), y_integer)
    class_weight = dict(zip(keys, values))

    classifier = build_classifier()
    # compile and train model
    classifier.compile(optimizer=SGD(lr=.0005),
                       loss='categorical_crossentropy',
                       metrics=['acc'])
    classifier.fit(x_train, y_train, epochs=400,
                   batch_size=64, class_weight=class_weight)
    classifier.save(AI_PATH + '/Models/classifier.h5')
    return classifier


def evaluate_classifier(train, generate_data=False):
    ''' Evaluates classifier keras model with loaded training(test) data.
        Also makes predictions on test data and saves them to filesystem.

        Keyword arguments:
        train -- True -> train model, False -> load model
        generate_data -- True -> generate data, False -> load data
    '''

    if train:
        classifier = train_classifier(generate_data)
    else:
        classifier = load_model(AI_PATH + '/Models/classifier.h5')

    if generate_data:
        train_data = save_train_data(CLASSIFIER)
        x = train_data['features']
        y = train_data['labels']
    else:
        y = classifier_labels_pd
        x = features_pd.reindex(y.index.values).dropna()
        y = y.reindex(x.index.values).dropna()

    x = x.sort_index()
    y = y.sort_index()

    # training data must contain same indexes
    assert x.index.equals(y.index)

    # split data in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(), y.to_numpy(), test_size=0.01, random_state=42)

    results = classifier.evaluate(x_test, y_test, batch_size=2)
    print('eva ' + str(results))

    predictions = classifier.predict(x_test)
    truth = y_test
    save_pred_truth(predictions, truth, 7, CLASSIFIER)


def save_pred_truth(predictions, truth, dim, model_type):
    ''' Saves predictions to filesystem.
        First row containing true values, second row contains predicted values.

        Keyword arguments:
        predictions -- Numpy array containing model predictions
        truth -- Numpy array containing true values
        dim -- (output) dimension of data
        model_type -- model name. 'regressor' or 'classifier'
    '''
    # mixing predictions and true values one by one
    length = int(3 * truth.size / dim)
    pred_truth = pd.DataFrame(
        columns=audio_features if model_type == REGRESSOR else cross_genres['title'])
    x, y = 0, 0
    for i in range(0, length):
        if i % 3 == 0:
            pred_truth.loc[i] = truth[x]
            x += 1
        elif i % 3 == 1:
            pred_truth.loc[i] = predictions[y]
            y += 1
        else:
            pred_truth.append(pd.Series(), ignore_index=True)

    pred_truth.to_csv(DATA_PATH + '/pred_' + model_type +
                      '.csv', float_format='%1.2f')


def predict_mp3_regressor():
    ''' Uses regressor model to predict seven audio features
        used for getting music recommendation from Spotify.
        Runs over .mp3 files and returns predictions array.
        Also prints them to stdout.
    '''
    regressor = load_model(AI_PATH + "/Models/regressor.h5")
    predictions = []

    for filename in os.listdir(DATA_PATH + '/to_predict/'):
        x_raw = []
        if filename.endswith('.mp3'):
            # generate feature from .mp3 and short to model input length
            x_raw.append(gen_features(mp3_to_wav(
                DATA_PATH + '/to_predict/' + filename, 'cust/' + filename[:-4]))[:, :1291])

            x = np.array(x_raw)
            x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

            labels = regressor.predict(x)
            predictions.append(labels)

            print(filename[:-4] + ' labels ' + str(labels))

    return predictions


def predict_mp3_classifier():
    ''' Uses regressor model to predict 44 genres probabilities
        used for getting music recommendation from Spotify.
        Runs over .mp3 files and returns top five genre names from predictions.
        Also prints them to stdout.
    '''
    classifier = load_model(AI_PATH + "/Models/classifier.h5")
    predictions = []

    for filename in os.listdir(DATA_PATH + '/to_predict/'):
        x_raw = []
        if filename.endswith('.mp3'):
            # generate feature from .mp3 and short to model input length
            x_raw.append(gen_features(mp3_to_wav(
                DATA_PATH + '/to_predict/' + filename, 'cust/' + filename[:-4]))[:, :1291])

            x = np.array(x_raw)
            x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

            labels = classifier.predict(x)
            # convert predictions to top five genres
            top_labels = np.argsort(-labels)[0, :3]
            predictions.append(
                cross_genres.loc[top_labels]['title'].to_numpy())

            print(filename[:-4] + ' labels ' +
                  str(cross_genres.loc[top_labels]['title'].to_numpy()))

    return predictions


def main():
    ''' Main method. Call methods of your choice and run script.
    '''
    sys.path.append('/path/to/ffmpeg')
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    features_pd.to_csv(DATA_PATH + '/features_10.csv')
    classifier_labels_pd.to_csv(DATA_PATH + '/classifier_labels_10.csv')
    regressor_labels_pd.to_csv(DATA_PATH + '/regressor_labels_10.csv')


    #evaluate_regressor(True)


# help function for showing progress
def delete_last_lines(n=1):
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


if __name__ == "__main__":
    main()

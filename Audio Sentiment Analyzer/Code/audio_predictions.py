import numpy as np                         # Numerical python     
import librosa                             # Audio Analysis Package
import wave                                # Audio Processing                        
from pydub import AudioSegment             # Import the AudioSegment class for processing audio and the 
from pydub.silence import split_on_silence # split_on_silence function for separating out silent chunks.
import os
from scipy.stats import mode
from keras.models import load_model
import pandas as pd

model_cnn_imp = load_model('Final_Model_new.h5')

def split(filepath):
	"""Split Audio on Silence"""
    sound = AudioSegment.from_wav(filepath)
    dBFS = sound.dBFS
    chunks = split_on_silence(sound, 
        min_silence_len = 500,
        silence_thresh = dBFS-16,
        keep_silence = 250)
    return chunks

def get_mfcc(x_wav, num_features = 100):
	"""Extract mfcc features"""
    y, sr = librosa.load(x_wav)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_features).T,axis=0)
    return mfccs

def decode(datum):
	"""Get CNN predictions"""
    res = np.argmax(datum,axis = 1)
    p1 = int(mode(res[0::2])[0][0])
    p2 = int(mode(res[1::2])[0][0])
    p1_pred = Emotions[p1]
    p2_pred = Emotions[p2]
    return p1_pred, p2_pred


def main_func(path):
	"""Main prediction Function"""

	Emotions = ['Anger','Fear','Surprised','Sad','Happy','Disgust','Neutral']

	ch = split(path)
	print("Processing Data")
	for i, chunk in enumerate(ch):
	    chunk.export(
	        "_Temp_files_2/chunk{0}.wav".format(i),
	        bitrate = "192k",
	        format = "wav")

	Data_for_predictions = []
	print("Fetching Predictions")
	for filename in os.listdir('_Temp_files_2'):
	    Data_for_predictions.append(get_mfcc('_Temp_files_2/' + filename))
	Data_for_predictions = np.array(Data_for_predictions)

	for filename in os.listdir('_Temp_files_2'):
	    os.remove('_Temp_files_2/' + filename)


	preds = model_cnn_imp.predict(np.expand_dims(Data_for_predictions,-1))

	pred1, pred2 = decode(preds)

	return pred1, pred2



def main(path1, language_id):
	"""Fetches Predictions and saves to csv"""

	input_output = []

	for file in os.listdir(path1):

	    print('Processing Audio file:',file)
	    pred1, pred2 = main_func(path1 + '/' + file)

	    print('person01: ' + str(pred1) + '\nperson02: ' + str(pred2))
	    print()

	    pred_line = 'person01: ' + str(pred1) + 'person02: ' + str(pred2) 
	    input_output.append((file,pred_line))
	    
	Data = pd.DataFrame(input_output)
	Data.columns = [['Input', 'Output']]

	if(language_id==1):
		Data.to_csv('DeepHackers_I-42SNF_ENGLISH.csv')
		print("CSV file Created")

	elif(language_id==2):
		Data.to_csv('DeepHackers_I-42SNF_HINDI.csv')
		print("CSV file Created")

	elif(language_id==3):
		Data.to_csv('DeepHackers_I-42SNF_TELUGU.csv')
		print("CSV file Created")

	else:
		print("CSV file NOT Created::NO PARAMETER")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy
import re

categories = ["Negative", "Positive"]
from sklearn.preprocessing import LabelEncoder
text=["A lot of good things are happening. We are respected again throughout the world, and that's a great thing"]


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
X = pad_sequences(X, maxlen=28)

model = load_model("model.h5")

prediction = model.predict(X)
pred_name = categories[np.argmax(prediction)]
print(pred_name)
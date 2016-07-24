'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

#path = get_file('out.txt', 'C:\Users\Dule\Desktop\textic')
text = open('bots85.txt', encoding="utf8").read().lower()
text = text[0:100000]

print('corpus length:', len(text))
with open("Output.txt", "a") as text_file:
    text_file.write('corpus length: ' + str(len(text)) + "\n")

chars = sorted(list(set(text)))
print('total chars:', len(chars))
with open("Output.txt", "a") as text_file:
    text_file.write("total chars: " + str(len(chars)) + "\n")
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
	
sentencesorder = np.random.permutation(len(sentences))
sentences = [ sentences[i] for i in sentencesorder]
next_chars = [ next_chars[i] for i in sentencesorder ]

#print(sentences[1:10])
	
print('nb sequences:', len(sentences))
with open("Output.txt", "a") as text_file:
    text_file.write("nb sequences: " + str(len(sentences)) + "\n")
	
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

briter = 40
history = []
	
# train the model, output generated text after each iteration
for iteration in range(1, briter + 1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    with open("Output.txt", "a") as text_file:
        text_file.write("\n\n" + "Iteration: " + str(iteration) + "\n")
		
    h = model.fit(X, y, batch_size=128, nb_epoch=1)
	
    model_architecture = model.to_json()
    with open('model_architecture.json', 'w') as output:
        output.write(model_architecture)
    model.save_weights('model_weights.h5', overwrite = True)
	
    history.append(h.history['loss'][0])
    start_index = random.randint(0, len(text) - maxlen - 1)
	
    with open("loss.txt", "a") as loss_txt:
        loss_txt.write(str(h.history['loss'][0]) + ",")
	
	

    for diversity in [0.2]:#, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        with open("Output.txt", "a") as text_file:
            text_file.write('diversity: ' + str(diversity))
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('\nSeed: "' + sentence + '"')
        sys.stdout.write(generated)
        with open("Output.txt", "a") as text_file:
            text_file.write('\nSeed: ' + generated + "\n\n")

        # for i in range(400):
            # x = np.zeros((1, maxlen, len(chars)))
            # for t, char in enumerate(sentence):
                # x[0, t, char_indices[char]] = 1.

            # preds = model.predict(x, verbose=0)[0]
            # next_index = sample(preds, diversity)
            # next_char = indices_char[next_index]

            # generated += next_char
            # sentence = sentence[1:] + next_char

            # #sys.stdout.write(next_char)
            # with open("Output.txt", "a") as text_file:
                # text_file.write(next_char)
            # sys.stdout.flush()
        # print()

        
print()

    
print(history)
# with open("loss.txt", "a") as text_file:
    # text_file.write(str(history))
# sys.stdout.flush()


# model_architecture = model.to_json()
# with open('model_architecture.json', 'w') as output:
    # output.write(model_architecture)
# model.save_weights('model_weights.h5', overwrite = True)
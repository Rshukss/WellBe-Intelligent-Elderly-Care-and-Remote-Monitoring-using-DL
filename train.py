# libraries
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
data_file = open("intents.json").read()
intents = json.loads(data_file)

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Process intents and patterns
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize words
        w = word_tokenize(pattern)
        words.extend(w)
        # Adding documents
        documents.append((w, intent["tag"]))
        # Adding classes to the class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes as pickle files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Prepare training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])
print("Training data created")

# Create an advanced model
model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# Compile the model using the Adam optimizer
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# Train and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("Model created and trained")

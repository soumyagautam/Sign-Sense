import pickle       # For dealing with stored dataset
from sklearn.ensemble import RandomForestClassifier         # Base AI Model
from sklearn.model_selection import train_test_split        # To get the x_train, y_train, x_test, y_test
from sklearn.metrics import accuracy_score, classification_report       # For calculating the model's prediction score
# from imblearn.over_sampling import RandomOverSampler         # Not used
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences      # For padding the dataset to remove whitespaces
import numpy as np          # For doing mathematical calculations

data_dict = pickle.load(open("data.pickle", "rb"))        # Loading the data

# Check the shape of items in data key of data_dict
# for idx, element in enumerate(data_dict["data"]):
#     print(f"Element {idx + 1} shape: {np.asarray(element).shape}")

# Padding the data to ensure that their length is same
max_length = max(len(seq) for seq in data_dict["data"])
data_padded = pad_sequences(data_dict["data"], maxlen=max_length, padding='post', truncating='post', dtype="float32")

# Creating numpy arrays out of data
data = np.asarray(data_padded)
labels = np.asarray(data_dict["labels"])

# over_sampler = RandomOverSampler()
# data_resampled, labels_resampled = over_sampler.fit_resample(data, labels)

# Getting the x_train, y_train, x_test, y_test
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels
)

model = RandomForestClassifier()           # Starting and training Random Forest Classifier
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)           # Getting the accuracy score of model
print(f"{score * 100}% of the samples were classified correctly!")  # Generating the classification report for the model
print("Classification Report:")
print(classification_report(y_test, y_predict))

file = open("model2.p", "wb")       # Saving the model
pickle.dump({"model": model}, file)
file.close()

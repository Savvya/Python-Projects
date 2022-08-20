import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
total_DS = image_dataset_from_directory(
    '../input/siim-isic-2019-organized/dataset organized',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
    seed=1
)


# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label
#Splitting dataset into train, test and validation
DATASET_SIZE = 518

train_size = int(0.75 * DATASET_SIZE)
val_size = int(0.125 * DATASET_SIZE)
test_size = int(0.125 * DATASET_SIZE)

full_dataset = total_DS
training_Data = full_dataset.take(train_size)
test_Data = full_dataset.skip(train_size)
validation_Data = test_Data.skip(val_size)
test_Data = test_Data.take(test_size)
AUTOTUNE = tf.data.experimental.AUTOTUNE
trainingData = (
    training_Data
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

validationData = (
    validation_Data
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

testData = (
    test_Data
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# A function that reduces the learning rate every 20 epochs after 15th epoch
def scheduler(epoch, lr):
    print(lr)
    if epoch % 2 == 0 and epoch < 15:
        return lr * tf.math.exp(-0.9) 
    else: 
        return lr
# Creating callbacks
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_f1_m",patience=20), # Monitors the validation f1 score and decides wether or not to sotp every 20 epochs
    tf.keras.callbacks.LearningRateScheduler(scheduler) # Calls scheduler function
]
#Calculating f1 score using the recall and percision

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Creating the metrics array
METRICS = [
    tf.keras.metrics.BinaryAccuracy(),
    recall_m, 
    tf.keras.metrics.AUC(name = 'ROC', curve = 'roc'),
    precision_m,
    f1_m,
    tf.keras.metrics.TruePositives(name = 'TP'),
    tf.keras.metrics.FalsePositives(name = 'FP'),
    tf.keras.metrics.TrueNegatives(name = 'TN'),
    tf.keras.metrics.FalseNegatives(name = 'FN')
]
# importing architectures
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD 
# Load in VGG16
base_vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
# Add a new fully connected layer at the end for our input data
vgg16_output = base_vgg16.output
vgg16_output = GlobalAveragePooling2D()(vgg16_output)
vgg16_output = Dense(1024, activation='relu')(vgg16_output)
# logistic layer for our 1 class, benign_malignant
vgg16_predictions = Dense(1, activation='sigmoid')(vgg16_output)

vgg16model = Model(inputs=base_vgg16.input, outputs=vgg16_predictions)

#Train only the top layers which are randomly initialized
#Freeze all the convolutional architecture layers
for layer in base_vgg16.layers: 
    layer.trainable = False

#compile model
vgg16model.compile(optimizer=SGD(learning_rate=0.1), loss="binary_crossentropy", metrics=METRICS)

#train the model on the new data for a few epochs
vgg16model.fit(
    trainingData, 
    validation_data = validationData, 
    epochs=3, 
    verbose=1, 
    batch_size=256, 
    class_weight = {0:1, 1:55.7226027397},
    callbacks = my_callbacks)

#Now the top layers are well trained so we can begin fine tuning

for layer in vgg16model.layers[:12]:
    layer.trainable = False 
for layer in vgg16model.layers[12:]:
    layer.trainable = True

# We recompile to apply these new  changes
# Use SGD with a lower learning rate than the previous time
vgg16model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=METRICS)

#Train the model again to fine tune it further

vggTrainer = vgg16model.fit(
    trainingData, 
    validation_data = validationData, 
    epochs=500, 
    verbose=1, 
    batch_size=256, 
    class_weight={0:1, 1:55.7226027397}, 
    callbacks=my_callbacks)
# Testing the models to see which one performed the best

vgg16model.evaluate(testData)
########## RNN CODE STARTS HERE ##############
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
tf.random.set_seed(99)
import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep
folder = "../input/images-siim-512x512/train/train_512x512"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")

for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=64, height=64))
import csv
#(train_images,train_labels),(test_images,test_labels)=fas_mnist.load_data()
mapImgToLabel = {}
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
channels = 3
train_files = []
y_train =[]

with open(r"../input/siim-isic-melanoma-classification/train.csv", newline = '') as csvfile:
    #csvfile = open('../input/siim-isic-melanoma-classification/train.csv')
    csvreader = csv.reader(csvfile)


    for row in csvreader: 
        label = row[7]
        imageId = row[0]
        mapImgToLabel[imageId] = label
        #print(label)
        #print(imageId)
        #print("Do I come here")

    print(mapImgToLabel["ISIC_0015719"])
    print(mapImgToLabel["ISIC_0247330"])   
    print("Outside Of first loop")

    #use source directory;
    src_dir = "../input/images-siim-512x512/train/train_512x512"
    #read all files
    #loop through each file...
     #     train_files.append(file)
      #    
       #   imageId = get the id from each file
        #        (removing the .jpg)
    i = 0
    counter = 0 
    for _file in onlyfiles:
        if(counter <= 34000):
            train_files.append(_file)
            imageIdFromFilename = _file.replace(".jpg", "")
            #print(imageIdFromFilename)
            image_label = mapImgToLabel.get(imageIdFromFilename, "0")
            #Fix this default later, use exception
            y_train.append(int(image_label))
            #print(int(image_label)) 
            counter += 1
        
print("Files in train_files: %d" % len(train_files))
print("All files mapped to right label")
import PIL
from PIL import Image
import matplotlib.pyplot as plt
sample_img = Image.open("../input/images-siim-512x512/train/train_512x512/ISIC_0015719.jpg")
plt.imshow(sample_img)
#512x512 image colored
# Original Dimensions

#ratio = 4

#image_width = int(image_width / ratio)
#image_height = int(image_height / ratio)

channels = 1
nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), 64, 64),
                     dtype=np.float32)

i = 0
counter = 0

for _file in train_files:
    if(counter <= 34000):
        img = Image.open(folder + "/" + _file)
        img = img.reduce(8)
        img = img.convert("1")
        #img = load_img(folder + "/" + _file, color_mode="grayscale")  # this is a PIL image
        # Convert to Numpy Array
        x = img_to_array(img, data_format = "channels_last")  
        x = x.reshape((64, 64))
        # Normalize
        # x = (x - 128.0) / 128.0
        dataset[i] = x
        i += 1
        counter += 1
        if i % 1000 == 0:
            print("%d images to array" % i)
print("All images to array!")
x = np.array(dataset)
y = np.array(y_train)
from sklearn.model_selection import train_test_split
x_train, val_test, y_train, val_test_labels = train_test_split(x, y, test_size=0.3, train_size = 0.7, shuffle=False) 
x_val, x_test, y_val, y_test = train_test_split(val_test, val_test_labels, test_size = 0.5, train_size = 0.5, shuffle = True)
x_train = tf.keras.utils.normalize(x_train, axis = -1)
x_val = tf.keras.utils.normalize(x_val, axis = -1)
x_test = tf.keras.utils.normalize(x_test, axis = -1)
# The concept is simple, we take each HxW matrix of images --> Flatten it like sequence of Multi-dimensional time-series and feed to LSTM
# HxW changes to TxD 
# In images H--> height, W--> width, similiarly T-->Timestamp(equals H), D-->Feature(equals W)
bidirectional_model = tf.keras.Sequential([
  tf.keras.Input(shape=(64,64)),
  tf.keras.layers.SimpleRNN(128),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
# compile the model (should be done *after* setting layers to non-trainable)
from tensorflow.keras.optimizers import SGD
bidirectional_model.compile(optimizer = SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=METRICS)
def schedulerRnn(epoch, lr):
    print(lr)
    if epoch % 2 == 0:
        return lr * tf.math.exp(-0.9)
    else: 
        return lr

        
my_callbacks_rnn = [
    tf.keras.callbacks.EarlyStopping(monitor="val_f1_m",patience=30),
    tf.keras.callbacks.LearningRateScheduler(schedulerRnn)
]
bidirectional_trainer = bidirectional_model.fit(x_train, y_train, epochs=100, callbacks = my_callbacks_rnn, class_weight = {0:1,1:55.7226027397}, validation_data = (x_val, y_val))
bidirectional_model.evaluate(x_test, y_test, verbose=1)
#BASIC PACKAGES
import numpy as np
import seaborn as sns
import matplotlib.pyplot as ply
import glob
import cv2
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16
SIZE = 32 # image size

# image data and labels lists

melanoma_images = [] 
melanoma_labels = []
# append images and labels to respective lists
for directory_path in glob.glob("../input/siim-isic-2019-organized/dataset organized/*"):
    label = directory_path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #print(img_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE)) # RESIZE ALL IMAGES TO 256x256
        
        melanoma_images.append(img)
        melanoma_labels.append(label)
melanoma_images = np.array(melanoma_images)
melanoma_labels = np.array(melanoma_labels)
from sklearn import preprocessing
LE = preprocessing.LabelEncoder() 
LE.fit(melanoma_labels) # encode labels as 0 and 1, not benign and malignant
encoded_labels = LE.transform(melanoma_labels)
from sklearn.model_selection import train_test_split
x_train, val_test, y_train, y_val_test = train_test_split(melanoma_images, encoded_labels, train_size = 0.6) # Splitting into train and test sets
x_test, x_val, y_test, y_val = train_test_split(val_test, y_val_test, test_size = 0.5)
# Change pixel values to be between 0 and 1
x_train, x_test, x_val  = x_train / 255.0, x_test / 255.0, x_val / 255.0
# Load in VGG16 as feature-extractor
VGG16_model = VGG16(weights="imagenet", include_top=False, input_shape=(SIZE, SIZE, 3))
for layer in VGG16_model.layers:
    layer.trainable = False # make all layers not trainable
VGG16_model.summary() # 0 trainable paramaters
feature_extractor = VGG16_model.predict(x_train)
# Set the training features as the new training set for xgboost
train_features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = train_features
# Set the validation features as the new validation set for xgboost
val_extractor = VGG16_model.predict(x_val)
val_features = val_extractor.reshape(val_extractor.shape[0], -1)
X_for_val = val_features
val_data = (X_for_val, y_val)
metrics = ["auc", "map", "logloss"] # AUC, PRECISION, LOSS
#XGBoost Time!
# Train the XGBoost Classifier
import xgboost as xgb 
xgb_model = xgb.XGBClassifier(booster='gbtree', scale_pos_weight=55.7226027397, learning_rate=0.1, n_estimators=150)
xgb_model.fit(X_for_training, y_train, eval_set=((X_for_training, y_train), val_data), eval_metric=metrics)
# Use feature extractor for test data to get the test features
X_test_feature = VGG16_model.predict(x_test)
X_test_features = X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)
prediction = xgb_model.predict(X_test_features) #Predict using XGBoost
from sklearn import metrics
print ("Loss: ", metrics.log_loss(y_test, prediction))
print ("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print ("Recall: ", metrics.recall_score(y_test, prediction))
print ("Precision: ", metrics.precision_score(y_test, prediction))
print ("AUC: ", metrics.roc_auc_score(y_test, prediction))
print ("F1_Score: ", metrics.f1_score(y_test, prediction))
# Upper Left: True Positives
# Upper RIght: False Positives
# Lower Left: False Negatives
# Lower Right: True Negatives

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, prediction)
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
# Accuracy of each model

plt.plot(vggTrainer.history['val_binary_accuracy'], label="CNN_VGG")
plt.plot(bidirectional_trainer.history['val_binary_accuracy'], label="RNN_Bidirectional")
plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Loss of each model
plt.plot(vggTrainer.history['val_loss'], label="CNN_VGG")
plt.plot(bidirectional_trainer.history['val_loss'], label="RNN_Bidirectional")
plt.plot(x_axis, results['validation_1']['logloss'], label='XGBoost')
plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Recall of each model

plt.plot(vggTrainer.history['val_recall_m'], label="CNN_VGG")
plt.plot(bidirectional_trainer.history['val_recall_m'], label="RNN_Bidirectional")
plt.title("Recall Comparison")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.show()
# ROC of each model
plt.plot(vggTrainer.history['val_ROC'], label="CNN_VGG")
plt.plot(bidirectional_trainer.history['val_ROC'], label="RNN_Bidirectional")
plt.plot(x_axis, results['validation_1']['auc'], label='XGBoost')
plt.title("ROC Comparison")
plt.xlabel("Epoch")
plt.ylabel("ROC")
plt.legend()
plt.show()
# Precision of each model

plt.plot(vggTrainer.history['val_precision_m'], label="CNN_VGG")
plt.plot(bidirectional_trainer.history['val_precision_m'], label="RNN_Bidirectional")
plt.plot(x_axis, results['validation_1']['map'], label='XGBoost')
plt.title("Precision Comparison")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.show()
# F1 Score of each model

plt.plot(vggTrainer.history['val_f1_m'], label="CNN_VGG")
plt.plot(bidirectional_trainer.history['val_f1_m'], label="RNN_Bidirectional")
plt.title("F1 Score Comparison")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

# load and evaluate a saved model
import preprocessing

from numpy import loadtxt
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pylab as plt
import numpy as np
import glob
import operator

# load model
cnn = load_model('model.h5')
# summarize model.
cnn.summary()

classLabels = ['Kanker', 'Normal', 'Tumor']
test_path = "0173-48.jpg"
picture_size = 64

test_datagen = preprocessing.imageDatagen()

idx = 0
result = {
  "Kanker": 0,
  "Normal": 0,
  "Tumor": 0
}
result_key = list(result)
for address in glob.glob("/koding/python/TA/test/norma/*"):
    idx += 1
    test_image = image.load_img(address, target_size = (picture_size, picture_size))
    imgplot = plt.imshow(test_image)
    x = image.img_to_array(test_image)
    x = np.expand_dims(x, axis=0)

    test_image_try = np.vstack([x])

    prediction = cnn.predict(test_image_try, verbose=0)

    for i in range(len(result_key)):
        if (prediction[0][i] == 1):
            result[result_key[i]] += 1

for i in range(len(result_key)):
    result[result_key[i]] /= idx

result_max = max(result.items(), key=operator.itemgetter(1))[0]

print(result)
print(result_max)


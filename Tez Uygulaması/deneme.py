# Gerekli kütüphaneleri içe aktarın
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing import image
from keras.models import Sequential
import PIL.Image

img = PIL.Image.open("C:\Users\Hakan\OneDrive\Desktop\Tez Uygulaması\Veriseti\Audi\A7\audi-a7-gen-2010-2014-2.jpg")

veri_seti_dizini = 'Veriseti'

# Model oluşturma
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Örnek olarak 10 sınıflı bir veri seti varsayılmıştır
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_data = train_datagen.flow_from_directory(veri_seti_dizini, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Modeli eğitme
model.fit(train_data, epochs=5)

# Eğitilmiş modeli kaydetme
model.save('araba_modeli.h5')

# Eğitilmiş modeli yükleyin
model = keras.models.load_model('araba_modeli.h5')

# Tahmin yapmak için kullanılacak görüntüyü yükleyin ve ön işleme yapın
img_path = '/path/to/test/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Tek bir örneklik veri kümesi oluşturmak için boyut ekleyin
img_array /= 255.0  # Normalizasyon

# Tahmin yapın
predictions = model.predict(img_array)

# Tahmin sonuçlarını görüntüleyin
print(predictions)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Diretório das imagens (subpastas por cor)
data_dir = 'data'
img_size = (224, 224)
batch_size = 32
seed = 42

# Carrega o dataset de treino e validação
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

# ✅ Salva as classes ANTES de aplicar map()
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Aplica o preprocessamento da ResNet50
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Otimiza o desempenho
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Modelo base: ResNet50 sem o topo (congelada)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

# Adiciona camadas finais
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Compilação do modelo
model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Salva o modelo
model.save("vehicle_color_classifier_resnet50.h5")
print("✅ Modelo salvo como vehicle_color_classifier_resnet50.h5")

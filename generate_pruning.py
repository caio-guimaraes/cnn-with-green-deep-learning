import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet import preprocess_input

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

data_dir = 'data'
img_size = (224, 224)
batch_size = 32
seed = 37

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


# Define parâmetros de pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000  # ajuste com base no número total de steps
    )
}

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

model = load_model("vehicle_color_classifier_resnet50_savedmodel")
model_for_pruning = prune_low_magnitude(model, **pruning_params)


model_for_pruning.compile(optimizer=Adam(1e-4),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Re-treinar por algumas épocas
model_for_pruning.fit(train_ds, validation_data=val_ds, epochs=5,
                      callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model.save("pruned_vehicle_color_classifier", save_format="tf")

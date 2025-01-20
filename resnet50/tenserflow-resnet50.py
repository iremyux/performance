import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import time
import psutil

# Monitor CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

print("Starting TensorFlow script...")

# 1. Prepare Data (Subset)
print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Data loaded.")

# Use a subset of the data (e.g., first 2000 samples for training, 500 for testing)
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]

# ResNet50 requires 224x224 input size
x_train = tf.image.resize(x_train, (224, 224)) / 255.0
x_test = tf.image.resize(x_test, (224, 224)) / 255.0
print("Data resized and normalized.")

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("Labels one-hot encoded.")

# Convert to TensorFlow Datasets for efficient batching
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=2000).batch(32).prefetch(tf.data.AUTOTUNE)
print("Train dataset prepared with TensorFlow Dataset API.")

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
print("Test dataset prepared with TensorFlow Dataset API.")

# 2. Load Model
print("Loading ResNet50 model...")
model = ResNet50(weights=None, input_shape=(224, 224, 3), classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded and compiled.")

# 3. Train Model
num_epochs = 5  # Reduced epochs for faster training
print("Starting training...")
cpu_usages = []

start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    batches = 0
    
    print(f"Epoch {epoch + 1}/{num_epochs} starting...")
    for x_batch, y_batch in train_dataset:  # Limit to 50 batches for faster execution
        loss = model.train_on_batch(x_batch, y_batch)
        epoch_loss += loss[0]  # Accumulate batch loss
        batches += 1
        cpu_usages.append(get_cpu_usage())
    
    epoch_time = time.time() - epoch_start_time
    avg_epoch_loss = epoch_loss / batches
    print(f"Epoch {epoch + 1}/{num_epochs} completed.")
    print(f"  - Loss: {avg_epoch_loss:.4f}")
    print(f"  - Time: {epoch_time:.2f} seconds")
    print(f"  - CPU Usage: {cpu_usages[-1]:.2f}%")

train_time = time.time() - start_time
print(f"Training complete. Total Training Time: {train_time:.2f} seconds")
print(f"Average CPU Usage During Training: {sum(cpu_usages) / len(cpu_usages):.2f}%")

# 4. Evaluate Model
print("Starting evaluation...")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"Evaluation complete. Test Accuracy: {test_accuracy * 100:.2f}%")

# Measure Inference Latency (Using a Larger Batch for Efficiency)
print("Measuring inference latency...")
inference_times = []
for x_batch, _ in test_dataset.take(10):  # Use a larger number of batches for latency measurement
    start_inference = time.time()
    model.predict(x_batch)
    inference_times.append(time.time() - start_inference)

print(f"Average Inference Latency per Batch: {sum(inference_times) / len(inference_times):.4f} seconds")

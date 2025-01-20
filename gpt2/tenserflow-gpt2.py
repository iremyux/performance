import tensorflow as tf
import time
import psutil
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# Helper function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

def train_and_test_llama2():
    print("Initializing...\n")

    # Define parameters
    model_name = "distilgpt2"  # Replace with a smaller LLaMA model if available
    batch_size = 2
    epochs = 5
    learning_rate = 5e-5

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    print("Model loaded successfully!\n")

    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset["train"]["text"][:2000]  # Reduce dataset size
    test_texts = dataset["test"]["text"][:500]  # Reduce dataset size

    def tokenize_function(texts):
        return tokenizer(
            texts,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

    print("Tokenizing dataset...")
    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]},
        train_encodings["input_ids"]
    )).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]},
        test_encodings["input_ids"]
    )).batch(1)

    print("Dataset prepared!\n")

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Training loop
    print("Starting training...")
    training_start_time = time.time()
    model.trainable = True

    training_cpu_usage = []
    latencies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for step, (inputs, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                start_time = time.time()
                outputs = model(inputs, training=True)
                logits = outputs.logits
                loss = loss_fn(labels, logits)
                latency = time.time() - start_time
                latencies.append(latency)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()

            training_cpu_usage.append(get_cpu_usage())

        epoch_loss /= (step + 1)
        print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}\n")

    training_time = time.time() - training_start_time
    avg_training_latency = sum(latencies) / len(latencies)
    avg_cpu_usage = sum(training_cpu_usage) / len(training_cpu_usage)
    print(f"Training completed in {training_time:.2f} seconds!\n")

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average training latency per step: {avg_training_latency:.4f} seconds")
    print(f"Average CPU usage during training: {avg_cpu_usage:.2f}%")

    # Testing
    print("Starting testing...")
    testing_start_time = time.time()
    model.trainable = False

    test_latencies = []
    correct_predictions = 0
    total_predictions = 0

    for step, (inputs, labels) in enumerate(test_dataset):
        # Cast inputs to int64
        inputs["input_ids"] = tf.cast(inputs["input_ids"], tf.int64)
        inputs["attention_mask"] = tf.cast(inputs["attention_mask"], tf.int64)

        start_time = time.time()
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)
        latency = time.time() - start_time
        test_latencies.append(latency)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test {step + 1}: {generated_text}")

        # Simple accuracy metric: check if generated text contains part of the input text
        if any(word in generated_text for word in tokenizer.decode(inputs["input_ids"][0]).split()):
            correct_predictions += 1
        total_predictions += 1

    testing_time = time.time() - testing_start_time
    avg_test_latency = sum(test_latencies) / len(test_latencies)
    test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("\nTesting completed!")
    print(f"Total testing time: {testing_time:.2f} seconds")
    print(f"Average latency per test: {avg_test_latency:.4f} seconds")
    print(f"Test accuracy: {test_accuracy:.2%}")    

if __name__ == "__main__":
    train_and_test_llama2()

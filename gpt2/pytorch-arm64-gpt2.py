import os
import requests
import torch
import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Dataset class for processing
class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=64):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# Function to download Wikitext
def download_wikitext(version="wikitext-2", save_dir="data"):
    """
    Downloads the Wikitext dataset from a public URL and saves it locally.
    
    Args:
        version (str): Dataset version to download (e.g., "wikitext-2-raw-v1").
        save_dir (str): Directory to save the dataset files.
    
    Returns:
        tuple: Paths to the train and test datasets.
    """
    base_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/"
    dataset_files = {
        "train": f"{version}/train.txt",
        "test": f"{version}/test.txt"
    }
    os.makedirs(save_dir, exist_ok=True)

    file_paths = {}
    for split, file_name in dataset_files.items():
        file_url = base_url + file_name
        file_path = os.path.join(save_dir, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(file_url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Saved {split} data to {file_path}")
        file_paths[split] = file_path

    return file_paths["train"], file_paths["test"]

# Helper function to load text data from a file
def load_text_data(file_path, sample_count=None):
    """
    Loads text data from a local file.
    
    Args:
        file_path (str): Path to the file containing the text data.
        sample_count (int): Maximum number of samples to load (optional).
    
    Returns:
        list: A list of text samples.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.readlines()
    return texts[:sample_count] if sample_count else texts

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}\n")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    print("Model loaded successfully!\n")

    # Download and prepare dataset
    print("Downloading and loading dataset...")
    train_path, test_path = download_wikitext()
    train_texts = load_text_data(train_path, sample_count=2000)
    test_texts = load_text_data(test_path, sample_count=500)

    print("Preparing dataset...")
    train_dataset = SimpleDataset(tokenizer, train_texts)
    test_dataset = SimpleDataset(tokenizer, test_texts)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Dataset prepared!\n")

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    training_start_time = time.time()
    model.train()

    epoch_losses = []
    training_cpu_usage = []
    latencies = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            latency = time.time() - start_time
            latencies.append(latency)

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            training_cpu_usage.append(get_cpu_usage())

        epoch_loss /= len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds. Loss: {epoch_loss:.4f}\n")

    training_time = time.time() - training_start_time
    avg_training_latency = sum(latencies) / len(latencies)
    avg_cpu_usage = sum(training_cpu_usage) / len(training_cpu_usage)
    avg_training_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Training completed in {training_time:.2f} seconds!\n")

    # Testing
    print("Starting testing...")
    testing_start_time = time.time()
    model.eval()

    test_latencies = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_time = time.time()
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
            latency = time.time() - start_time
            test_latencies.append(latency)

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Test {step + 1}: {generated_text}")

            # Simple accuracy metric: check if generated text contains part of the input text
            if any(word in generated_text for word in tokenizer.decode(input_ids[0]).split()):
                correct_predictions += 1
            total_predictions += 1

    testing_time = time.time() - testing_start_time
    avg_test_latency = sum(test_latencies) / len(test_latencies)
    test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("\nTesting completed!")
    print(f"Total testing time: {testing_time:.2f} seconds")
    print(f"Average latency per test: {avg_test_latency:.4f} seconds")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average training latency per step: {avg_training_latency:.4f} seconds")
    print(f"Average CPU usage during training: {avg_cpu_usage:.2f}%")
    print(f"Average training loss: {avg_training_loss:.4f}")

if __name__ == "__main__":
    train_and_test_llama2()

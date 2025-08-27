import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
from transformers import (
    Wav2Vec2Processor,   
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
import torch
import evaluate

# Visualization function for the emotions count
def visualize_data(data_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data_path, x="Emotions", order=data_path["Emotions"].value_counts().index)
    plt.title("Count of Emotions", size=16)
    plt.ylabel("Count", size=12)
    plt.xlabel("Emotions", size=12)
    sns.despine()
    plt.show()

# Custom Dataset class to load audio samples one at a time
class AudioDataset(TorchDataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):  
        try:
            # Load the audio file
            audio, sr = librosa.load(self.file_paths[idx], sr=16000)

            # Process audio using Wav2Vec2 processor
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

            # Return the input values as a tensor
            input_values = inputs.input_values.squeeze(0)  # Remove batch dimension

            # Return the input values and label
            return {"input_values": input_values, "labels": torch.tensor(self.labels[idx])}
        except Exception as e:
            print(f"Error processing file {self.file_paths[idx]}: {e}")
            return None

if __name__ == "__main__":
    # Set multiprocessing start method (necessary on Windows)
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Load the CSV file directly
    csv_file = "data_path.csv"
    data_path = pd.read_csv(csv_file)

    # Visualize the data
    visualize_data(data_path)

    # Map emotions to numerical labels
    emotion_mapping = {e: i for i, e in enumerate(data_path["Emotions"].unique())}
    data_path["Label"] = data_path["Emotions"].map(emotion_mapping)

    # Initialize the processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # Split the data into train and eval datasets
    train_data, eval_data = train_test_split(data_path, test_size=0.2, random_state=42)

    # Create train and eval datasets
    train_dataset = AudioDataset(
        file_paths=train_data["Path"].values, 
        labels=train_data["Label"].values, 
        processor=processor
    )
    eval_dataset = AudioDataset(
        file_paths=eval_data["Path"].values, 
        labels=eval_data["Label"].values, 
        processor=processor
    )

    # Filter out None entries
    train_dataset = [item for item in train_dataset if item is not None]
    eval_dataset = [item for item in eval_dataset if item is not None]

    # Use a Data Collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=processor.feature_extractor)

    # Initialize the model (move this before `training_args`)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base", 
        num_labels=len(emotion_mapping)
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./wav2vec2-results",
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=20,  
        gradient_accumulation_steps=10,
        fp16=True,
        num_train_epochs=5,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Metric computation function
    def compute_metrics(pred):
        metric = evaluate.load("accuracy")
        predictions = np.argmax(pred.predictions, axis=-1)
        references = pred.label_ids
        return metric.compute(predictions=predictions, references=references)

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=data_collator,  # Add collator for padding
        compute_metrics=compute_metrics,  # Metric computation function
    )

    # Train the model
    trainer.train()



# E:\ser\test_wav.py:124: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
#   trainer = Trainer(
#   0%|                                                                                                                                        | 0/240 [00:00<?, ?it/s]2024-12-08 02:48:04.588438: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-12-08 02:48:08.309550: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From E:\spectro\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

# E:\spectro\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
# Downloading builder script: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 4.20k/4.20k [00:00<00:00, 701kB/s]
# {'eval_loss': 1.6547746658325195, 'eval_accuracy': 0.42827784628031235, 'eval_runtime': 4826.8793, 'eval_samples_per_second': 0.504, 'eval_steps_per_second': 0.063, 'epoch': 1.0}
#  20%|████████████████████████                                                                                              | 49/240 [34:34:47<154:34:43, 2913.53s/it]E:\spectro\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
# {'eval_loss': 1.3470487594604492, 'eval_accuracy': 0.6185778873818332, 'eval_runtime': 4847.2254, 'eval_samples_per_second': 0.502, 'eval_steps_per_second': 0.063, 'e': 0.5, 'eval_steps_per_second': 0.063, 'epoch': 3.0}
#  61%|████████████████████████████████████████████████████████████████████████▎                                             | 147/240 [96:32:21<49:07:59, 1901.93s/it]E:\spectro\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
# {'eval_loss': 1.1076452732086182, 'eval_accuracy': 0.7254418413481298, 'eval_runtime': 4852.192, 'eval_samples_per_second': 0.501, 'eval_steps_per_second': 0.063, 'epoch': 4.0}
#  82%|███████████████████████████████████████████████████████████████████████████████████████████████▌                     | 196/240 [126:59:54<23:31:48, 1925.19s/it]E..)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
# {'eval_loss': 1.1076452732086182, 'eval_accuracy': 0.7254418413481298, 'eval_runtime': 4852.192, 'eval_samples_per_second': 0.501, 'eval_steps_per_second': 0.063, 'epoch': 4.0}
#  82%|███████████████████████████████████████████████████████████████████████████████████████████████▌                     | 196/240 [126:59:54<23:31:48, 1925.19s/it]E{'eval_loss': 1.1076452732086182, 'eval_accuracy': 0.7254418413481298, 'eval_runtime': 4852.192, 'eval_samples_per_second': 0.501, 'eval_steps_per_second': 0.063, 'epoch': 4.0}
#  82%|███████████████████████████████████████████████████████████████████████████████████████████████▌                     | 196/240 [126:59:54<23:31:48, 1925.19s/it]Eoch': 4.0}
#  82%|███████████████████████████████████████████████████████████████████████████████████████████████▌                     | 196/240 [126:59:54<23:31:48, 1925.19s/it]E 82%|███████████████████████████████████████████████████████████████████████████████████████████████▌                     | 196/240 [126:59:54<23:31:48, 1925.19s/it]E:\spectro\Lib\site-packages\torch\utils\checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
# ..)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
# {'eval_loss': 1.071927547454834, 'eval_accuracy': 0.7455815865187012, 'eval_runtime': 4857.8733, 'eval_samples_per_second': 0.501, 'eval_steps_per_second': 0.063, 'epoch': 4.9}
# {'train_runtime': 553261.053, 'train_samples_per_second': 0.088, 'train_steps_per_second': 0.0, 'train_loss': 1.3838055928548176, 'epoch': 4.9}
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 240/240 [153:41:01<00:00, 2305.25s/it] 

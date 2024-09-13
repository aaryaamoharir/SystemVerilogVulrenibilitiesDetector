import pandas as pd

# Load dataset
df = pd.read_csv('/kaggle/input/codedatabase2/codeDatabase 2.csv')

from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.metrics import classification_report

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)  # Adjust num_labels based on your dataset
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['has_security_issues'])

# split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(df['code_snippet'], y, test_size=0.2, random_state=42)

# Prepare dataset for CodeBERT
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: torch.squeeze(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = CodeDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length=512)
test_dataset = CodeDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_length=512)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: classification_report(p.label_ids, p.predictions.argmax(-1), output_dict=True)
)

trainer.train()

# results 
results = trainer.evaluate()
print(results)

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import classification_report
import torch
# Train model
trainer.train()

# predictions 
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# ensure label_encoder.classes_ is a list of class names
class_names = [str(class_name) for class_name in label_encoder.classes_]


# used for debugging 
print("y_true type:", type(y_true))
print("y_true:", y_true)
print("y_pred type:", type(y_pred))
print("y_pred:", y_pred)
print("class_names type:", type(class_names))
print("class_names:", class_names)

# print classification report 
print(f"Epoch {training_args.num_train_epochs}")
print(classification_report(y_true, y_pred, target_names=class_names))




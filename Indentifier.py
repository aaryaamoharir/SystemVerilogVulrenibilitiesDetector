import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


model.eval()

# defined a function to extract features 
def extract_features(text, tokenizer, model, max_length=512):
    # Tokenize the text with padding and truncation
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    if len(hidden_states) < 4:
        raise ValueError("Not enough hidden states returned by the model")
    
    token_vecs = []
    for layer in range(-4, 0):
        token_vecs.append(hidden_states[layer][0])
    
    features = []
    for token in token_vecs:
        features.append(torch.mean(token, dim=0))
    
    # returns the features as a tensor 
    return torch.stack(features)


# load dataset
df = pd.read_csv('/kaggle/input/codesnippets/code_snippets.csv')
# used to check the type of each data point 
print(type(df['code_snippet'].iloc[2]))

df['code_snippet'].fillna('', inplace=True)
df['code_snippet'] = df['code_snippet'].apply(lambda x: x if isinstance(x, str) and x.strip() != '' else '[empty]')

features = [extract_features(row['code_snippet'], tokenizer, model) for _, row in df.iterrows()]
features = torch.stack(features).numpy()

features = []
for i in range(len(df)):
    # Extract features without wrapping in an additional list
    feature_tensor = extract_features(df.iloc[i]['code_snippet'], tokenizer, model)
    features.append(feature_tensor)

features = torch.stack(features)

features = features.numpy()

types = df['has_security_issue'].values
types

# reshapes the array for better performance 
num_elements = features.size
correct_shape = (num_elements // 96, 96)
features_reshaped = features.reshape(correct_shape)
print(f"Shape of reshaped features: {features_reshaped.shape}")
print(f"Shape of reshaped features: {features_reshaped.shape}")

# labels is already a 1D array of size 1000, we need to reshape it to 2D for concatenation
labels_reshaped = types.reshape((-1, 1))

padding_size = features_reshaped.shape[0] - labels_reshaped.shape[0]
labels_reshaped = np.pad(labels_reshaped, ((0, padding_size), (0, 0)), mode='constant')



# concatenates the feature array with the label array horizontally
dataset = np.hstack((features_reshaped, labels_reshaped))

# used to check the shape of the dataset 
print(f"Shape of dataset: {dataset.shape}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
!pip install transformers torch


# converts text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['code_snippet'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['has_security_issue'])

# split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# logistic regression
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

# results from the model 
# achieved an accuracy of 76% with the codeSnippets dataset 
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


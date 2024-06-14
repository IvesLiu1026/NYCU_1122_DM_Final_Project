import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle

# Load Data & Embedding
train_behaviors = pd.read_csv('train/train_behaviors.tsv', sep='\t', names=['id', 'user_id', 'time', 'clicked_news', 'impressions'])
train_news = pd.read_csv('train/train_news.tsv', sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
train_entity_embedding = pd.read_csv('train/train_entity_embedding.vec', sep='\t', header=None)
test_behaviors = pd.read_csv('test/test_behaviors.tsv', sep='\t', names=['id', 'user_id', 'time', 'clicked_news', 'impressions'])
test_news = pd.read_csv('test/test_news.tsv', sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
test_entity_embedding = pd.read_csv('test/test_entity_embedding.vec', sep='\t', header=None)

# load BERT model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = BertModel.from_pretrained('bert-base-uncased', config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    print('Using CPU')
model.to(device)

def get_news_embeddings(new_df, filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            news_embeddings = pickle.load(f)
        print(f"Loaded embeddings from {filename}")
    else:
        news_embeddings = {}
        for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Processing news embeddings"):
            inputs = tokenizer(row['title'], return_tensors='pt', truncation=True, padding=True, max_length=32).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-4:]
            hidden_states = [state.to(device) for state in hidden_states]
            news_embeddings[row['news_id']] = torch.mean(torch.cat(hidden_states, dim=0), dim=0).mean(dim=0).detach().cpu().numpy()
        with open(filename, 'wb') as f:
            pickle.dump(news_embeddings, f)
        print(f"Saved embeddings to {filename}")
    return news_embeddings

train_news_embeddings = get_news_embeddings(train_news, 'train_news_embeddings.pkl')
test_news_embeddings = get_news_embeddings(test_news, 'test_news_embeddings.pkl')

# define
class UserEmbeddingGRU(torch.nn.Module):
    def __init__(self):
        super(UserEmbeddingGRU, self).__init__()
        self.gru = torch.nn.GRU(input_size=768, hidden_size=768, batch_first=True)

    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n.squeeze(0)
    

user_embedding_model = UserEmbeddingGRU().to(device)
# torch.save(user_embedding_model.state_dict(), 'user_embedding_model.pth')

def get_user_embedding(click_history, news_embeddings, model):
    embeddings = [news_embeddings[news] for news in click_history.split() if news in news_embeddings]
    if embeddings:
        embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            user_embedding = model(embeddings)
        return user_embedding.cpu().numpy()
    else:
        return np.zeros((1, 768))
    
tqdm.pandas()
train_behaviors['user_embedding'] = train_behaviors['clicked_news'].progress_apply(lambda x: get_user_embedding(x, train_news_embeddings, user_embedding_model))


def clean_impressions(impressions):
    cleaned_impressions = []
    for impression in impressions.split():
        if '-' in impression:
            news_id, label = impression.split('-')
            if label == '2':
                cleaned_impressions.append(news_id)
    return ' '.join(cleaned_impressions)

train_behaviors['cleaned_impressions'] = train_behaviors['impressions'].apply(clean_impressions)

# train dataset
train_labels = []
train_features = []
for _, row in tqdm(train_behaviors.iterrows(), total=train_behaviors.shape[0], desc="Processing train behaviors"):
    user_embedding = row['user_embedding']
    for impression in row['impressions'].split():
        if '-' in impression:
            news_id, label = impression.split('-')
            if news_id in train_news_embeddings:
                train_features.append(np.dot(user_embedding, train_news_embeddings[news_id].T).flatten())
                train_labels.append(int(label))

train_features = np.array(train_features)
train_labels = np.array(train_labels)

class EnhancedNN(torch.nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = torch.nn.Linear(1, 512)  # Change input size to 1
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

def train_enhanced_model(train_features, train_labels):
    model = EnhancedNN().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(50): 
        model.train()
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    return model

#load trained model from final_model.pth
trained_model = EnhancedNN().to(device)
state_dict = torch.load('final_model.pth')
trained_model.load_state_dict(state_dict)

print(trained_model)
# trained_model = train_enhanced_model(train_features, train_labels)
# torch.save(trained_model.state_dict(), 'final_model.pth')

def generate_predictions(behaviors_df, news_df, user_embedding_model, news_embeddings, trained_model):
    predictions = []
    for idx, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc="Generating predictions"):
        user_embedding = get_user_embedding(row['clicked_news'], news_embeddings, user_embedding_model)
        if len(user_embedding) == 0:
            print(f"No user embedding for row {idx}")
        impression_ids = [imp.split('-')[0] for imp in row['impressions'].split() if '-' in imp]
        for news_id in impression_ids:
            if news_id in news_embeddings:
                news_embedding = news_embeddings[news_id]
                feature = np.dot(user_embedding, news_embedding.T).flatten()
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = trained_model(feature_tensor).item()
                predictions.append({'id': row['id'], 'news_id': news_id, 'prediction': pred})
            else:
                print(f"News ID {news_id} not found in news_embeddings for row {idx}")
    return pd.DataFrame(predictions)

# Generate predictions and debug the contents
predictions = generate_predictions(test_behaviors, test_news, user_embedding_model, test_news_embeddings, trained_model)
print(predictions.head())  # Check the first few rows to ensure 'id' column is present

def create_submission_file(predictions):
    if 'id' not in predictions.columns:
        print("Error: 'id' column not found in predictions DataFrame.")
        print(predictions.columns)
        return
    predictions.sort_values(by='id', inplace=True)
    grouped = predictions.groupby('id').apply(lambda x: x.nlargest(15, 'prediction')).reset_index(drop=True)
    submission = grouped.pivot(index='id', columns='news_id', values='prediction').fillna(0)
    submission.columns = ['p'+str(i) for i in range(1, len(submission.columns)+1)]
    submission = submission.reindex(columns=['p'+str(i) for i in range(1, 16)], fill_value=0)
    submission.to_csv('submission.csv', index=True)

create_submission_file(predictions)

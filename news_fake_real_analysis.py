import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Step 1: Load and Combine Data ===
def load_and_label_data(fake_paths, real_paths):
    dataframes = []
    for fake_path, real_path in zip(fake_paths, real_paths):
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)

        fake_df['label'] = 0  # Fake news
        real_df['label'] = 1  # Real news

        combined_df = pd.concat([fake_df, real_df], ignore_index=True)
        dataframes.append(combined_df)
    
    return pd.concat(dataframes, ignore_index=True)

# === Step 2: Preprocess Text Data ===
def preprocess_data_optimized(df):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    texts = df['title'].fillna("").tolist()
    entity_counts_list = []

    for doc in nlp.pipe(texts, batch_size=1000):
        entity_counts = {'ORG': 0, 'GPE': 0, 'PERSON': 0}
        for ent in doc.ents:
            if ent.label_ in entity_counts:
                entity_counts[ent.label_] += 1
        entity_counts_list.append(entity_counts)

    df['entities'] = entity_counts_list
    df['length'] = df['title'].apply(lambda x: len(str(x).split()))
    df['sentiment_polarity'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    entity_df = pd.json_normalize(df['entities'])
    return pd.concat([df, entity_df], axis=1)

# === Step 3: Define Dataset and Model ===
class FakeRealNewsModel(nn.Module):
    def __init__(self, input_dim):
        super(FakeRealNewsModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# === Step 4: Visualization ===
def visualize_relationships(df):
    # 1. Bar Chart: Frequency of Entities
    entity_cols = ['ORG', 'GPE', 'PERSON']
    entity_counts = df[entity_cols].sum()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=entity_counts.index, y=entity_counts.values, palette="viridis")
    plt.title("Frequency of Named Entities in Articles")
    plt.ylabel("Count")
    plt.xlabel("Entity Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # 2. Scatter Plot: Correlation between entity counts and engagement
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='ORG', y='length', label="ORG vs. Length", color='blue')
    sns.scatterplot(data=df, x='GPE', y='sentiment_polarity', label="GPE vs. Sentiment", color='green')
    plt.title("Correlation Between Entity Counts and Metrics")
    plt.xlabel("Entity Count")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Heatmap: Entity counts vs. Metrics
    heatmap_data = df[['ORG', 'GPE', 'PERSON', 'length', 'sentiment_polarity']].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap: Entity Counts and Metrics")
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    fake_paths = ["politifact_fake.csv", "gossipcop_fake.csv"]
    real_paths = ["politifact_real.csv", "gossipcop_real.csv"]

    df = load_and_label_data(fake_paths, real_paths)
    print("Starting Preprocessing...")
    df = preprocess_data_optimized(df)
    print("Preprocessing Complete!")

    # Drop non-numeric columns before scaling
    numeric_cols = ['length', 'sentiment_polarity', 'ORG', 'GPE', 'PERSON', 'label']
    df = df[numeric_cols]

    visualize_relationships(df)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df.drop(columns=['label']))
    test_features = scaler.transform(test_df.drop(columns=['label']))

    train_labels = train_df['label'].values
    test_labels = test_df['label'].values

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float32).to(device),
        torch.tensor(train_labels, dtype=torch.float32).to(device)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = train_features.shape[1]
    model = FakeRealNewsModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train Model
    model.to(device)
    for epoch in range(10):
        model.train()
        epoch_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/10, Loss: {epoch_loss / len(train_loader):.4f}")

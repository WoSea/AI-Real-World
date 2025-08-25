import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# 1. Parameters
vocab_size = 10000   # keep only 10k most common words
max_len = 200        # each review max 200 words
embedding_dim = 32   # embedding vector for each word

# 2. Load dataset (IMDB sentiment: positive/negative)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Sample review (before padding):", x_train[0][:10])
print("Label:", y_train[0])

# 3. Pad sequences to the same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print("Shape after padding:", x_train.shape)

# 4. Build RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(64, activation='tanh'),  # RNN layer
    Dense(1, activation='sigmoid')     # output binary classification
])

# 5. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6. Train
history = model.fit(x_train, y_train, epochs=3, batch_size=64,
                    validation_data=(x_test, y_test))

# 7. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# PyTorch equivalent
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# 1. Parameters
vocab_size = 10000
embedding_dim = 32
hidden_size = 64
max_len = 200
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load IMDB dataset
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

tokenizer = get_tokenizer("basic_english")

# Build vocab
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"], max_tokens=vocab_size)
vocab.set_default_index(vocab["<pad>"])

# Encode text
def encode(text):
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.long)

# Collate fn for DataLoader
def collate_batch(batch):
    label_map = {"neg": 0, "pos": 1}
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_map[label])
        tokens = encode(text)[:max_len]  # cắt bớt nếu dài
        texts.append(tokens)
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    if texts.size(1) < max_len:  # pad thêm nếu ngắn
        pad_len = max_len - texts.size(1)
        pad_tensor = torch.full((texts.size(0), pad_len), vocab["<pad>"], dtype=torch.long)
        texts = torch.cat([texts, pad_tensor], dim=1)
    return texts.to(device), torch.tensor(labels, dtype=torch.float32).to(device)

train_iter, test_iter = IMDB(split=('train', 'test'))
train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=batch_size, collate_fn=collate_batch)

# 3. Define RNN model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab["<pad>"])
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        out = self.fc(hidden.squeeze(0))
        return self.sigmoid(out)

model = RNNClassifier(vocab_size, embedding_dim, hidden_size).to(device)

# 4. Loss & optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 6. Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts).squeeze()
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.4f}")

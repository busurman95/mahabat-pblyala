import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
with open('recipes.json', 'r', encoding='utf-8') as f:
    recipes = json.load(f)

train_texts = []
for recipe in recipes:
    title = recipe['title']
    ingredients = ' '.join(recipe['ingredients']) if isinstance(recipe['ingredients'], list) else recipe['ingredients']
    instructions = ' '.join(recipe['instructions']) if isinstance(recipe['instructions'], list) else recipe['instructions']
    train_texts.append(' '.join([title, ingredients, instructions]))

train_labels = [recipe['category'] for recipe in recipes]

# Кодирование меток категорий
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
num_classes = len(set(train_labels))

# Разделение данных на обучающий и валидационный наборы
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Создание словаря
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>', filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1

# Подготовка данных
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=100, padding='post')

val_sequences = tokenizer.texts_to_sequences(val_texts)
val_padded = pad_sequences(val_sequences, maxlen=100, padding='post')

# Создание наборов данных TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((train_padded, train_labels)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_padded, val_labels)).batch(32)

# Определение архитектуры нейронной сети

class RecipeClassifier(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RecipeClassifier, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(hidden_dim, return_sequences=True)
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.dense2 = Dense(output_dim)  # Без активации softmax

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense1(x)
        x = tf.reduce_mean(x, axis=1)
        logits = self.dense2(x)  # Получаем необработанные логиты
        return tf.nn.softmax(logits)  # Применяем softmax

# Создание модели, функции потерь и оптимизатора
model = RecipeClassifier(vocab_size, embedding_dim=100, hidden_dim=128, output_dim=num_classes)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])



# Обучение нейронной сети
num_epochs = 10
for epoch in range(num_epochs):
    for batch_inputs, batch_labels in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs)
            loss = loss_fn(batch_labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Оценка на валидационных данных
    val_loss = 0.0
    for batch_inputs, batch_labels in val_dataset:
        predictions = model(batch_inputs)
        loss = loss_fn(batch_labels, predictions)
        val_loss += loss.numpy()

    print(f'Epoch: {epoch+1}, Validation Loss: {val_loss/len(val_dataset)}')

# Сохранение предобученной модели
model.save_weights('model.weights.h5')


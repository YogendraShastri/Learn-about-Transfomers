# Let's Learn About Transformers
simple, easy-to-understand explanations

## Introduction to GPT and Transformers
- The **Transformer** is a type of neural network introduced in the 2017 paper **"Attention is All You Need"** by Vaswani and others. It changed the game for natural language processing (NLP) by swapping out older recurrent neural networks (RNNs) for a technique called self-attention. This makes it faster and better at handling sequences of data, like words in a sentence, all at once.
- Straight to **Transformer**, It is a type of AI architecture used to understand and generate human-like text. It’s the backbone of modern AI models.
- **GPT (Generative Pre-trained Transformer)**, A language model created by OpenAI that uses transformers to predict the next word in a sentence. not just GPT but other models like Google's **BERT** and **LLama** predict the next word in a sentence to generate text using transformer architecture.
- **Transformer** utilize word and sentence embeddings, along with feed forward, encoding & decoding to predict the next word in a sequence. This prediction relies on various factors, including the context, topic, and speaker identity.
- Let's understand with GPT example:
  - GPT reads a lot of text during training from wikipedia, web articles etc.
  - It learns patterns, grammar, facts, and relationships between words using transformer architecture.
  - When you give it a prompt, it predicts one word at a time to create meaningful sentences.
  - Also when understand the sentiment, like apple is great in tech, here apple is company, thanks -> Happy with response. etc.

### How GPT Predicts the Next Word
1. **Input Prompt** : You give GPT some text for Example: "I love to play".
2. **Tokenization** : The text is broken into tokens (small word pieces or subwords). Example: "I" → token1, "love" → token2, "to" → token3, "play" → token4. it depends on models like for some eating is sigle token but for some models eat -> token1, ing-> token2.
3. **Transformer Layers** : GPT passes these tokens through many transformer layers. Each layer learns relationships between words — like which words influence others.
4. **Probability Prediction** : PT predicts which word is most likely to come next based on context. Example: After "I love to play", it calculates probabilities: ["football" → 0.65, "music" → 0.20, "chess" → 0.10, "potato" → 0.05]
5. **Next Word Selection** : GPT picks the highest probability word — here: "football". and process repeats 1-5 for next prompt.

### Diagram :
<div align='center'>
<img width="324" height="536" alt="image" src="https://github.com/user-attachments/assets/20852895-cdb2-44be-8ccf-ebda388a4b48" />
</div>

## Before learning about transformer lets understand some core topics:
- Word Embeddings
- High Dimensinal Space
- Contextual or sentence embedding
- etc...

### Word Embeddings 
- Machine can't understand words or text like humans do, that's why we need to convert the words in to numbers, but you cant just assign any number to an word, like if u assign car as 1 and bicycle as 0, your model wont understand the relationship or meaning.
- Neural networks are trained on huge text datasets (like Wikipedia, books, and the internet).During training, the model learns patterns and converts words into vectors (lists of numbers).
- In word embedding, each word is represented as list of numbers (vector). like cow, can have many many semantic attributes like, has_legs: Yes, is_object: No, breath: Yes, ... and so on. These attributes converted into numbers, forming a vector that describes the concept of “cow”.
- Words with similar meanings have vectors close to each other in this high-dimensional space.

### High-dimensional space :
- These vectors don’t live in 2D or 3D but in hundreds or thousands of dimensions.
- like Google’s Word2Vec has 300D (dimensions). GPT has approx ~12,000 dimensions.
- Each dimension represents some semantic aspect of words, but we don’t know exactly what each one means.
- Still, words with similar meanings end up close together in this space.

### Word Embedding Techniques & Types
<div align='center'>
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/aca54762-861c-429f-bdff-650631f25e35" />
<p align='center'>Image taken from geeksforgeeks</p>
</div>

### Contextual Word Embeddings
- Static embeddings (like **Word2Vec, GloVe**) give one fixed vector per word. But words can have multiple meanings depending on context.
  - **Example**: "dish" → could mean rice dish, Biryani dish, or dish TV.
- ChatGPT and similar systems need embeddings that change with **context → contextual embeddings**.
- In **Contextual embeddings**, the meaning of a word shifts based on surrounding words. The model modifies the base (static) embedding by adding directional vectors for context.

```bash
- sentence1 : I Like indian food.
- sentence2 : I want to eat ?

chances of suggesting Biryani is high, than Pizza as context is about indian food.
```

## Transformers Architecture Diagram:
- The Transformer Model - MachineLearningMastery.comThe transformer architecture is a neural network that excels at sequential data by using attention mechanisms and positional encoding instead of recurrence
<div align='center'>
<img width="520" height="760" alt="image" src="https://github.com/user-attachments/assets/65c1cbb2-f59f-41e3-8722-243efc62237a" />
</div>

- transformer is devided into two parts:
- Left Part : **Encoder** processes input.
- Right Part : **Decoder** generates output.

## Input Embedding
- **Input Embedding**: Converts each token into a vector. so input embedding is nothing but word embedding which we saw above.

## Positional Encoding:
- Transformers don’t have recurrence (like **RNNs**) or convolution (like **CNNs**), so by default, they don’t know the position of tokens. 
- In simple words, how does model know which token occured first, so to solve this, we add **positional encoding (PE)** to input embeddings.
- After converting tokens into embeddings E, we add a fixed or learned positional vector P to each embedding:

### Example :
- For the sentence “I love NLP”:
<div align='center'>
<img width="678" height="246" alt="image" src="https://github.com/user-attachments/assets/e61c6a18-9d1b-45a7-9a88-90b2ce180849" />
</div>

- Commonly used positional encoding is **Sinusoidal Encodings**.

## Masked Multi-Head Attention
Imagine you're reading a paragraph and trying to understand the meaning of one word, say **“bank”**.
- Sometimes, **“bank”** refers to a **river bank**.
- Sometimes, it refers to a **financial bank**.

How do you know which meaning is right?
You look around — the context. That’s what attention does: “Which other words should I look at to understand this word better?”.
But there's a catch, One attention head can focus on one type of relationship only.

- What if we want to capture multiple relationships at the same time?
- So for that Solution is → **Multi-Head Attention**
  
Multiple **“attention heads”** work in parallel —
- One head might focus on nearby words
- Another on verbs
- Another on long-distance dependencies

Then we combine everything for a richer understanding.

### 1. Queries (Q), Keys(K), and Values (V) Calculation:
**Are Q, K, and V Calculated During Training?**
- Yes — but not directly learned like weights & biases. Instead, we learn three weight matrices: **Wq, Wk, Wv**
- These are trainable parameters, just like weights in a linear layer.
- [**Note** : inference is the process of applying a trained model to new, unseen data to generate outputs]
- Then, during both training and inference, for each input token embedding X, we calculate:

$$
Q = X \cdot Wq
$$

$$
K = X \cdot Wk
$$

$$
V = X \cdot Wv
$$

**Where:**
\( X \) = input matrix  
\( Wq \) = weight matrix for **Query**   
\( Wk \) = weight matrix for **Key**  
\( Wv \) = weight matrix for **Value**

- Weights learned during training.
- Q, K, V vectors → computed on the fly from inputs during both training and inference.

<div align='center'>
<img width="860" height="610" alt="image" src="https://github.com/user-attachments/assets/7af74174-963e-4061-a6a8-9d30c38d1d9a" />
</div

### 2. Compute Similarity Scores
- The attention scores are computed by taking the dot product of the **Query matrix (Q)** with the transpose of the **Key matrix (K)** for each head.
- This calculates a measure of similarity between each query and all keys. A higher dot product indicates greater similarity.

$$
S = QK^\top
$$

**Where:**
- S = similarity score matrix  
- Q = query matrix  
- K^T = transpose of the key matrix  
- Each element Sij represents the similarity between the i'th query and the j'th key.

### 3. Scale the Scores
- The resulting dot products are then scaled by dividing by the square root of the dimensionality of the keys sqrt(dk). This scaling helps to prevent the dot products from becoming too large, which can lead to vanishing gradients during training.

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

### 4. Applying Softmax
- The Softmax function converts raw scores (called logits) into probabilities that sum up to 1. these probabilities represent the **"attention"** or **"focus"** that a query places on each key.
- for example, i love ai, how much attention each word get, is given by softmax, like i get 65% attention, love get 25% & 'AI' only 9%.

$$
A = \text{softmax}(S)
$$

<img width="580" height="380" alt="image" src="https://github.com/user-attachments/assets/58b3b94c-137f-492a-96db-f6aee31dd1d9" />


### 5. Weighted Sum of Values:
- Finally, the attention probabilities are applied to calculate a weighted sum of the Value matrix (V).
- This sum produces a context-sensitive output for each head, with each value contributing in proportion to its attention score.

$$
\text{Output} = A \cdot V
$$

```python
           ┌───────────────┐
Input X →  │  Q, K, V Proj │  → Q, K, V
           └───────────────┘
                  │
                  ▼
         Compute QKᵀ similarities
                  │
        Divide by √dₖ for scaling
                  │
           Apply softmax → Attention weights
                  │
       Multiply weights by V → Contextual Output

```

## Feed Forward Network
- The **FFN** comes after the attention mechanism and is responsible for transforming and enriching the contextual embeddings.
- A **feed-forward** layer is like a mini neural network inside the Transformer, It looks at each token (word) in your sentence independently and transforms it.
- It has **two linear layers** (simple matrix multiplications) with a **non-linear activation** (like **ReLU** or **GELU**) in the middle.
- FFN helps the model learn more complex patterns for each token, not just relationships between tokens (attention does that).
- **"Key-value memory”** analogy:
  - **Keys**: detect patterns in the token (like “verb”, “noun”, or context clues).
  - **Values**: transform those patterns into useful info for predicting the next word or understanding meaning.
- After FFN, each token is smarter and more meaningful for the next steps in the Transformer.

<div align='center'>
<img width="500" height="330" alt="image" src="https://github.com/user-attachments/assets/0ea60bf7-7237-417d-8be2-36517fc868b4" />
</div>

## Summarize the whole Process

  

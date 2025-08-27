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

## Transformers Architecture Diagram:
- The Transformer Model - MachineLearningMastery.comThe transformer architecture is a neural network that excels at sequential data by using attention mechanisms and positional encoding instead of recurrence
<div align='center'>
<img width="520" height="760" alt="image" src="https://github.com/user-attachments/assets/65c1cbb2-f59f-41e3-8722-243efc62237a" />
</div>

## 

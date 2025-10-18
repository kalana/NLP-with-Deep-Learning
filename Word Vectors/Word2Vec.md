# Lecture Note: Understanding Word2Vec and Its Mathematical Foundations

## 1. Introduction to Word2Vec

**Word2Vec** is a powerful neural network model used to represent words as numerical vectors in a continuous vector space.  
The key idea is that words with similar meanings will have similar vector representations — meaning their vectors will be close together in the embedding space.

For example:

- “king” and “queen” will have similar vector representations.  
- The model can even capture relationships like:

vector("king") - vector("man") + vector("woman") ≈ vector("queen")


This shows how Word2Vec captures **semantic and syntactic relationships** between words.

---

## 2. Motivation: Why Word Embeddings?

Before Word2Vec, traditional NLP models used **one-hot encoding**, which represented words as large, sparse vectors of 0s and 1s.  
However, this had two main issues:

1. **No notion of similarity:** Every word vector was orthogonal — there was no relationship between “cat” and “dog.”  
2. **High dimensionality:** The size of the vector equals the vocabulary size, which can be huge.

Word2Vec solved this by learning **dense, low-dimensional, continuous-valued representations** where similar words share similar vectors.

---

## 3. Two Architectures: CBOW and Skip-Gram

Word2Vec comes in two main forms:

### a. Continuous Bag of Words (CBOW)
- Predicts a **target word** given its **context** (surrounding words).  
- Example: Given “the cat ___ on the mat”, predict the missing word “sat”.
- Input: Context words  
- Output: Target word  

### b. Skip-Gram
- The inverse of CBOW. It predicts **context words** given a **target word**.  
- Example: Given “sat”, predict context words “the”, “cat”, “on”, “the”, “mat”.
- Input: Target word  
- Output: Context words  

The Skip-Gram model tends to perform better for larger datasets and captures rare word representations more effectively.

---

## 4. Mathematical Representation

Let’s formalize the Skip-Gram model (as it’s more common in explanations).

For a given word sequence \( w_1, w_2, ..., w_T \):

The **objective** is to maximize the probability of predicting the context words given the target word:

\[
\max_{\theta} \prod_{t=1}^{T} \prod_{-c \le j \le c, j \ne 0} P(w_{t+j} | w_t; \theta)
\]

where:
- \( c \) = context window size  
- \( w_t \) = target word  
- \( w_{t+j} \) = context word  
- \( \theta \) = model parameters (weights of neural network)

To simplify computation, we take the logarithm of the objective (since log converts product to sum):

\[
\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} | w_t)
\]

---

## 5. Objective Function (Loss Function)

This **objective function** defines *what the model learns*.  
It measures how well the model predicts context words from a target word.  
The better the model predicts, the higher the log-likelihood (and the lower the loss).

The probability \( P(w_O | w_I) \) is modeled using a **softmax function**:

\[
P(w_O | w_I) = \frac{\exp(v_{w_O}^\top v_{w_I})}{\sum_{w=1}^{|V|} \exp(v_w^\top v_{w_I})}
\]

where:
- \( v_{w_I} \) = input vector of the target word  
- \( v_{w_O} \) = output vector of the context word  
- \( |V| \) = vocabulary size  

This function ensures that all probabilities sum to 1 across the vocabulary.

However, computing the denominator (sum over all words) is **very expensive**, especially for large vocabularies.

---

## 6. Optimization Techniques: Negative Sampling and Hierarchical Softmax

To overcome computational inefficiency, Word2Vec uses two techniques:

### a. Negative Sampling
- Instead of updating all weights for every prediction, the model updates only a few:
- The correct (positive) word.
- A few randomly chosen incorrect (negative) words.
- This drastically reduces training time while maintaining performance.

### b. Hierarchical Softmax
- Uses a **binary tree structure** where each leaf node represents a word.
- Instead of computing probabilities for all words, the model walks a path through the tree, updating only the nodes on that path.

These methods make training feasible for large vocabularies.

---

## 7. Model Training Process

### Step 1: Input Representation
- Each input word is represented as a one-hot vector.  
- For example, for vocabulary size 10,000, the word “cat” might be `[0, 0, 0, 1, 0, ..., 0]`.

### Step 2: Hidden Layer Transformation
- Multiply the one-hot vector by a weight matrix \( W \), producing the **embedding vector**.

\[
h = W^T x
\]

Here:
- \( x \) = one-hot vector of input word  
- \( W \) = weight matrix (each column is a word embedding)  
- \( h \) = hidden layer output (the word’s vector representation)

### Step 3: Output Layer
- Multiply \( h \) by another weight matrix \( W' \) and apply softmax to get probabilities of context words.

\[
P(w_O | w_I) = \text{softmax}(W' h)
\]

### Step 4: Backpropagation and Weight Update
- Compute loss from the objective function (log-likelihood or cross-entropy).  
- Update weights \( W \) and \( W' \) using **stochastic gradient descent (SGD)** or similar optimizers.

Through many iterations, words that appear in similar contexts move closer in the embedding space.

---

## 8. Prediction (Using Trained Model)

Once trained:
- Each word has an embedding vector (a row in matrix \( W \)).
- These embeddings can be used to measure **similarity** using cosine similarity:

\[
\text{similarity}(A, B) = \frac{v_A \cdot v_B}{\|v_A\| \|v_B\|}
\]

- Similar words (like *car* and *automobile*) will have high similarity.
- These vectors are often visualized in 2D using **t-SNE** or **PCA** for interpretability.

---

## 9. Summary of Key Concepts

| Concept | Description |
|----------|--------------|
| **Word2Vec** | Neural network model to learn dense word representations |
| **CBOW** | Predicts a target word from context |
| **Skip-Gram** | Predicts context words from a target |
| **Objective Function** | Log-likelihood of context words given target |
| **Softmax Function** | Converts scores to probabilities |
| **Negative Sampling** | Efficient training with sampled negative words |
| **Embedding Matrix** | Learned dense vectors for all words |

---

## 10. Intuition Behind Learning

During training, the model adjusts embeddings so that:
- Words appearing in similar contexts (e.g., “king” and “queen”) have **similar vectors**.
- Linear relationships between words are preserved in vector space:

vector("Paris") - vector("France") ≈ vector("Berlin") - vector("Germany")


Thus, Word2Vec captures **semantic meaning** through simple neural training and a clever objective function.

---
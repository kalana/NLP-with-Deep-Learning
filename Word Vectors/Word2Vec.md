# **Lecture Note: Word2Vec, Likelihood, and the Objective Function**

## **1. Introduction to Word2Vec**

**Word2Vec** is a family of models used to learn **vector representations of words** — known as *word embeddings*.  
These embeddings capture semantic relationships between words in a continuous vector space, such that words appearing in similar contexts have similar representations.

For example, in a well-trained model, vectors for words like *king* and *queen* or *Paris* and *France* occupy nearby regions in the embedding space, allowing arithmetic-like operations such as:

\[
\text{vector("king")} - \text{vector("man")} + \text{vector("woman")} \approx \text{vector("queen")}
\]

The goal of Word2Vec is to learn these word vectors from large text corpora in such a way that words occurring in similar contexts have similar embeddings.

---

## **2. The Probabilistic Framing**

To achieve this, Word2Vec defines a **probabilistic model** of language.  
For each *center word* \( c \), the model aims to predict its *context words* \( o \) — i.e., the words that appear nearby in a sentence.

Mathematically, this is expressed as:

\[
P(o \mid c)
\]

which means the probability of observing the word \( o \) given the center word \( c \).  
The model’s task is to **maximize this conditional probability** across all observed word pairs in the training corpus.

---

## **3. Model Representation**

Each word in the vocabulary is represented by two vectors:

- \( v_c \): the **vector representation** when the word is a *center word*  
- \( u_o \): the **vector representation** when the word is a *context word*

Using these, Word2Vec defines the conditional probability as a **softmax function**:

\[
P(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V} \exp(u_w^\top v_c)}
\]

where \( V \) is the vocabulary size.  

The numerator measures the similarity between the context and center word, and the denominator normalizes this across all possible words in the vocabulary so that the probabilities sum to 1.

---

## **4. The Likelihood Function**

Given a large corpus containing many word pairs \((c, o)\), the **likelihood function** represents the probability of the entire dataset under the current model parameters:

\[
L = \prod_{(c,o) \in D} P(o \mid c)
\]

This likelihood indicates how well the model with its current parameters explains the observed data.  
However, multiplying many small probabilities can lead to **numerical instability** and make optimization difficult.

---

## **5. The Log-Likelihood Function**

To simplify computation, we take the **logarithm** of the likelihood function.  
The log operation converts products into sums:

\[
\log L = \sum_{(c,o) \in D} \log P(o \mid c)
\]

Maximizing the log-likelihood is equivalent to maximizing the likelihood itself because the logarithm is a monotonically increasing function.  
It also improves numerical stability and simplifies differentiation, which is crucial for gradient-based optimization.

---

## **6. The Objective Function**

The **objective function** is a general term for the mathematical expression we aim to optimize — either maximize or minimize — to make our model perform better.  
It quantifies *how well* our model represents the data.

In the case of **Maximum Likelihood Estimation (MLE)** used in Word2Vec, the **objective function** is the **log-likelihood**:

\[
J(\theta) = \sum_{(c,o) \in D} \log P(o \mid c)
\]

The parameters \( \theta \) (which include all word vectors \( u \) and \( v \)) are updated to **maximize** this objective.  
In practice, we often minimize the **negative log-likelihood**, which is mathematically equivalent:

\[
\text{Minimize: } -J(\theta) = -\sum_{(c,o) \in D} \log P(o \mid c)
\]

This objective function guides learning — by adjusting parameters to maximize log-likelihood, the model assigns higher probabilities to word pairs that truly occur in the corpus.

---

## **7. Model Training Process**

Training involves iteratively optimizing the objective function using **stochastic gradient descent (SGD)** or similar algorithms.

**Steps:**

1. **Initialize parameters** — assign small random values to all word vectors \( u \) and \( v \).  
2. **Compute the loss** — for each word pair \((c, o)\), calculate the negative log-probability \(-\log P(o \mid c)\).  
3. **Compute gradients** — determine how much each parameter contributes to the loss.  
4. **Update parameters** — adjust parameters in the direction that reduces loss:

   \[
   \theta \leftarrow \theta - \eta \nabla_\theta (-J(\theta))
   \]

   where \( \eta \) is the learning rate.

This process repeats over many epochs until convergence — when parameter updates no longer improve the objective significantly.

---

## **8. Computational Challenge and Approximations**

The denominator in the softmax function involves a sum over the entire vocabulary \( V \), which can be extremely large (millions of words).  
Computing this for every update is computationally expensive.

To address this, Word2Vec uses approximation techniques such as:

- **Hierarchical Softmax** — organizes words in a binary tree to reduce computation.  
- **Negative Sampling** — updates only a few randomly chosen “negative” samples at each training step.

These methods drastically reduce training time while maintaining good embedding quality.

---

## **9. Making Predictions**

After training, each word has a learned vector representation.  
These embeddings can be used in several ways:

- **Semantic similarity:** words with similar meanings have similar vectors (cosine similarity).  
- **Analogical reasoning:** e.g., *king – man + woman ≈ queen*.  
- **Feature representation:** word embeddings serve as inputs to downstream NLP models (e.g., sentiment analysis, translation).

The trained model doesn’t predict words directly — instead, it provides a **semantic embedding space** where vector proximity encodes linguistic relationships.

---

## **10. Summary**

- **Likelihood function:** measures how probable the observed data is, given model parameters.  
- **Log-likelihood function:** transforms products into sums, improving numerical stability and optimization.  
- **Objective function:** defines what the model optimizes (here, the log-likelihood).  
- **Training:** iterative parameter updates using gradient-based methods to maximize log-likelihood (or minimize negative log-likelihood).  
- **Prediction:** learned embeddings capture semantic meaning and are used in various downstream NLP tasks.

In essence, the entire Word2Vec training process revolves around the **objective function**, which translates linguistic patterns into a mathematical form that guides the model to learn meaningful word representations.

# Introduction to Neural Networks

> *Neural networks are not black boxes - they are stacks of simple mathematical functions that learn through gradient descent. Understanding the underlying calculus reveals their transparency: they are smooth landscapes of parameters where learning is the art of navigating downhill, guided by gradients and sculpted by non-linearities.*
>
>*The mathematics of learning, once seen as a curiosity of early neural research, now defines the architecture of modern intelligence systems - from image classifiers to conversational agents.* 



In previous sections we had looked at small “baby” neural networks and noted that each of those orange circles — the neurons — can be thought of as little *logistic regression* units.  

![Neural Network](/img/nn-1.png)

In traditional statistics or introductory machine learning courses, you typically deal with a *single* logistic regression: you manually define the input features, compute a weighted sum, and produce one decision output. Neural networks, however, stack many such logistic-like units together — creating cascades of transformations.  

The key insight is that, while we still define the desired *final* output through an objective or loss function, the layers in the middle are not manually designed. Instead, they *learn* to represent features that are useful for the downstream prediction task. These hidden layers self-organize to capture increasingly abstract representations.  

That’s the core magic of neural networks — they automatically learn intermediate representations rather than relying on handcrafted features. This ability to learn hierarchical representations is what makes them so much more powerful than traditional machine learning models.

---

## From Neurons to Matrices

While one could, in theory, wire neurons together arbitrarily — as biological brains do — artificial neural networks usually follow a *layered* architecture. Each layer takes as input the outputs of the previous layer, multiplies them by a weight matrix, adds a bias vector, and applies a non-linear activation function.  

Mathematically, we can describe this as:  

$$
\mathbf{z} = W \mathbf{x} + \mathbf{b}
$$

$$
\mathbf{h} = f(\mathbf{z})
$$

Here:

- $\mathbf{x}$ is the input vector  
- $W$ is the weight matrix  
- $\mathbf{b}$ is the bias vector  
- $f(\cdot)$ is the activation function applied elementwise  

This compact notation captures what each neuron does: it performs a weighted sum followed by a non-linear transformation. The activation function is essential — it introduces non-linearity and makes deep learning possible.

As an example, recall the small network we looked at earlier that predicted whether a word in the middle of a context window represented a *location*. The computation involved a matrix multiplication, a non-linearity, a dot product, and finally a sigmoid to produce a “yes” or “no” output.

---

## Activation Functions (*Why we need them?*)

Historically, neural networks began with a *threshold* activation — a neuron would “fire” (output 1) if its input exceeded a threshold $\theta$, and otherwise output 0. This was inspired by biological neurons and dates back to the 1940s.  

However, threshold units have a serious limitation: their outputs are flat (either 0 or 1), so their derivatives are zero almost everywhere. Without slopes, we cannot use *gradient-based optimization*, the foundation of modern neural network learning.  

To enable learning, we need functions that have *non-zero gradients*. Hence, smoother activation functions were introduced.

---

## Sigmoid and Tanh

The first widely used activation was the **sigmoid** (or logistic) function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

It outputs values between 0 and 1, and its derivative is simple and positive everywhere, which allows gradient-based learning. Sigmoids were particularly popular because they can be interpreted probabilistically.

However, sigmoid activations have two main drawbacks:

1. Their outputs are always positive, which causes activations to shift toward higher values (“biased saturation”).  
2. For large positive or negative inputs, the gradients become very small — a phenomenon known as *vanishing gradients*.

To mitigate this, people started using the **tanh** function:

$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

The tanh is essentially a *rescaled* sigmoid — it outputs values between -1 and 1 instead of 0 and 1. This centers activations around zero, which often leads to faster convergence.  

Mathematically, the relationship is:

$$
\tanh(x) = 2\sigma(2x) - 1
$$

Despite these benefits, both sigmoid and tanh require computing exponentials, which are computationally expensive and still suffer from gradient saturation for large $|x|$.

---

## ReLU and its Variants

To find simpler, faster, and more effective alternatives, researchers introduced *piecewise linear* activation functions.

The most influential is the **Rectified Linear Unit (ReLU)**:

$$
\text{ReLU}(x) = \max(0, x)
$$

ReLU outputs 0 for negative inputs and grows linearly for positive inputs. This simplicity has several benefits:

- It avoids saturation in the positive region (gradient = 1)  
- It’s computationally cheap  
- It allows fast and effective gradient propagation during backpropagation  

The downside is that neurons can “die” - once they enter the negative region, their gradient becomes zero, and they may stop updating. Still, empirically, networks with ReLU activations train faster and achieve better performance in many cases.  

To address the “dying ReLU” issue, several variants were proposed:

- **Leaky ReLU:** introduces a small slope (e.g., $0.01x$) in the negative region.  
- **Parametric ReLU (PReLU):** learns that negative slope during training.  
- **ELU (Exponential Linear Unit):** adds a smooth exponential tail on the negative side to maintain some gradient flow.

More recently, modern transformer-based architectures often use **Swish** and **GELU** activations.  

- **Swish:** $f(x) = x \cdot \sigma(x)$  
- **GELU:** $f(x) = x \cdot \Phi(x)$, where $\Phi(x)$ is the standard Gaussian CDF.

These newer functions behave roughly like ReLU for large $x$, but are smoother near zero, improving optimization stability and performance. GELU, in particular, is now the default in models like BERT and GPT.

---

## Why Non-Linearity is Essential?

So why do we even need non-linearities at all?  
The answer is *representational power*.

A matrix multiplication (plus bias) is an **affine transformation**, which is fundamentally linear. Composing multiple linear layers still gives you a linear function - you can collapse them into a single matrix. Thus, without non-linearities, stacking layers does not increase the expressive capacity of your model.

Non-linear activation functions break this constraint. They let neural networks represent complex, non-linear mappings - enabling them to approximate *any* continuous function, as guaranteed by the **Universal Approximation Theorem**.

Even though linear networks are theoretically interesting to study (for understanding learning dynamics), in practice, non-linear activations are indispensable for learning expressive models.

---

## Gradient-Based Learning and Backpropagation

With non-linearities in place, the next challenge is learning the parameters - the weights and biases. Neural networks learn by **gradient descent**, where we compute the derivative of the loss function with respect to each parameter and update in the direction that reduces the loss.

Mathematically, a simple gradient descent update rule is:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} J(\theta)
$$

Here:

- $\theta$ represents the model parameters  
- $J(\theta)$ is the loss function  
- $\eta$ is the learning rate  
- $\nabla_{\theta} J(\theta)$ is the gradient of the loss with respect to the parameters  

The symbol $\nabla$ (nabla) denotes the *gradient operator*, which gives the slope of the function in parameter space.  

In practice, we use **stochastic gradient descent (SGD)** - estimating the gradient on small random batches of data rather than the whole dataset at once.

The crucial computational step underlying all this is **backpropagation**. Backpropagation is simply the *systematic application of the chain rule* from calculus to efficiently compute gradients across many layers of a network. It automates what, in theory, you could derive by hand.

To perform this correctly, we rely on **matrix calculus** (derivatives of vector) and matrix-valued functions. The goal in this lecture (and in the upcoming assignment) is to understand both the manual math and the automated process behind backpropagation.

---

## Summary

Neural networks can be viewed as cascades of logistic regression units that learn hierarchical feature representations through gradient-based optimization. Their power comes from three key ideas:

1. **Layered composition of linear and non-linear functions** - allowing rich, compositional representations.  
2. **Non-linear activation functions** - introducing the ability to approximate complex mappings.  
3. **Gradient-based learning** - enabling optimization of parameters through backpropagation.

In the next part of the lecture, we’ll move from these conceptual foundations to the mechanics of computing gradients (both by hand and computationally) which is the mathematical backbone of how neural networks learn.

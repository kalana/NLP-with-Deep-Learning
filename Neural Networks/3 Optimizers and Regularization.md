# Optimizers and Regularization in Neural Networks

---

# Optimization

Every neural network is a _parametric function_ - a system of interconnected weights and biases that transform inputs into outputs.  To make the network “learn”, we must ***adjust these parameters*** so that its predictions become more accurate.

Every machine learning or deep learning model has a ***Loss Function*** - a mathematical expression that quantifies how far the model’s predictions are from the actual values. We measure how well the network performs using this loss function, denoted \( J(\theta) \), where \( \theta \) represents all model parameters.

Our goal is to find parameter values that **minimize** this loss.

Optimization, then, is the process of navigating through a ***high-dimensional landscape*** (the loss surface) to find a minimum point.

---

## Concept of the Gradient

The gradient is the **compass** that tells us which direction to move to decrease the loss most efficiently.  

Mathematically, the gradient \( \nabla_\theta J(\theta) \) is a vector containing the ***partial derivatives*** of the loss with respect to each parameter. If the gradient for a parameter is positive, decreasing that parameter reduces the loss; if it’s negative, increasing it does.  

> *Following the **negative gradient** is like walking downhill in a fog - you don’t see the valley, but you always move down the slope beneath your feet.*

---

## Gradient Descent - Core Idea

***Gradient Descent (GD)*** is the simplest and most fundamental optimization algorithm.

The gradient of the loss function, denoted:

\[
\nabla_\theta J(\theta)
\]

The update rule is:

\[
\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
\]

Here:
- \( \theta \): model parameters  
- \( \alpha \): learning rate (step size)  
- \( \nabla_\theta J(\theta) \): gradient of the loss function  

Each iteration takes a step proportional to the negative gradient.  

![Convex vs Non-Convex](/img/convex-vs-non-convex.png)

If the loss surface were a smooth bowl (convex), gradient descent would move us directly toward the bottom.  

But in deep learning, the surface is **non-convex** (filled with ridges, flat plateaus, and local minima) making the journey complex.



### Learning Rate (*Step Size of Change*)

The learning rate \( \alpha \) determines how big a step we take each time we update the parameters.

- A **small** learning rate leads to **slow but stable** convergence.  
- A **large** learning rate may overshoot the minimum or oscillate without converging (divergence).

Choosing the right learning rate is one of the most important practical aspects of training neural networks.  

Adaptive optimizers (like ***Adam***) automatically adjust learning rates per parameter to handle this.

---

## Stochastic Gradient Descent (SGD)

Computing the full gradient using the entire dataset is computationally expensive in large datasets.  ***Stochastic Gradient Descent*** (SGD) addresses this by using only a small `minibatch` of examples at each iteration.

\[
\theta \leftarrow \theta - \alpha \nabla_\theta J_{\text{minibatch}}(\theta)
\]

This makes training **faster** and **more scalable**, especially for large datasets.

The downside is that minibatch gradients are **noisy** approximations of the true gradient, causing oscillations - the updates may not point perfectly downhill each time.  

That’s where **momentum** helps.

---

## Momentum

In plain SGD, each update depends only on the current gradient. But gradients can fluctuate wildly - especially when the loss surface has steep, narrow valleys.

To stabilize learning, concept of ***momentum*** is introduced, which accumulates a running average of past gradients.

The update equations:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_\theta J_{\text{minibatch}}(\theta)
\]
\[
\theta_t = \theta_{t-1} - \alpha m_t
\]

Here:
- \( m_t \): moving average of gradients (like velocity)  
- \( \beta_1 \): momentum coefficient (typically 0.9)

So momentum smooths the updates: instead of reacting sharply to each noisy minibatch gradient, we move along the average direction.

Momentum gives the optimizer **inertia**, letting it move smoothly in consistent directions and reducing oscillation across gradients that point in opposite directions.  

Compared to vanilla SGD, momentum leads to ***lower variance***, ***faster convergence***, and smoother progress toward minima.

---

## Adaptive Learning Rate Methods

Different parameters may require different step sizes.  
Some weights experience large gradients, while others barely change.  

***Adaptive optimizers*** adjust the learning rate for each parameter individually based on gradient history.

Two key ideas:
1. Momentum-like tracking of gradients (first moment)
2. Scaling step sizes inversely by gradient magnitude (second moment)

> *The second option above (scaling step sizes) mean; the optimizer takes smaller steps for parameters whose gradients are large,
and larger steps for parameters whose gradients are small.*

This combination leads to methods like **RMSProp** and **Adam**.

---

## RMSProp (Root Mean Square Propagation)

**RMSProp** maintains a moving average of squared gradients to scale the learning rate adaptively.

The idea is to divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight

\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_\theta J_{\text{minibatch}}(\theta))^2
\]
\[
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t + \epsilon}} \nabla_\theta J_{\text{minibatch}}(\theta)
\]

- \( v_t \): rolling average of squared gradients  
- \( \beta_2 \): decay rate (typically 0.99)  
- \( \epsilon \): small constant to prevent division by zero  

RMSProp reduces step sizes for parameters with consistently large gradients and increases them for parameters with small gradients.  

This balancing effect helps the model learn more efficiently, especially in cases like recurrent networks or noisy objectives.

---

## Adam (Adaptive Moment Estimation)

**Adam** combines **Momentum** and **RMSProp** into one elegant algorithm.  

It keeps track of two moving averages:
1. \( m_t \): the first moment (mean of gradients)
2. \( v_t \): the second moment (mean of squared gradients)

The equations:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t \odot g_t)
\]
where \( g_t = \nabla_\theta J_{\text{minibatch}}(\theta_t) \).

Bias-corrected estimates:

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

Update rule:

\[
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

### Understanding the Roles:
- \( \beta_1 \) (~0.9): controls how much momentum (first moment) to retain  
- \( \beta_2 \) (~0.99): controls how much the squared gradient (second moment) influences scaling  
- \( \odot \): element-wise multiplication  
- \( \alpha \): base learning rate  

### Intuition:
- The **momentum term** smooths noisy gradients.  
- The **adaptive term** scales each parameter’s update based on the history of gradient magnitudes.  
- Parameters with **small average gradients** get **larger updates**, while those with **large gradients** are dampened.  

> *Adam is like a car with shock absorbers (momentum) and smart brakes (adaptive scaling) - smooth and stable.*

This dynamic adjustment makes Adam robust and efficient across diverse architectures, and it typically **converges faster** than plain SGD or momentum-based methods.

---

## AdamW (Small Fix to Adam)

Adam’s built-in weight decay interacts poorly with adaptive learning rates, weakening regularization.  
**AdamW** separates weight decay from the gradient update:

\[
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_{t-1}
\]

This simple change leads to **better generalization** and is now the **standard optimizer** for large-scale models such as **Transformers**, **BERT**, and **GPT** architectures.

---

# Regularization — Preventing Overfitting

Even with perfect optimization, a neural network can ***overfit*** - meaning it memorizes training data instead of learning general patterns.

Regularization methods reduce overfitting by constraining how much the model can adapt to noise in the data.

One of the most effective regularization methods in deep learning is ***Dropout***.

---

## Dropout (Randomly Forget to Learn Better)

### Problem:
Neural networks with many hidden units can co-adapt - certain neurons rely on the presence of others, creating fragile dependencies.  
This leads to poor generalization when new data is introduced.

### Idea:
Dropout combats this by **randomly “dropping out” (turning off)** some neurons during training.

![Dropout](/img/dropout.png)

That means, for each minibatch, some fraction \( p_{drop} \) of neurons in a layer are ignored - they neither contribute forward nor receive gradient updates backward.

### Mathematically:

\[
h_{\text{drop}} = \gamma (d \circ h)
\]

where:
- \( h \): vector of hidden activations  
- \( d \): dropout mask (vector of 0s and 1s)  
- \( \circ \): element-wise multiplication  
- \( \gamma \): scaling factor chosen so that \( \mathbb{E}[h_{\text{drop}}] = h \)

Each entry in \( d \) is:

\[
d_i = 
\begin{cases}
0, & \text{with probability } p_{drop} \\
1, & \text{with probability } 1 - p_{drop}
\end{cases}
\]

For example, if:

\[
h = [0.33, -1.18, 0.7, -1.8, 0.21], \quad p_{drop} = 0.6
\]

and a randomly generated mask:

\[
d = [1, 0, 0, 1, 0]
\]

then:

\[
h_{\text{drop}} = \gamma [0.33, 0, 0, -1.8, 0]
\]

Here \( \gamma = \frac{1}{1 - p_{drop}} = 2.5 \), ensuring the expected value of the activations remains constant.

### Intuition:
Dropout can be viewed as training **an ensemble of smaller networks** within the full network.  

Each subset of neurons learns to work independently, improving robustness and reducing overfitting.  

During inference (testing), dropout is **turned off**, and activations are scaled down by \( 1 - p_{drop} \) to maintain balance.

---

# Summary of Key Concepts

| Concept | What It Does | Key Equation / Mechanism | Intuition |
|----------|---------------|---------------------------|-----------|
| **Gradient Descent** | Basic optimization | \( \theta \leftarrow \theta - \alpha \nabla_\theta J(\theta) \) | Step downhill along gradient |
| **Learning Rate** | Controls step size | α (scalar) | Small α = slow, large α = unstable |
| **SGD** | Random minibatch updates | \( \nabla_\theta J_{\text{minibatch}}(\theta) \) | Faster but noisy |
| **Momentum** | Adds inertia | \( m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t \) | Smooths updates |
| **RMSProp** | Adapts step size per parameter | \( v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2 \) | Scales steps inversely to gradient magnitude |
| **Adam** | Combines momentum + adaptive scaling | \( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \) | Fast, stable, widely used |
| **AdamW** | Proper weight decay in Adam | Adds \( -\alpha \lambda \theta \) term | Better generalization |
| **Dropout** | Regularization via random neuron dropping | \( h_{\text{drop}} = \gamma (d \circ h) \) | Prevents co-adaptation, reduces overfitting |




Optimization and regularization are two sides of the same coin:  
- **Optimizers** teach a network *how to learn efficiently*.  
- **Regularizers** teach it *how to forget wisely*.

The interplay between the two is what enables deep learning models to generalize — to move beyond memorizing data toward capturing underlying structure.

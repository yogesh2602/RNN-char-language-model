# RNN-char-language-model

### # Food for thought 

**Question:** The hidden state is a fixed-size vector, yet the sequence can be arbitrarily long. What does this imply about the quality of memory the RNN maintains for very long sequences?

**Answer:** Think of the hidden state ($h_t$) as a single, hard-shell suitcase. In this project, that suitcase has exactly 64 slots (`hidden_size = 64`). 

If you go on a 3-day trip (a short sequence), your memories fit easily. But if you go on a 500-day trip (an arbitrarily long sequence), you cannot magically expand the suitcase. Every time you pack a new character, you have to either throw an old memory out or crush everything together until it's a wrinkled mess.

Because the hidden vector size is strictly fixed, the quality of memory degrades heavily over time, resulting in two major limitations:

* **Severe Short-Term Bias:** The RNN suffers from "short-term memory loss." It will vividly remember the last 5 to 10 characters it just read, but the memory of the first character of the paragraph will be completely diluted, mathematically overwritten, and forgotten.
* **The "Information Bottleneck":** The network is forced to compress a potentially infinite amount of historical text into a finite array of numbers. Because lossless compression is impossible here, the model learns to prioritize immediately useful, local context and drops the rest.

* ### Why Not Just Increase `hidden_size` to 256?

It is tempting to think that simply expanding the suitcase from 64 to 256 slots fixes the problem. While it *does* improve short-term performance, it fails to solve the core architectural flaw.

#### 1. The Short-Term Illusion
Bumping the hidden size to 256 gives you immediate, visible gains on the training loop:
* **Lower Loss:** The loss curve will drop significantly below the 1.75 baseline achieved with size 64.
* **Better Local Grammar:** The generated text will look much more coherent, spelling longer words correctly and capturing immediate sentence structures. 
* **More Capacity:** A 256-dimensional vector simply provides more capacity to temporarily store features.


---

### What happens if we replace `np.tanh(z)` with a linear activation (no squashing) or ReLU?

**Question:** We used `tanh` as the activation. What would happen if we used a linear activation (no squashing) instead? Try changing `np.tanh(z)` to just `z` in `rnn_step` and re-run training — what do you observe?

**Answer:** Removing the `tanh` squashing function breaks both the network's capacity to learn and its training stability. 

#### 1. Purely Linear Activation (`z`)
If you change `np.tanh(z)` to just `z`, the hidden state dynamics become entirely linear. No matter how many layers or time steps you unroll, a succession of linear operations collapses into a single linear transformation. The network loses the ability to learn complex, non-linear text patterns (like syntax, hierarchy, or open/close brackets) and becomes no more powerful than a basic linear regression model.

#### 2. The Trap of Unbounded Activations (like ReLU)
It is tempting to think an activation function like ReLU would work here since it introduces non-linearity. However, if a hidden unit value is positive under ReLU, it passes through untouched and gets multiplied by $W_{hh}$. In the next step, it gets multiplied by $W_{hh}$ again. 

Because there is no upper ceiling to compress the numbers back down, unbounded activations almost guarantee **Exploding Activations** in a vanilla RNN. The values compound exponentially loop after loop until the hidden states hit `NaN` and the network crashes.

#### 🎯 The Verdict
Vanilla RNNs are practically forced to use `tanh`. It provides the non-linearity required to learn complex relationships while acting as a strict mathematical ceiling (squashing values between -1 and 1) to survive the brutal, repetitive multiplication of the $W_{hh}$ loop.

  

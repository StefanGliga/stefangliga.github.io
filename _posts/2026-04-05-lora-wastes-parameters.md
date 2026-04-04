---
layout: post
title: "LoRA Wastes Parameters"
date: 2026-04-05 00:00:00 +0200
authors:
  - Stefan Gligorijevic
affiliations:
  - Independent Research
reading_time: 6
brief_for_practitioners: |-
  LoRA wastes a tiny fraction of it's parameters, this is fixable by hardcoding the top $r$ rows of the $A$ to identity.

  Practically, this could be implemented as:
  ```py
  y = x@W + (x[:r] + x@A) @ B # A shape [m-r, r] now
  ```
references:
  - id: hu2021lora
    text: "Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models."
    url: "https://arxiv.org/abs/2106.09685"
---

By now, LoRA has become a ubiquitous finetuning method. {% include cite.html id="hu2021lora" %} What is less well known, is that LoRA wastes parameters.

## LoRA refresher

At the core of LoRA is a simple idea. Express the target weights as the old weights plus a delta, then rewrite the delta in a form that is more convenient. In the case of LoRA specifically, delta is rewritten as a low rank factorization, evoking the form of a truncated SVD with the $\Sigma$ folded into $A$ or $B$ (with the important distinction that $A$ and $B$ are unrestricted, unlike $U$ and $V$ in SVD)

$$ W = W_0 + \Delta $$

$$ \Delta = AB \quad \quad A \in R^{m \times r}, B \in R^{r \times n} $$

## The waste

For a rank $r$, the number of parameters added by LoRA is $mr+nr$, or in the special case of rectangular matrix targets, $2mr$. Let's consider the extreme case of a rank equal to the matrix size for a square matrix. The total number of added parameters is $2 \times m \times m = 2m^2$, while expressing the update unfactorized takes only $m^2$. This is a hint that we're losing expressive power somewhere!

# The symmetry of LoRA
<p style="font-size:0.85em; color:#5a6b78; font-style:italic; margin-top:-0.2rem;">
No I am not actually suffering from LLM psychosis!
</p>

Observe that for any invertible matrix $Q$ of dimensions $r \times r$:

$$ 
\Delta_1 = A_1 B_1 = A_1 Q Q^{-1} B_1 = A_2 B_2 = \Delta_2 
\quad \quad \quad \quad
A_2 = A_1Q \quad 
B_2=B_1Q^{-1} 
$$

If you ask ChatGPT about this, it will start ranting and raving something about $\mathrm{GL}(r)$ being the gauge group of LoRA(it seems to love bringing up symmetry and gauge even when completely inappropriate).  
Or put simply, we do not care about the exact values in $A$ or $B$, we only care about the resulting operator defined by $AB$, and furthermore, we know of a whole family of transformations on $A$ and $B$ jointly that leave $AB$ unchanged. Another way to phrase the same realization is that the problem of factorizing a matrix into two (general) matrices is underdetermined. The solution is to introduce additional constraints on parameters of $A$ and $B$, precisely $r^2$ of them.  

## Symmetry breaking

As we have full freedom to pick whatever form of constrainsts we want, in the interest of being friendly towards the hardware, I propose to hardcode the top $r$ rows of the $A$ matrix to the identity matrix. I will call this IBA-LoRA, "Identity Block in A LoRA".

Formally, define $A$ as:

$$
A = \begin{bmatrix}
I_{r \times r} \\ A'_{(m-r) \times r}
\end{bmatrix}
$$

This form also gives us a convenient way to implement this without a custom kernel, as the output of the LoRA module can be expressed as:

$$ y = x@W + (x[:r] + x@A') @ B $$

<details class="article-dropdown">
<summary markdown="0">Alternative forms of constraints to be applied</summary>
<div markdown="1">

In the interest of imposing the least structure possible, we could pick a random subset of parameters from $A$ and $B$ and hardcode them to random values from some distribution. However, this could be quite hardware unfriendly as it might require modified address generation and data reshuffling within the matmul kernels.  
If instead we aim to harken to forms familiar from linear algebra, we could impose a diagonal matrix structure(in this case, trapezoidal structure specifically, as matrices are non square). This is best expressed as jagged arrays, which again are not ideal for hardware where the primitive is a tile matmul.  

A whole other family of alternatives is to impose constraints in the forms of equations on multiple parameters at the same time. The canonical example would be imposing orthogonality on rows of $A$.  
That is $a_{0,0}a_{1,0}+...+a_{0,r-1}a_{1,r-1}==0; \quad a_{0,0}a_{2,0}+...+a_{0,r-1}a_{2,r-1}==0; ... $ and so forth...  
This could be expressed using Householder reflectors, which can be hardware unfriendly; or using a full matrix together with an optimizer that enforces orthogonality, but that reintroduces the parameter waste.  

One observation is that if we impose lower diagonal structure on $A$ and upper diagonal structure on $B$(with diagonal hardcoded to ones), we recover the form of the well known **LU** decomposition.  
Likewise, if we impose orthonormality on $A$ and upper diagonal structure on $B$, we recover the **QR** decomposition. Reshuffling also gives us the **LQ**.  
Lastly, imposing orthogonality on $A$ and orthonormality on $B$ gives us the **SVD**, albeit with the $\Sigma$ matrix folded into $A$.

</div>
</details>

# Practical results

To sanity-check that fixing the redundant degrees of freedom does not damage optimization, I ran three overnight fine-tunes comparing standard LoRA against IBA-LoRA on Gemma 2 2B in 4-bit across 3 different seeds. The pattern is consistent: train loss follows the same general trajectory and the delta between IBA-LoRA and baseline is within run to run variance; and surprisingly eval loss is better for IBA-LoRA by a small but non negligible margin, while still following the overall shape of the baseline curve.

<details class="article-dropdown">
<summary markdown="0">Details of the experimental setup</summary>
<div markdown="1">
All experiments were done on a single RTX 4060Ti with a vibe coded training script. Hyperparameters were as follows:
- Model: Gemma 2 2B
- Quantization: Bitsandbytes 4-bit, nf4 with double quantization
- Adapter rank: 16, alpha: 16
- Adapters applied to all linear layers
- "paged_adamw_8bit" optimizer
- Learning rate 3e-4 (decided by random guessing, not tuned at all)
- Batch size: 3
- Sequence length: 4096
</div>
</details>

<div class="image-grid" markdown="1">
<div class="image-tile" markdown="1">
Seed 41, training loss.

![Training loss comparison for seed 41 between identity-block LoRA and standard LoRA.](/assets/images/posts/lora-wastes-parameters/seed-41-train-loss.png)
</div>
<div class="image-tile" markdown="1">
Seed 42, training loss.

![Training loss comparison for seed 42 between identity-block LoRA and standard LoRA.](/assets/images/posts/lora-wastes-parameters/seed-42-train-loss.png)
</div>
<div class="image-tile" markdown="1">
Seed 43, training loss.

![Training loss comparison for seed 43 between identity-block LoRA and standard LoRA.](/assets/images/posts/lora-wastes-parameters/seed-43-train-loss.png)
</div>
<div class="image-tile" markdown="1">
Seed 41, evaluation loss.

![Evaluation loss comparison for seed 41 between identity-block LoRA and standard LoRA.](/assets/images/posts/lora-wastes-parameters/seed-41-eval-loss.png)
</div>
<div class="image-tile" markdown="1">
Seed 42, evaluation loss.

![Evaluation loss comparison for seed 42 between identity-block LoRA and standard LoRA.](/assets/images/posts/lora-wastes-parameters/seed-42-eval-loss.png)
</div>
<div class="image-tile" markdown="1">
Seed 43, evaluation loss.

![Evaluation loss comparison for seed 43 between identity-block LoRA and standard LoRA.](/assets/images/posts/lora-wastes-parameters/seed-43-eval-loss.png)
</div>
</div>

# Final comments

This work is mostly of theoretical interest, as the savings are tiny. Rough math suggests that in the scenario I tested above, the adapter parameter saving is ~0.2%, or equivalently a total parameter saving of 0.001%. Even at ranks as high as 256, the saving is only ~3.6% of the adapter parameters, or ~0.4% of the total parameters.

Low rank factorizations are also used elsewhere in deep learning, for example in attention, the $W_q$ and $W_k$ matrices can be considered a low rank factorization of a single matrix $W_{qk}$, and the same goes for $W_v$ and $W_o$. And there are further savings to be made inside Multihead Latent Attention and LatentMoE.

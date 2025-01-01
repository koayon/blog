---
layout: post
title: "Shazeer Typing"
stub: shazeer-typing
tags: machine-learning pytorch
toc: true
---

### Tensor Naming for Sanity and Clarity

<div align="center">
  <figure>
    <img src="/blog/images/shazeer-typing/karpathy_tweet.png" width="800" alt="Shazeer typing is good for your skin">
    </figure>
</div>

<br>

Few people have made as big an impact on language modelling as Noam Shazeer.

Shazeer invented MoEs, Multihead Attention, Multiquery Attention,
Tensor-Parallel LLM Training, SwiGLU and co-invented the Transformer.
In 2021, after Google's delay to ship LLMs,
Shazeer left Google to found Character.AI[^google].
This ended a string of insightful research papers and since joining Character, Shazeer has released only 4 research blog posts.
Three of the posts are on the Character blog [here](https://research.character.ai/) about efficient inference and prompting.
But the last is a curious tiny post on what he calls [Shape Suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd) which
we call as `Shazeer typing`.
The post is only 114 words and a single code block, but I've found it
remarkably useful for writing better PyTorch code.

Let's get into it.

[^google]: Shazeer is now back at Google to lead the Gemini team following a $2.7B Inflection-style acqui-hire. It's not _quite_ true that Google spent $2.7B to hire one researcher, but it's not _not_ true 🤷


## Core Idea

### The Problem

When writing ML code,
we often need to understand what each tensor and its dimensions represent
(as well as the size of the dimensions) at a glance.
For example, if a matrix M represents a linear transformation from space X to space Y,
that's immediately useful information.

Other solutions to this problem of communicating shapes exist:
`jaxtyping` checks tensor shapes effectively
but it's fairly verbose and you only see the shape when a tensor is introduced.

### The Solution

The **Shazeer typing** system is simpler and more lightweight:

- Designate a system of single-letter names for logical dimensions, e.g. `B` for batch size, `S` for sequence length, etc., and document it somewhere in your file/project/codebase.
- When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. tokens_BS represents a two-dimensional tensor with batch_size and seq_length dimensions.

We can now write an MLP (in pseudocode) as:

```python
""" Example MLP torch pseudocode code with Shazeer typing.
For illustration purposes only.

Dimension key:

B: batch_size
S: seq_len (sequence length)
N: neuron_dim (sometimes called model dimension, d_model or embedding_dim)
V: vocab_size (vocabulary size)
F: feature_dim (feed-forward subnetwork hidden size)
"""

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.W_in_NF = nn.Parameter(t.zeros(out_dim, in_dim))
        self.W_out_FN = nn.Parameter(t.zeros(in_dim, out_dim))

        self.init_params()

    def forward(self, x_BSN: t.Tensor) -> t.Tensor:
        y_BSF = x_BSN @ self.W_in_NF
        y_BSF = t.gelu(y_BSF)
        out_BSN = y_BSF @ self.W_out_FN

        return out_BSN
```

From this, we can immediately see that we won't have any share errors:
all the shapes match up for our matrix multiplications.
And it's pretty clear what each tensor represents.

<br>

Shazeer typing is an excellent lesson in following the [Zen of Python](https://peps.python.org/pep-0020/). It tracks with:

> Explicit is better than implicit.
>
> Simple is better than complex.
>
> Readability counts.
>
> Practicality beats purity.
>
> In the face of ambiguity, refuse the temptation to guess.
> There should be one-- and preferably only one --obvious way to do it.
>
> If the implementation is easy to explain, it may be a good idea.

We also get all the benefits of using type signatures for clarity without the headache of being ultra precise with types.
In Python, types are for communication with humans, not for communicating with the compiler[^compiler], so it's better to optimise for readability.

[^compiler]: The compiler ignores your type hints anyways.

## Extensions

We now further extend the original Shazeer typing approach.

### Output types

In the above MLP module,
we can look at the type signature of the forward function
and immediately infer the input argument types.
However, it's harder to infer the output types at a glance as we don't use Shazeer typing for the output type.
This is unfortunate.
If this becomes unclear in your code,
I've found the best solution is to use Shazeer typing within your functions and input arguments and
jaxtyping for output types.

### Data Types

Data types are also important for understanding tensor code.
For example, you might have a tensor which is made up of boolean or integer values.
Here, I recommend including the datatype before the shape suffix
e.g. `x_Int_BSN` or `z_Bool_FN`.
Where the data type isn't specified, we assume float values.

Note that for code where grokking quantisation is important you can also
use this approach for quantisation levels.
For example `x_8_FN` can signify that the value is an 8-bit rather than 16-bit float.

### Singleton tensors

For 1D tensors (e.g. loss, summary statistics etc),
leave the shape suffix blank, e.g. `loss: t.Tensor`.

### Rearranges

You can use lowercase shape suffixes to signify a reshaped tensor. For example, using einops (covered [here](https://www.kolaayonrinde.com/blog/2024/01/08/einops.html)) we can have:

```python
x_BsN = rearrange(x_BSN, "batch seq_len neuron_dim -> (batch seq_len) neuron_dim")
```

The lowercase `s` indicates that we can reshape this back when needed and signals that the first dimension should be of size B*S.

## Conclusion

That's all folks! I've found using Shazeer typing means that I ~never have shape errors,
can very quickly grok code and understand sensible transformations,
can keep my tensor code clean and
am able to easily give additional context to Copilot to stop LLM errors.
I highly recommend adopting this practice for your ML code.

<br>
<br>

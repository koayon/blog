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

There aren't very many people as accomplished in language modelling as Noam Shazeer.

Shazeer invented MoEs, Multihead Attention, Multiquery Attention,
Tensor-Parallel LLM Training, SwiGLU and co-invented the Transformer.
In 2021, after Google's slowness to release LLM technology,
Shazeer left Google to found Character.AI[^google].
This ended a string of incredibly insight research papers and since joining Character Shazeer has released only 4 blog posts about research.
Three are on the Character blog [here](https://research.character.ai/) about efficient inference and prompting.
But the last is a curious tiny post on what he calls [Shape Suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd) and
we refer to as `Shazeer typing`.
The post is 114 words and a single code block but I've found it
remarkably useful in writing better PyTorch code.

Let's get into it.

[^google]: Shazeer is now back at Google to lead the Gemini team following a $2.7B Inflection-style acqui-hire.
It's not _quite_ true that Google spent $2.7B to hire one researcher,
but it's not _not_ true 🤷


## Core Idea

### The Problem

When writing ML code, we need to understand what each tensor represents, what each dimensions represents and the shapes of the dimensions. For example if some matrix represents a linear transformation from space X to space Y, then this is useful information to know.

There have been some other more heavyweight approaches to this: I used to use jaxtyping which checks tensor shapes but you only see the shape when a tensor is introduced and jaxtyping is fairly verbose.

### The Solution

The **Shazeer typing** system is much simpler and more lightweight:

- Designate a system of single-letter names for logical dimensions, e.g. `B` for batch size, `S` for sequence length, etc., and document it somewhere in your file/project/codebase
- When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. tokens_BS represents a two-dimensional tensor with batch and seq_length dimensions.

You can then write an MLP as (in pseudocode):

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
        self.W_out_FN =  nn.Parameter(t.zeros(in_dim, out_dim))

        self.init_params()

    def forward(self, x_BSN: t.Tensor) -> t.Tensor:
        y_BSF = x_BSN @ self.W_in_NF
        y_BSF = t.gelu(y_BSF)
        out_BSN = y_BSF @ self.W_out_FN

        return out_BSN
```

Note that from just looking at this we can immediately see that we won't have any share errors:
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
In Python types are for communication with humans not for communicating with the compiler[^compiler], so it's better to optimise for readability.

[^compiler]: The compiler ignores your type hints anyways.

## Extensions

We now move to extending the original Shazeer typing approach
to more settings.

### Output types

Note that in the above MLP module we can look at the type signature of the forward function
and immediately infer the input argument types.
However it's a little more difficult to infer the output types at a glance as we don't use Shazeer typing for the output.
This is unfortunate and I've found the best way around this if it's really unclear
is to use Shazeer typing within your functions and for arguments and jaxtyping for output types.

### Data Types

Another important piece of information for understanding tensor code is the data types.
For example you might have a tensor which is made up of boolean or integer values.
Here I recommend including the datatype before the shape suffix
e.g. `x_Int_BSN` or `z_Bool_FN`.
Where the data type isn't specified we assume float values.

Note that for code where grokking quantisation is important you can also
use this for tracking which quantisation level you're at e.g.
`x_8_FN` can signify that the value is an 8bit rather than 16bit float.

### Singleton tensors

For tensors which are 1 dimensional (e.g. loss, summary statistics etc),
leave the shape suffix blank e.g. `loss: t.Tensor`.

### Rearranges

If you reshape some tensor you can use lowercase shape suffixes to signify this change. For example, using einops (covered [here](https://www.kolaayonrinde.com/blog/2024/01/08/einops.html)) we can have:

```python
x_BsN = rearrange(x_BSN, "batch seq_len neuron_dim -> (batch seq_len) neuron_dim")
```

The lower case `s` tells us that we can reshape this back when needed and signals that the size of the first dimension should be B*S.

## Conclusion

That's all folks! I've found using Shazeer typing means that I ~never have shape errors,
can very quickly grok code and understand sensible transformations,
can keep my tensor code clean and
am able to easily give additional context to Copilot to stop LLM errors.
I highly recommend adopting this practise for your ML code.

<br>
<br>

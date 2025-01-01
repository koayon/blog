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

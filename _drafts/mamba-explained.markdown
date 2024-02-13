---
layout: post
title: "Mamba Explained"
stub: mamba
tags: machine-learning state-space-models mamba ssm
toc: true
---

### The State Space Model taking on Transformers

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/c28c819e-18bd-43b4-9ce6-4f6e2b746c66-RackMultipart20240210-150-g5wqqs.png)

Right now, AI is eating the world.

And by AI, I mean Transformers. All the big breakthroughs in AI over the last
few years are due to Transformers.

**Mamba**, however, is one of an alternative class of models called **State
Space Models** (**SSMs**). Importantly, for the first time, Mamba promises
similar performance (and crucially similar
[_scaling laws_](https://arxiv.org/pdf/2203.15556.pdf)) as the Transformer
whilst being feasible at long sequence lengths (say 1 million tokens). We
achieve this long context by removing the “quadratic bottleneck” in the
Attention Mechanism. It also runs _fast_ - like up to 5x faster than Transformer
fast \[footnote: see Figure 8 in the Mamba paper\].

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/6d2eadc9-af92-4228-af7a-da42cc691617-RackMultipart20240211-159-fp1m2s.png)

Here we'll discuss:

- The advantages (and disadvantages) of Mamba (🐍) vs Transformers (🤖),
- Analogies and intuitions for thinking about Mamba, and
- What Mamba means for Interpretability, AI Safety and Applications.

## Problems with Transformers - Maybe Attention _Isn’t_ All You Need

We’re very much in the Transformer era of history. ML used to be about detecting
cats and dogs. Now, with Transformers, we’re
[generating human-like poetry](https://openai.com/research/gpt-4),
[coding better than the median competitive programmer](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf),
and
[solving the protein folding problem](https://www.nature.com/articles/s41586-021-03819-2).

But Transformers have one core problem. In a transformer, every token can look
back at every previous token when making predictions. So, we cache detailed
information about each token in the so-called KV cache.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/83c83d9a-3b1a-4240-bad4-f3766c3ab482-RackMultipart20240210-150-v31nde.png)

This pairwise communication means a forward pass is O(n²) time complexity in
training (the dreaded “quadratic bottleneck”) and each new token generated
autoregressively takes O(n) time. That is to say, as the context gets larger,
the model gets _slower_.

To add insult to injury, storing this KV cache to compute new tokens efficiently
requires O(n) space. The fateful CUDA OOM error looms large as the memory
footprint balloons. If space were the only issue, we might just add more GPUs
but with latency growing quadratically, perhaps not.

On the margin, we can mitigate the quadratic bottleneck with techniques like
sliding window attention or clever CUDA optimisations like FlashAttention. But
ultimately, for super long context windows (like a chatbot which remembers every
conversation you’ve shared), we need a different approach.

---

Fundamentally, all good ML architecture backbones have components for two
important operations:

1. **Computation** _within_ a token
2. **Communication** _between_ tokens

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/6893c3cc-9c27-41d9-8f4f-0234feacc4c0-RackMultipart20240210-96-tmrp3i.png)

In transformers, this is MLPs (computation) and Attention (communication). We
improve transformers by optimising these two operations (and scaling up with
massive compute).

We would like to replace the Attention component (footnote: more specifically
the scaled dot-product Attention popularised by Transformers) with some other
method for communicating between tokens. **Mamba** uses the (Control Theory
inspired) **SSM** for Communication and keeps MLP-style projections for
Computation.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/414d6ce7-4cf9-4489-b586-a2afb3466d82-RackMultipart20240210-139-5ze4ks.png)

Like a Transformer made up of stacked transformer blocks, Mamba is made up of
stacked Mamba blocks as above.

We would like to understand and motivate the choice of the SSM sequence
transformation.

## Motivating Mamba - A Throwback to Temple Run

Imagine we’re building a Temple Run agent \[footnote: for people who don’t see
Temple Run as the cultural cornerstone it is 🤣 Temple Run was an iPhone game
from 2011 similar to Subway Surfer\]. It chooses if the runner should move left
or right at any time.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/b710cce8-aa5f-486c-ab57-c27e0374b462-RackMultipart20240210-115-3n90zs.png)

To successfully pick the correct direction, we need information about our
surroundings. Let’s call the collection of relevant information the `state`.
Here the state likely includes your current position and velocity, the position
of the nearest obstacle, weather conditions, etc.

> Claim 1: if you know the current state of the world and how the world is
> evolving, then you can use this to determine the direction to move.

Note that you don’t need to look at the whole screen all the time. You can
figure out what will happen to most of the screen by noting that as you run, the
obstacles move down the screen. You only need to look at the top of the screen
to understand the new information and then simulate the rest.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/d14b2b90-72eb-4418-97fb-f2e56cc10ca9-RackMultipart20240210-105-erd1of.png)

This lends itself to a natural formulation. Let h be the hidden state, relevant
knowledge about the world. Also let x be the input, the observation that you get
each time. h’ then represents the derivative of the hidden state, i.e. how the
state is evolving. We’re trying to predict y, the optimal next move (right or
left).

Now, Claim 1 states that from the hidden state h, h’, and the new observation x,
you can figure out y.

More concretely, h, the state, can be represented as a differential equation (Eq
1a):

\\( h’(t) = Ah(t) + Bx(t) \\)

Knowing h allows you to determine your next move y, by some function (Eq 1b):

\\( y(t) = Ch(t) + Dx(t)\\)

The system evolves as a function of the current state and new observations. A
small new observation is enough because we can determine most of the state by
applying known state dynamics to the previous state. That is, most of the screen
isn’t new, it’s just the natural downward movement of the previous state. Fully
knowing the state would allow us to pick the best next move, y.

You can learn a lot about the system dynamics by observing the top of the screen
\- if it's moving faster, we can infer the whole screen is and the game is
speeding up \[footnote: here we assume the environment is sufficiently smooth\].
In this way, even if we start off knowing nothing about the game except our
limited observation, pretty soon we could understand the whole screen.

### What’s the State?

Here, **state** refers to the variables that, when combined with the input
variables, fully determine the future system behaviour. In theory, once we have
the state, there's nothing else we need to know about the past to predict the
future. With this choice of state, the system is converted to a **Markov
Decision Process**. Ideally, the state is a fairly small amount of information
which captures the essential properties of the system. That is, **the state is a
compression of the past**.

\[Footnote: One pretty important constraint for this to be efficient is that we
don’t allow the individual elements of the state vector to interact with each
other directly. We’ll use a combination of the state dimensions to determine the
output but we don’t e.g. allow the velocity of the runner and the direction of
the closest obstacle (or whatever else was in our state) to directly interact.
This helps with efficient computation and we achieve this practically by
constraining A to be a diagonal matrix. \]

## Discretisation - How To Deal With Living in a Quantised World

Okay, great! So, given some state and input observation, we have an
autoregressive-style system to determine the next action. Amazing!

In practice though, there’s a little snag here. We’re modelling time as
continuous here. But in real life, we get new inputs and take new actions at
discrete time steps \[footnote: concretely consider the case of Language
Models - each token is a discrete step\].

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/6e65b3cc-1fe1-4591-a06a-fee091977f30-RackMultipart20240211-159-1nxi41.png)

We would like to convert this _continuous-time differential equation_ into a
_discrete-time difference equation_. This conversion process is known as
`discretisation`. It turns out there are many ways to do this in the literature.
Mamba uses the [Zero-Order Hold](https://en.wikipedia.org/wiki/Zero-order_hold)
(ZOH) discretisation described here \[Footnote: ZOH also has nice properties for
the initialisations - we want A_bar to be close to the identity so that the
state can be mostly maintained from timestep to timestep if desired. ZOH gives
A_bar as an exponential so any diagonal element initialisations close to zero
give values close to 1\]. To give an idea of what’s happening morally, consider
a naive first-order approximation \[footnote: this is known as the Euler
discretisation in the literature\].

From Equation 1a, we have

\\( h’(t) = Ah(t) + Bx(t) \\)

And for small ∆,

\\( h’(t) \\approx \\frac{h(t+\\Delta) - h(t)}{\\Delta} \\)

by the definition of the derivative.

We let \\( h_t = h(t)\\ \\textup{and} \\ h\_{t+1} = h(t + \\Delta) \\)and
substitute into Equation 1a giving:

\\( h\_{t+1} - h_t \\approx \\Delta (Ah_t + Bx_t) \\)

\\( \\Rightarrow h\_{t+1} \\approx (I + \\Delta A)h_t + (\\Delta B)x_t \\)

Hence, after renaming the coefficients and relabelling indices, we have the
discrete representations:

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/fcb617fe-b48e-45da-8def-ce5707a5854d-RackMultipart20240211-189-7lvszg.png)

If you've ever looked at an RNN before \[footnote: it's wild to note that some
readers might not have, we're so far into the age of Attention that RNNs have
been forgotten!\] and this feels familiar - trust your instincts:

> We have some input x, which is combined with the previous hidden state by some
> transform to give the new hidden state. Then we use the hidden state to
> calculate the output at each time step.

## Understanding the SSM Matrices

Now, we can interpret the A, B, C, D matrices more intuitively:

- A is the transition state matrix. It shows how you transition the current
  state into the next state. It asks "How should I forget the less relevant
  parts of the state over time?”
- B is mapping the new input into the state, asking "What part of my new input
  should I remember?”
- C is mapping the state to the output of the SSM. It asks, “How can I use the
  state to make a good next prediction?”
- D is how the new input passes through to the output. It's a kind of modified
  skip connection that asks “How can I use the new input in my prediction?”

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/0b68dc7a-d62e-447b-835b-d36407b841f8-RackMultipart20240211-145-qxx4h0.png)

Additionally, ∆ has a nice interpretation - it's the step size, or what we might
call the "linger time" or the “dwell time”. For large ∆, you focus more on that
token; for small ∆, you skip past the token immediately and don't include it
much in the next state.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/09997a3d-027f-4da4-a3f1-3627236eb4ab-RackMultipart20240210-150-2ql00f.png)

And that’s it! That’s the SSM, our ~drop-in replacement for Attention
(Communication) in the Mamba block. The Computation in the Mamba architecture
comes from regular linear projections, non-linearities, and local convolutions -
the regular ML building blocks we know and love!

Okay great, that’s the theory - but does this work? Well…

## Effectiveness vs Efficiency: Attention is Focus, Selectivity is Prioritisation

At WWDC ‘97, Steve Jobs famously noted that
“[focusing is about saying no](https://www.youtube.com/watch?v=H8eP99neOVs&t=98s)”.
Focus is ruthless prioritisation. It’s common to think about Attention
_positively_ as choosing what to _notice_. In the Steve Jobs sense, we might
instead frame Attention _negatively_ as choosing what to _discard_.

There’s a classic intuition pump in Machine Learning known as the
[Cocktail Party Problem](https://ieeexplore.ieee.org/document/8555495)
\[footnote: non-alcoholic options also available!\]. Imagine a party with dozens
of simultaneous loud conversations:

Question:

> How do we recognise what one person is saying when others are talking at the
> same time? \[Footnote: especially as all voices roughly occupy the same space
> on the audio frequency spectrum! Intuitively this seems really hard!\]

Answer:

> The brain solves this problem by focusing your “attention” on a particular
> stimulus _and hence_ drowning out all other sounds as much as possible.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/fea8e536-e6d4-4dc4-85c2-cca893008cb8-RackMultipart20240210-139-y6mn1v.png)

---

Transformers use dot-product attention to focus in on the most relevant tokens.
The reason attention is so great is that you have the potential to look back at
everything that ever happened in its context. This is like photographic memory
when done right. \[footnote: note that photographic memory doesn’t necessarily
imply perfect inferences from that memory!\]

Transformers (🤖) are extremely **effective**. But they aren’t very
**efficient**. They store everything from the past so that they can look back at
tokens with theoretically perfect recall.

Traditional RNNs (🔁) are the opposite - they forget a lot, only recalling a
small amount in their hidden state and discarding the rest. They are very
**efficient** - their state is small. Yet they are less effective as discarded
information cannot be recovered.

We’d like something closer to the Pareto frontier of the
effectiveness/efficiency tradeoff. Something that’s more effective than
traditional RNNs and more efficient than transformers.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/b9f1ae8d-4cc7-47aa-90a0-1d86fdf53b2c-RackMultipart20240211-189-l8ar6g.png)

SSMs are as **efficient** as RNNs, but we might wonder how **effective** they
are. After all, it seems like they would have a hard time discarding only
_unnecessary_ information and keeping everything relevant. Each token is being
processed the same way, applying the same A and B matrices as if in a factory
assembly line for tokens. We would like the forgetting and remembering matrices
(A and B respectively) to vary and dynamically adapt to inputs.

### The Selection Mechanism

**Selectivity** allows each token to be transformed into the state in a way that
is unique to its own needs. Selectivity is what takes us from vanilla SSM models
(applying the same A (forgetting) and B (remembering) matrices to every input)
to Mamba, the **_Selective_** _State Space Model_.

In regular SSMs, A, B, C and D are learned matrices - that is \\( A = A\_\\theta
\\) etc. (where theta represents the learned parameters)

With the Selection Mechanism in Mamba, A, B, C and D are also functions of x.
That is \\( A = A\_\\theta(x) \\) etc.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/f2854ec9-ec6e-4d6d-8a68-5cac9c5a9a46-RackMultipart20240210-125-qihuy6.png)

Making A and B functions of x allows us to get the best of both worlds:

- We're selective about what we include in the state, which improves
  **effectiveness** vs traditional SSMs.
- Yet, since the state size is bounded, we improve on _efficiency_ relative to
  the Transformer. We have O(1), not O(n) space and O(n) not O(n²) time
  requirements.

The Mamba paper authors write:

> The efficiency vs. effectiveness tradeoff of sequence models is characterized
> by how well they compress their state: efficient models must have a small
> state, while effective models must have a state that contains all necessary
> information from the context. In turn, we propose that a fundamental principle
> for building sequence models is selectivity: or the context-aware ability to
> focus on or filter out inputs into a sequential state. In particular, a
> selection mechanism controls how information propagates or interacts along the
> sequence dimension.

---

Humans (obviously) don’t have photographic memory for everything they experience
within a lifetime - or even within a day! There's just way too much information
to retain it all. Subconsciously, we select what to remember by choosing to
forget, throwing away most information as we encounter it. Transformers (🤖)
decide what to focus on at **recall time**. Humans (🧑) also decide what to
throw away at **memory-making time**. Humans filter out information early and
often.

If you had infinite capacity for memorisation, it’s clear the transformer
approach is better than the human approach - it truly is more effective. But
it’s less efficient - transformers have to store so much information about the
past that might not be relevant. Transformers (🤖) only decide what’s relevant
at **recall time**. The innovation of Mamba (🐍) is allowing the model better
ways of forgetting earlier - it’s focusing by choosing what to _discard_ using
**Selectivity**, throwing away less relevant information at **memory-making
time**.

\[Footnote: to be clear, if you have a short sequence, then a transformer should
theoretically be a better approach. If you _can_ store the whole context, then
why not!? If you have enough memory for a high-resolution image, why compress it
into a JPEG? But Mamba-style architectures are likely to hugely outperform with
long-range sequences.\]

### The Problems of Selectivity

Applying the Selection Mechanism does have its gotchas though. Non-selective
SSMs (i.e. A,B not dependent on x) are fast to compute in training. This is
because the component of \\( y_t \\ \\textup{which depends on} \\ x_i \\) can be
expressed as a linear map, i.e. a single matrix that can be precomputed!

For example (ignoring the D component, the skip connection):

\\( y_2 = CBx_2 + CABx_1 + CAABx_0 \\)

If you’re paying attention, you might spot something even better here - this
expression can be written as a convolution. Hence we can apply the Fast Fourier
Transform and the Convolution Theorem to compute this _very_ efficiently on
hardware as in Equation 3 below.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/13ebc206-21a3-4735-9953-cdbceedbf76a-RackMultipart20240210-125-kljunn.png)

Unfortunately, with the Selection Mechanism, we lose the convolutional form.
Much attention is given to making Mamba efficient on modern GPU hardware using
similar hardware optimisation tricks to Tri Dao’s Flash Attention.\[Footnote:
More details are available for engineers interested in CUDA programming -
[Tri’s talk](https://www.youtube.com/watch?v=foG0ebzuw34&list=PLDEUW02OCkqGFMLHEpET24ArjE0By8JwS&index=9&pp=gAQBiAQB),
Mamba paper section **3.3.2**, and the
[official CUDA code](https://github.com/state-spaces/mamba/tree/main/csrc/selective_scan)
are good resources for understanding the Hardware-Aware Scan\] With the hardware
optimisations, Mamba is able to run faster than comparably sized Transformers.

### Machine Learning for Political Economists - How Large Should The State Be?

The Mamba authors write, “the efficiency vs. effectiveness tradeoff of sequence
models is characterised by how well they compress their state”. In other words,
like in political economy \[footnote: or in Object Oriented Programming\], the
fundamental question is how you manage your state.

> 🔁 **Traditional RNNs are anarchic** - they have a small, minimal state. The
> size of the state is bounded. The compression of state is poor.

> **🤖 Transformers are communist** - they have a maximally large state. The
> "state" is just a cache of the entire history with no compression. Every
> context token is treated equally until recall time.

> **🐍 Mamba has a compressed state,** but it’s selective about what goes in.
> Mamba says we can get away with a small state if the state is well focused and
> effective. \[footnote: implications to actual Political Economy are left to
> the reader but maybe Gu and Dao accidentally solved politics!?\]

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/9df5248e-cf67-402d-8946-1155ffd4d087-RackMultipart20240211-93-x8z1fv.png)

The upshot is **state representation is critical**. A smaller state is more
efficient; a larger state is more effective. The key is to **selectively** and
**dynamically** compress data into the state. Mamba’s Selection Mechanism allows
for context-dependent reasoning, focusing and ignoring. For both performance and
interpretability, understanding the state seems to be very useful.

## Information Flow in Transformer vs Mamba

How do Transformers know anything? At initialisation, a transformer isn’t very
smart. It learns in two ways:

1. Training data (Pretraining, SFT, RLHF etc)
2. In context-data

#### Training Data

Models learn from their training data. This is a kind of lossy compression of
input data into the weights. We can think of the effect of pretraining data on
the transformer kinda like the effect of your ancestor’s experiences on your
genetics - you can't recall their experiences, you just have vague instincts
about them. \[Footnote: this isn’t a perfect analogy as human evolution follows
a genetic algorithm rather than SGD\]

#### In Context-Data

Transformers use their context as short-term memory, which they can recall with
~perfect fidelity. So we get
[In-Context Learning](https://thegradient.pub/in-context-learning-in-context/),
e.g. using induction heads to solve the
[Indirect Object Identification](https://arxiv.org/pdf/2211.00593.pdf) task, or
[computing Linear Regression](https://proceedings.neurips.cc/paper_files/paper/2022/file/c529dba08a146ea8d6cf715ae8930cbe-Paper-Conference.pdf).

#### Retrieval

Note that Transformers don’t filter their context at all until recall time. So
if we have a bunch of information we think _might_ be useful to the Transformer,
we filter it _outside_ the Transformer (using Information Retrieval strategies)
and then stuff the results into the prompt. This process is known as Retrieval
Augmented Generation (RAG). RAG determines relevant information for the context
window of a transformer. A human with the internet is kinda like a RAG system -
you still have to know what to search but whatever you retrieve is as salient as
short-term memory to you.

#### Information Flow for Mamba

Training Data acts similarly for Mamba. However, the lines are slightly blurred
for in-context data and retrieval. In-context data for Mamba _is_
compressed/filtered similar to retrieval data for transformers. This in-context
data is also accessible for look-up like for transformers (although with
somewhat lower fidelity).

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/e333f1ed-1bf8-4eb8-bb84-9ac317b0713f-RackMultipart20240211-100-nytvtv.png)

Transformer context is to Mamba states what short-term is to long-term memory.
Mamba doesn’t just have “RAM”, it has a hard drive. \[Footnote: Albeit a pretty
weird hard drive at that - it morphs over time rather than being a fixed
representation.\]

\[Footnote: I've started calling the hidden_state the state space dimension (or
selective state dimension) which shortens to `SSD,` a nice reminder for what
this object represents - the long-term memory of the system.\]

### Swapping States as a New Prompting Paradigm

Currently, we often use RAG to give a transformer contextual information.

With Mamba-like models, you could instead imagine having a library of states
created by running the model over specialised data. States could be shared kinda
like LoRAs for image models.

For example, I could do inference on 20 physics textbooks and, say, 100 physics
questions and answers. Then I have a state which I can give to you. Now you
don’t need to add any few-shot examples; you just simply ask your question. The
in-context learning is in the state.

In other words, you can drag and drop states from others into your model, like
literal plug-in cartridges. And note that “training” a state doesn’t require any
backprop. It’s more like a highly specialised one-pass fixed-size compression
algorithm. This is unlimited in-context learning applied at inference time for
zero-compute or latency. \[footnote: I’m thinking about this similarly to the
relationship between harmlessness finetuning and activation steering. State
swapping, like activation steering, is an inference time intervention giving
comparable results to its train time analogue\].

The structure of an effective LLM call goes from…

1. System Prompt
2. Preamble
3. Few shot-examples
4. Question

…for Transformers, to simply…

1. Inputted state (with problem context, initial instructions, textbooks, and
   few-shot examples)
2. Short question

…for Mamba.

This is cheaper and faster than few-shot prompting (as the state is infinitely
reusable without inference cost). It’s also MUCH cheaper than finetuning and
doesn’t require any gradient updates. We could imagine retrieving states in
addition to context.

## Mamba & Mechanistic Interpretability

Transformer interpretability typically involves:

1. understanding token relationships via attention,
2. understanding circuits, and
3. using
   [Dictionary Learning](https://www.kolaayonrinde.com/blog/2023/11/03/dictionary-learning.html)
   for unfolding MLPs.

Most of the ablations that we would like to do for Mamba are still valid, but
understanding token communication (1) is now more nuanced. All information moves
between tokens via hidden states instead of the Attention Mechanism which can
“teleport” information from one sequence position to another.

For understanding in-context learning (ICL) tasks with Mamba, we will look to
intervene on the SSM state. A classic task in-context learning task is Indirect
Object Identification in which a model has to finish a paragraph like:

> _Then, Shelby and Emma had a lot of fun at the school. \[Shelby/Emma\] gave an
> apple to \[BLANK\]_

The model is expected to fill in the blank with the name that is not repeated in
the paragraph. In the chart below we can see that information is passed from the
\[Shelby/Emma\] position to the final position via the hidden state (see the two
blue lines in the top chart).

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/bc3ae841-a11b-44fe-9a47-aabbc44edcb1-RackMultipart20240211-138-5mskpz.png)

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/272a674c-564f-4e8a-8725-865bf1c9f405-RackMultipart20240211-174-fpjnar.png)

^Credit to Tessa for the charts above - can roll my own patching if preferable!

Since it’s hypothesised that much of In-Context Learning in Transformers is
downstream of more primitive sequence position operations (like
[Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)),
Mamba being able to complete this task suggests a more general In-Context
Learning ability.

## What’s Next for Mamba & SSMs?

Mamba-like models are likely to excel in scenarios requiring extremely long
context and long-term memory. Examples include:

- Processing DNA
- Generating (or reasoning over) video
- Writing novels

An illustrative example is agents with long-term goals.

> Suppose you have an agent interacting with the world. Eventually, its
> experiences become too much for the context window of a transformer. The agent
> then has to compress or summarise its experiences into some more compact
> representation.
>
> But how do you decide what information is the most useful as a summary? If the
> task is language, LLMs are actually fairly good at summaries - okay, yeah,
> you'll lose some information, but the most important stuff can be retained.
>
> However, for other disciplines, it might not be clear how to summarise. For
> example, what’s the best way to summarise a 2 hour movie? \[Footnote: this is
> a very non-trivial problem! How do human brains represent a movie internally?
> (It's not a series of the most salient frames, nor is it a text summary of the
> colours, nor is it a purely vibes-based summary if you can memorise some lines
> of the film)\] Could the model itself learn to do this naturally rather than a
> hacky workaround like trying to describe the aesthetics of the movie in text?

This is what Mamba allows. Actual long-term memory. A real state where the model
learns to keep what's important.
[Prediction is compression](https://arxiv.org/pdf/2309.10668.pdf) - learning
what's useful to predict what's coming next inevitably leads to building a
useful compression of the previous tokens.

---

The implications for Assistants are clear:

Your chatbot co-evolves with you. It remembers.

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/ec4bfd1b-ba53-412f-b737-005ea51866bb-RackMultipart20240210-189-qp4v5h.png)

Caption: Movie HER

### Agents & AI Safety

One reason for positive updates in existential risk from AGI is Language Models.
Previously, Deep-RL agents trained via self-play looked set to be the first
AGIs. Language models are inherently much safer since they aren’t trained with
long-term goals. \[Footnote: They’re also safer since they inherently understand
(though don’t necessarily embody) human values. It’s not all clear that how to
teach an RL agent human morality.\]

The potential for long-term sequence reasoning here brings back the importance
of agent-based AI safety. Few agent worries are relevant to Transformers with an
8k context window. Many are relevant to systems with impressive long-term
memories and possible instrumental goals.

### The Best Collab Since Taco Bell & KFC: 🤖 x 🐍

We might want to combine Mamba’s long context with the Transformer’s high
fidelity over short sequences. For example, if you’re making long videos, you
likely can't fit a whole movie into a Transformer’s context for attention
\[footnote: note that typically an image (i.e. a single frame) counts as X
tokens, and movies are typically 24 fps so you’ll fill a 32k context window in Y
seconds 🤯\]. You can imagine having Attention look at the most recent frames
for short-term fluidity and an SSM for long-term narrative consistency.

\[Footnote: Another possibility that I’m excited about is applying optimisation
pressure to the state itself as well as the output to have models that respect
particular use cases.\]

---

This isn’t the end of Transformers. Their high effectiveness is exactly what’s
needed for many tasks. But now Transformers aren’t the only option. Other
architectures are genuinely feasible.

So we’re not in the post-Transformer era. But for the first time, we’re living
in the post-”_only_\-Transformers” era \[Footnote: this is slightly hyperbolic,
the TS-Mixer for time series, Gradient Boosting Trees for tabular data and Graph
Neural Networks for weather prediction exist and are currently used, but these
aren’t at the core of AI\]. And this blows the possibilities wide open for
sequence modelling with extreme context lengths and native long-term memory.

Two ML researchers, Sasha Rush (HuggingFace, Annotated Transformer, Cornell
Professor) and Jonathan Frankle (Lottery Ticket Hypothesis, MosaicML, Harvard
Professor), currently have a bet [here](http://www.isattentionallyouneed.com/).

![](https://lex-img-p.s3.us-west-2.amazonaws.com/img/c064c339-aacd-4154-8aa8-8569556fa56a-RackMultipart20240210-169-1001ji.png)

Currently Transformers are far and away in the lead. With 3 years left, there’s
now a research direction with a fighting chance.

All that remains to ask is: `Is Attention All We Need?`

Thanks to Gonçalo for reading an early draft and Jaden for the nnsight library
used for the Interpretability analysis.

Also see: [Mamba paper](https://arxiv.org/pdf/2312.00752.pdf), Mamba Python
code, [Annotated S4](https://srush.github.io/annotated-s4/),
[Labenz podcast](https://www.cognitiverevolution.ai/emergency-pod-mamba-memory-and-the-ssm-moment/)

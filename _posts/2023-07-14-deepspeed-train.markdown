---
layout: post
title: "DeepSpeed's Bag of Tricks for Speed & Scale: Part I"
# date: 2023-07-14 20:29:35 +0100
tags: machine-learning deepspeed training
---

<!-- # DeepSpeed's Bag of Tricks for Speed & Scale -->

## Part I: An Introduction to DeepSpeed for Training

In the literature and the public conversation around Natural Language Processing, lots has been made of the results of scaling up data, compute and model size. For example we have the [original](https://arxiv.org/abs/2001.08361) and [updated](https://arxiv.org/abs/2203.15556) transformer scaling laws.

<div align="center">
  <figure>
    <img src="/blog/images/deepspeed/stack_more_layers.webp" width="500" alt="Layers">
    <figcaption>Keep it stacking</figcaption>
    </figure>
</div>

One sometimes overlooked point is the vital role of engineering breakthroughs in enabling large models to be trained and served on current hardware.

This series is about the engineering tricks that bring the research to life.

> _Note: This post assumes some basic familiarity with PyTorch/Tensorflow and transformers. If you've never used
> these before check out the [PyTorch docs](https://pytorch.org/docs/stable/index.html) and the
> [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Some background on backpropagation
> works will also be useful - check out [this video](https://www.youtube.com/watch?v=Ilg3gGewQ5U) if you want a
> refresher!_

### 0.1 DeepSpeed's Three Innovation Pillars

[DeepSpeed](https://www.deepspeed.ai) has three main use cases: enabling large training runs, decreasing inference latency and model compression.

<div align="center">
  <figure>
  <img src="https://github.com/microsoft/DeepSpeed/raw/master/docs/assets/images/3pillars.png" width="700" alt="">
  </figure>
</div>

This post covers training optimizations. Future posts will detail other pillars.

### 0.2 Problems Training Large Models

Training large models (e.g. LLMs) on huge datasets can be can be prohibitively slow, expensive, or even impossible with available hardware.

In particualar, very large models generally do not fit into the memory of a single GPU/TPU node. Compared to CPUs, GPUs are generally higher throughput but lower memory capacity.
(A typical GPU may have 32GB memory versus 1TB+ for CPUs).

Our aims are:

1. To train models too large for a single device
2. Efficiently distribute computation across devices
3. Fully utilize all devices as much as possible
4. Minimize communication bottlenecks _between_ devices

DeepSpeed reduces compute and time to train by >100x for large models.

If you just want to see how to implement DeepSpeed in your code, see the "Using DeepSpeed" section below.

## 1. Partial Solutions

### 1.1 Naive Data Parallelism

Without any data parallelism, we get this sorry sight:

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/gpu_unused.png" width = "700" alt="Unused GPU potential">
  <figcaption>Oh dear</figcaption>
  </figure>
</div>

We've spent a lot of money on GPU cores for them all to sit there idle apart from one! Unless you're single-handedly trying to prop up the NVIDIA share price, this is a terrible idea!

One thing that we might try is splitting up the data, parallelising across devices. Here we copy the entire model onto each worker, each of which process different subsets of the training dataset.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/data_parallel.png" width = "700" alt="Data Parallisation">
  <figcaption>Data Parallelisation</figcaption>
  </figure>
</div>

Each device compute its own gradients and then we average out the gradients across all the nodes to update our parameters with `all_reduce`. This approach is pretty straightforward to implement and works for any model type.

We've turned more GPUs into more speed - great!

In addition we also increase effective batch size, reducing costly parameter updates. Since with larger batch sizes there is more signal in each gradient update, this also improves convergence (up to a point).

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/whats_the_catch.gif" alt="What's The Catch">
  <figcaption>I thought you'd never ask</figcaption>
  </figure>
</div>

Unfortunately the memory bottleneck still remains. For Data Parallelism to work, the entire model has to fit on every device, which just isn't going to happen for large models.

### 1.2 Naive Model Parallelism

Another thing we could try is splitting up the computation of the model itself, putting different layers (transformer blocks) on different devices. With this model parallelism approach we aren't limited by the size of a memory of a single GPU, but instead by all the GPUs that we have.

However two problems remain. Firstly how to split up a model efficiently is very dependant on the specific model architecture (for example the number of layers and attention heads). And secondly communicating _between_ nodes now bottlenecks training.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/model_parallel.png" width = "600" alt="Model parallelisation">
  <figcaption>One batch moving through the parallelised model. In model parallelisation, one forward and backward pass requires all the devices, most of which are idle at any one time</figcaption>
  </figure>
</div>

Since each layer requires the input to the previous layer in each pass, workers spend most of their time waiting. What a waste of GPU time!
Here it looks like the model takes the same amount of time as if we had a GPU to fit it on but it's even worse. The communication overhead of getting data between nodes makes it even _slower_ than a single GPU.

Can we do better than this?

### 1.3 A Better Way: DeepSpeed

Data Parallelism gave speedups but couldn't handle models too large for a single machine. Model Parallelism allowed us to train large models but it's slow.

We really want a marriage of the ideas of both data and model parallelism - speed and scale together.

We don't always get what we want, but in this case we do. With DeepSpeed, Microsoft packaged up a bag of tricks to allow ML engineers to train larger models more efficiently. All in, DeepSpeed enables >100x lower training time and cost with minimal code changes - just 4 changed lines of PyTorch code. Let's walk through how.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/dp_vs_mp.png" width = "700" alt="DP vs MP">
  <figcaption>Data Parallelisation vs Model Parallelism</figcaption>
  </figure>
</div>

## 2. DeepSpeed Deep Dive: Key Ideas

~~One~~ Seven Weird Tricks to Train Large Models:

0. Mixed precision training
1. Delaying Weight Updates
2. Storing the optimiser states without redundancy (ZeRO stage 1)
3. Storing gradients and parameters without redundancy (ZeRO stages 2 & 3)
4. Tensor Clicing
5. Gradient Checkpointing
6. Quality of Life Improvements and Profiling

### 2.0 Mixed Precision Training

Ordinarily mathenatical operations are performed with 32 bit floats (fp32).
Using half precision (fp16) vs full precision (fp32) halves memory and speeds up computation.

We forward/backward pass in fp16 for speed, keeping copies of fp32 optimizer states (momentum, first order gradient etc.) for accuracy. The high precision fp32 maintains the high dynamic range so that we can still represent very slight updates.

### 2.1 Delaying Weight Updates

A simple training loop might contain something like:

```python
for i, batch in enumerate(train_loader):

    for j, minibatch in enumerate(batch):

        loss = model(minibatch)
        local_gradients = gradients(loss / batch_size)
        average_gradients = distributed.all_reduce(local_gradients) # reduce INSIDE inner loop

    optimizer.step(average_gradients)
```

Note here that within every loop we're calculating not only the local gradients but also synchronizing gradients which requires communicating with all the other nodes.

Delaying synchronization improves throughput e.g:

```python
for i, batch in enumerate(train_loader):

    for j, minibatch in enumerate(batch):

        loss = model(minibatch)
        local_gradients = gradients(loss / batch_size)

    average_gradients = distributed.all_reduce(local_gradients) # reduce OUTSIDE inner loop
    optimizer.step(average_gradients)
```

### 2.2 Storing Optimiser States Without Redundancy (ZeRO stage 1)

Suppose we have a GPU with 50GB of memory and our model weights are 10GB of memory. That's all great right?

For inference we feed in our input data and get out activations at each step.
Then once we pass each layer, we can throw away activations from prior layers. Our model fits on the single GPU.

For training however, it's a different story. Each GPU needs its intermediate activations, gradients and the fp32 optimiser states for backpropagation.
Pretty soon we're overflowing the GPU with our model's memory footprint 😞

The biggest memory drain on our memory is the optimisation states.

We know that we're going to need to get multiple GPUs and do some model parallelisation here. Eventually we want to partition the whole model but a good first move would be to at least remove optimisation state redundancy.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/zero_stages.png" width="800" alt="The Stages of ZeRO">
  <figcaption>The Stages of Zero Redundancy Optimisation (ZeRO)</figcaption>
  </figure>
</div>

For ZeRO stage 1, in the backward pass, each device calculates the (first order) gradients for the final section of the model. The final device `gathers` all these gradients, averages them and then computes the Adam optimised gradient with the optimisation states. It then `broadcasts` back the new parameter states for the final section of the model to all devices. Then the penultimate device will do the same and so on until we reach the first device.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/zero1-t1.gif" width = "800" alt="ZeRO Stage 1">
  <figcaption>
    ZeRO Stage 1
</figcaption>
  </figure>
</div>
We can think of this as a 5 step process:

1. All nodes calculate gradients from their loss (note they all did a forward pass on different data so their losses will be different!)
2. Final node collects and averages the gradients from all nodes via `reduce`
3. Final node calculates gradient update using optimiser states
4. Final node `broadcasts` the new gradients to all of the nodes.
5. Repeat for penultimate section and so on to complete the gradient updates.

ZeRO stage 1 typically reduces our memory footprint by ~4x.

```ruby
🔄 Fun Fact: The name DeepSpeed is a palindrome! How cute 🤗
```

### 2.3 Storing Gradients and Parameters Without Redundancy (ZeRO stages 2 & 3)

We can take the partitioning idea further and do it for parameters and gradients as well as optimisation states.

#### In the forward pass:

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/zero3_forward.gif" width="800" alt="ZeRO Stage 3 (Forward)">
  <figcaption>
    ZeRO Stage 3: forward pass
</figcaption>
  </figure>
</div>
1. The first node `broadcasts` the parameters for the first section of the model.
2. All nodes complete the forward pass for their data for the first section of the model.
3. They then throw away the parameters for first section of the model.
4. Repeat for second section and so on to get the loss.

#### And the backward pass:

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/zero3_backward.gif" width = "800" alt="ZeRO Stage 3 (Backward)">
  <figcaption>
    Zero Stage 3: backward pass
</figcaption>
  </figure>
</div>
1. The final node `broadcasts` its section gradients.
2. Each backpropagate their own loss to get the next gradients.
3. As before, final node accumulates and averages all gradients (`reduce`), calculates gradient update with optimiser and then `broadcasts` the results, which can be used for the next section.
4. Once used, all gradients are thrown away by nodes which are not responsible for that section.
5. Repeat for penultimate section and so on to complete the gradient updates.

If we have `N` cores, we now have an `N`x memory footprint reduction from ZeRO.

#### A breather

That was the most complex part so feel free to check out these resources to make sure you understand what's going on:

- [DeepSpeed founder at MLOps community](https://www.youtube.com/watch?v=y4_bCiAsIAk&list=PLDEUW02OCkqGZ5_8jVQUK0dRJx8Um-hpc&index=1&t=20s)
- [Microsoft blog post](https://www.microsoft.com/en-us/research/blog/ZeRO-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

It's all downhill from here!

#### Benefits of ZeRO

Overall, ZeRO removes the redundancy across data parallel process by partitioning optimizer states, gradients and parameters across nodes. Look at how much memory footprint we've saved!

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/deepspeed_benefits.png" width="800" alt="DeepSpeed Benefits">
  <figcaption>Benefits of DeepSpeed</figcaption>
  </figure>
</div>

One surprising thing about this approach is that it scales superlinearly. That is, when we double the number of GPUs that we're using, we _more than_ double the throughput of the system!
In splitting up the model across more GPUs, we leave more space per node for activations which allows for higher batch sizes.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/superlinear_scale.png" alt="Superlinear Scale">
  <figcaption>Superlinear Scale of DeepSpeed vs Perfect Scaling</figcaption>
  </figure>
</div>

### 2.4 Tensor Slicing

Most of the operations in a large ML model are matrix multiplications followed by non-linearities. If we want to parallelise this across GPUs _within the same core_, we can slice up huge tensors into smaller ones and then combine the results at the end.

For matrices $ X = \begin{bmatrix} X_1 & X_2 \end{bmatrix} $ and $ A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} $, we note that:

$
XA = \begin{bmatrix} X_1 & X_2 \end{bmatrix} \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}
$

For example:

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/row_slicing_numbers.png" width = "700" alt="Row Slicing">
  <figcaption>Row Slicing</figcaption>
  </figure>
</div>

However if there is a non-linear map after the M e.g. if $ Y = \text{ReLU}(MX) $, this slicing isn't going to work. $ \text{ReLU}(M_1X_1 + M_2X_2) \neq \text{ReLU}(M_1X_1) + \text{ReLU}(M_2X_2) $ in general by non-linearity.
So we should instead split up X by columns and duplicate M across both nodes such that we have:

$ Y = [Y_i, Y_z] = [\text{GeLU}(X A_1), \text{GeLU}(X A_2)] = XA $

For example:

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/column_slicing_numbers.png" width = "700" alt="Column Slicing">
  <figcaption>Column Slicing</figcaption>
  </figure>
</div>

Note: normally we think of A acting on X by left multiplication. In this case X is our data and A is the weights which we want to parallelise. Through taking transposes we can swap the order of the geometric interpretation so we can think of the above as linear map A acting on our data X and still retain the slicing.

### 2.5 Gradient Checkpointing

In our description of ZeRO each core cached (held in memory) the activations for it's part of the model.

<div align="center">
  <figure>
  <img src="https://github.com/cybertronai/gradient-checkpointing/raw/master/img/output.gif" alt="Regular backprop">
  <figcaption>The top layer represents the activations in the model populating during the forward pass and the lower layer, the gradients populated in the backward pass. The first circle is the input data and the bottom right is the loss.</figcaption>
  </figure>
</div>

Suppose we had extremely limited memory but were flush with compute.
An alternative approach to storing all the activations would be to simply recompute them when we need in the backward pass. We can always recompute the activations by running the same input data through a forward pass.

<div align="center">
  <figure>
  <img src="https://github.com/cybertronai/gradient-checkpointing/raw/master/img/output_poor.gif" alt="Memory poor backprop">
  <figcaption>Here each activation is computed just before it's needed using forward passes.</figcaption>
  </figure>
</div>

This recomputing approach saves lots of memory but is quite compute wasteful, incurring `m` extra forward passes for an `m-layer` transformer.

A middle ground approach to trading off compute and memory is [gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing). Here we store some intermediate activations with $\sqrt m$ of the memory for the cost of one forward pass.

<div align="center">
  <figure>
  <img src="https://github.com/cybertronai/gradient-checkpointing/raw/master/img/output2.gif" alt="Gradient Checkpointing">
  <figcaption>Here the only the second layer activations are cached as a "checkpoint". Now for activations after the checkpoint instead of comptuting from the input data, we can compute from the checkpoint. This approach trades off memory and compute.</figcaption>
  </figure>
</div>

### 2.6 Profiling etc

While not strictly causing any code optimisations, DeepSpeed provides developer friendly features like convenient profiling and monitoring to track latency and performance. We also have model checkpointing so you can recover a model from different points in training.
Developer happiness matters almost as much as loss!

<div align="center">
  <figure>
  <img src="https://i.imgflip.com/7s8ojc.jpg" width = "500" alt="Happy">
  <figcaption>Happy engineers write happy code</figcaption>
  </figure>
</div>

Check out the [docs](https://deepspeed.readthedocs.io/en/latest/) for more info!

## 3. In Pictures

<video controls width = "700">
  <source src="/blog/images/deepspeed/Turing-Animation.mp4" type="video/mp4">
</video>

_Animated Video from Microsoft: warning, it's a little slow._

## 4. In Code

The full DeepSpeed library, with all the hardware level optimisations, is open-sourced. See the [core library](https://github.com/microsoft/DeepSpeed/), the [docs](https://www.deepspeed.ai/training/) and [examples](https://github.com/microsoft/DeepSpeedExamples).

## 5. Using DeepSpeed

DeepSpeed integrates with PyTorch and TensorFlow to optimize training.

<div align="center">
  <figure>
  <img src="/blog/images/deepspeed/stack.png" width = "600" alt="Stack">
  </figure>
</div>

In PyTorch we only need to change 4 lines of code to apply DeepSpeed such that our code is optimised for training on a single GPU machine, a single machine with multiple GPUs, or on multiple machines in a distributed fashion.

First we swap out:

```python
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

with initialising DeepSpeed by writing:

```python
ds_config = {
  "train_micro_batch_size_per_gpu": batch_size,
  "optimizer": {
      "type": "Adam",
      "params": {
          "lr": 1e-4
      }
  },
  "fp16": {
      "enabled": True
  },
  "zero_optimization": {
      "stage": 1,
      "offload_optimizer": {
         "device": "cpu"
      }
  }
}

model_engine, *_ = initialize(model=model_architecture,
                       model_parameters=params,
                       config = ds_config)
```

Then in our training loop we change out the original PyTorch...

```python
for step, batch in enumerate(data_loader):
    # Calculate loss using model e.g.
    output = model(batch)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()
```

for:

```python
for step, batch in enumerate(data_loader):
    # Forward propagation method to get loss
    loss = ...

    # Runs backpropagation
    model_engine.backward(loss)

    # Weights update
    model_engine.step()
```

That's all it takes! In addition, DeepSpeed's backend has also been integrated with HuggingFace via the [Accelerate library](https://huggingface.co/docs/accelerate/index).

## That's All Folks!

There's a lot of clever improvements that go into the special sauce for training large models. And for users, with just a few simple code changes, DeepSpeed works its magic to unleash the power of all your hardware for fast, efficient model training.

Next time we'll talk about the engineering tricks for Inference.

Happy training!

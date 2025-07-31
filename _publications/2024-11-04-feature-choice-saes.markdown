---
layout: publication
title: "Adaptive Sparse Allocation with Mutual Choice & Feature Choice Sparse Autoencoders"
authors:
  - "Kola Ayonrinde"
year: 2024
journal: "arXiv"
external_url: https://arxiv.org/abs/2411.02124
---

Sparse autoencoders (SAEs) are a promising approach to extracting features from neural networks, enabling model interpretability as well as causal interventions on model internals. SAEs generate sparse feature representations using a sparsifying activation function that implicitly defines a set of token-feature matches. We frame the token-feature matching as a resource allocation problem constrained by a total sparsity upper bound. For example, TopK SAEs solve this allocation problem with the additional constraint that each token matches with at most k features. In TopK SAEs, the k active features per token constraint is the same across tokens, despite some tokens being more difficult to reconstruct than others. To address this limitation, we propose two novel SAE variants, Feature Choice SAEs and Mutual Choice SAEs, which each allow for a variable number of active features per token. Feature Choice SAEs solve the sparsity allocation problem under the additional constraint that each feature matches with at most m tokens. Mutual Choice SAEs solve the unrestricted allocation problem where the total sparsity budget can be allocated freely between tokens and features. Additionally, we introduce a new auxiliary loss function, 𝚊𝚞𝚡_𝚣𝚒𝚙𝚏_𝚕𝚘𝚜𝚜, which generalises the 𝚊𝚞𝚡_𝚔_𝚕𝚘𝚜𝚜 to mitigate dead and underutilised features. Our methods result in SAEs with fewer dead features and improved reconstruction loss at equivalent sparsity levels as a result of the inherent adaptive computation. More accurate and scalable feature extraction methods provide a path towards better understanding and more precise control of foundation models.

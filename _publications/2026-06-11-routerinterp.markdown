---
layout: publication
title: "RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing"
authors:
  - "Ilya Lasy"
  - "Nora Yinuo Cai"
  - "Kola Ayonrinde"
year: 2026
journal: "ICML 2026"
external_url: https://openreview.net/forum?id=PHNsolfvMj
selected: true
selected_order: 2
---

Sparse Mixture-of-Experts (MoE) models scale more efficiently than dense models by routing tokens to modular expert networks that are only active when relevant to the task. A leading hypothesis for the performance of MoE models is that each expert specialises in a single, coherent domain. However, interpretability efforts that assume this hypothesis have generally been unsuccessful. We propose and present evidence for an alternative account that we call the Superposed Specialisation Hypothesis (SSH): experts specialise in a disjoint union of fine-grained features rather than one broad domain. Leveraging the SSH, we introduce RouterInterp, a method for interpreting expert routing that identifies Sparse Autoencoder features most predictive of routing decisions and produces unified natural-language explanations. On gpt-oss-20b, explanations from RouterInterp predict expert routing with 77% higher accuracy than prior methods. This work provides a scalable method for generating concise and more accurate explanations of expert routing and increases our understanding of a previously uninterpretable component of foundation models.

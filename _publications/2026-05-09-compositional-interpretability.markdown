---
layout: publication
title: "From Mechanistic to Compositional Interpretability"
authors:
  - "Ward Gauderis"
  - "Thomas Dooms"
  - "Steven T. Homer"
  - "Kola Ayonrinde"
  - "Geraint A. Wiggins"
year: 2026
journal: "arXiv"
external_url: https://arxiv.org/abs/2605.08934
---

Mechanistic interpretability aims to explain neural model behaviour by reverse-engineering learned computational structure into human-understandable components. Without a formal framework, however, mechanistic explanations cannot be objectively verified, compared, or composed. We introduce compositional interpretability, a category-theoretic framework grounded in the principles of compositionality and minimum description length. Compositional interpretations are pairs of syntactic and semantic mappings that must commute to enforce consistency between a model's decomposition and its observed behaviour. We deconstruct explanation quality into measures of faithfulness and complexity to cast interpretability as a constrained optimisation problem, and introduce compressive refinement to systematically restructure models into simpler parts without altering their function. Finally, we derive a parsimony criterion under which syntactic compression theoretically guarantees more concise, human-aligned explanations. Our framework situates prominent mechanistic methods as subclasses of refinement, and clarifies why their compressibility heuristics tend to align with human interpretability. Our work provides a measurable, optimisable blueprint for automating the discovery and evaluation of mechanistic explanations.

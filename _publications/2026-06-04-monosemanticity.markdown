---
layout: publication
title: "Getting Monosemantic About Monosemanticity"
authors:
  - "*Raphaël Millière"
  - "*Kola Ayonrinde"
year: 2026
journal: "🌟 ICML 2026, Oral at Philosophy Meets Machine Learning Workshop"
---

Mechanistic interpretability often treats monosemanticity as a central target: successful decompositions of neural networks should replace polysemantic neurons with units that carry single meanings. We argue that this target is underspecified. Current research uses several criteria to determine whether a unit is "monosemantic": that it fires on a coherent set of inputs, that it reliably detects a particular concept, that its behaviour admits a short natural-language description, that it plays a single causal role in the model's computation, or that it cannot be split into smaller components. These criteria are related but not equivalent, and the same unit can pass under one and fail under another. We argue that criteria for monosemanticity play three different roles, and that distinguishing the roles dissolves much of the apparent disagreement in the literature. Some criteria supply inconclusive evidence that a unit is monosemantic, while others pick out desirable properties for human-intelligibility rather than monosemanticity directly. We argue that only two criteria are actually constitutive of monosemanticity: the unit must map onto a single causally relevant variable in the model's computation, and decomposing it further must not improve the explanation at the chosen level of grain. Building on these two criteria, we outline a framework that clarifies what feature-level evaluations and interventions can and cannot establish, and provides better reporting norms for claims about monosemanticity.

# Maths on Mamba

$h^*_t =  A(x_t)h_{t-1} + B(x_t)x_t$

<!-- $h_t = ... + A_tA_{t-1}B_{t-1}x_{t-1} + ...$ -->

$h_t = h_{t-1}$

OR

$h_t = A(x'_t) h_{t-1}$

h\_{t-1} = (1,2)

A_t = (4,6)

h_t = (4, 12)

---

$h_{t+1} = A_{t+1}h_t + B_{t+1}x_{t+1}$

$B_t = B(x_t)$

x = clean_prompt

x' = corrupted_prompt

A = Float[t.Tensor, "batch seq input_dim state_dim"]

A_t = Float[t.Tensor, "batch input_dim state_dim"]

```python
h_t = einsum("batch input_dim state_dim, batch input_dim state_dim -> batch input_dim state_dim", A_t, h_{t-1})
```

$A_t = A(x_t)$

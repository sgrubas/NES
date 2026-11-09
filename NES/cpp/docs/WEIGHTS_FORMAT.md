# Weight file format (`.nesw`), version 1

`tools/export_nes_weights.py` writes a trained NES network to one file; `nes_cpp::NesModel::load` reads it. The file is a short text header followed by the weights as raw binary numbers. The network it describes is defined in [`NES_INFERENCE.md`](NES_INFERENCE.md).

## Header

ASCII lines `key = value`, ending with a line `end_header`. Lines end with `\n`. Numbers are written in Python's shortest round-trip form (`repr(float)`), so every float64 survives the text exactly; C++ reads them with `std::from_chars`, which does not depend on the locale. Lists are separated by single spaces. Booleans are `0` or `1`.

| key | value | meaning |
|---|---|---|
| `format` | `nes-weights` | identifies the file |
| `version` | `1` | format version; the reader refuses other versions |
| `solver` | `NES_OP` or `NES_TP` | one-point or two-point network |
| `dim` | `2` or `3` | number of coordinates |
| `xs` | `dim` numbers | source position (NES-OP only) |
| `reciprocity` | `none` or `output` | swap averaging (NES-TP only) |
| `input_scale` | number | factor $s$ applied to the features |
| `factored` | `0` / `1` | multiply by $\lvert\mathbf{x}_r - \mathbf{x}_s\rvert$ |
| `out_vscale` | `0` / `1` | map $\tau$ into $[\sigma_{\min}, \sigma_{\max}]$ |
| `slow_min`, `slow_max` | numbers | $\sigma_{\min} = 1/v_{\max}$, $\sigma_{\max} = 1/v_{\min}$ |
| `widths` | $L + 1$ integers | number of features $n_0$, then the hidden widths $n_1 \dots n_L$ |
| `act` | name | hidden activation function $g$ (table in `NES_INFERENCE.md`) |
| `act_n` | number | constant $n$ of the hidden activation |
| `act_a` | $L$ numbers | trainable $a$ of each hidden layer (1 if not adaptive) |
| `act_adaptive` | `0` / `1` | whether the hidden `a` are trainable (1) or constant (0) |
| `out_act`, `out_act_n` | name, number | output activation $\psi$ and its $n$ |
| `out_act_a` | number | its $a$ (1 if not adaptive) |
| `out_act_adaptive` | `0` / `1` | whether the output `a` is trainable |

Every key is required, except that `xs` appears only for NES-OP and `reciprocity` only for NES-TP. Unknown keys are an error.

## Weights

Right after the `end_header` line follow float64 numbers in little-endian byte order (the native order of x86-64 and ARM64), with no padding, in this order:

1. for each hidden layer $k = 1, \dots, L$: the kernel $W_k$, $n_{k-1} \times n_k$ values, row by row (C order, as NumPy stores a Keras kernel), then the bias $\mathbf{b}_k$, $n_k$ values;
2. the head kernel $\mathbf{w}$, $n_L$ values, then the head bias $c_0$, one value.

The reader checks that exactly $8\left(\sum_k (n_{k-1} + 1)\,n_k + n_L + 1\right)$ bytes follow the header.

## Example

A NES-TP network with `nl=2`, `nu=16`, `act='ad-gauss-1'`, default output activation:

```
format = nes-weights
version = 1
solver = NES_TP
dim = 2
reciprocity = output
input_scale = 1.0
factored = 1
out_vscale = 1
slow_min = 0.3333333333333333
slow_max = 0.5
widths = 4 16 16
act = gauss
act_n = 1.0
act_a = 1.0130213499069214 0.9814519882202148
act_adaptive = 1
out_act = sigmoid
out_act_n = 1.0
out_act_a = 1.0057282447814941
out_act_adaptive = 1
end_header
<(4*16 + 16 + 16*16 + 16 + 16 + 1) float64 values = 2952 bytes>
```

## Versions

A change that old readers cannot read correctly (a new key that changes the network, a different layout) increases `version`. Version 1 covers dense `TraveltimeNet` models without `improved_mlp`, with reciprocity `None` or `'output'`.

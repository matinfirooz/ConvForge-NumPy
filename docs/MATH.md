# CNNs, with the forward and backward passes exposed

This repository uses **NCHW** tensors: batch, channels, height, width.
The first layer of the default model receives `(N, 1, 24, 24)`.

## Convolution is weight-shared cross-correlation

The layer called `Conv2D` follows the common machine-learning cross-correlation
convention: it does **not** flip the kernel spatially.

$$
Y_{n,o,y,x}=b_o+\sum_c\sum_i\sum_j
W_{o,c,i,j}X_{n,c,ys_h+id_h-p_t,xs_w+jd_w-p_l}.
$$

Coordinates outside the input use zero padding. The same learned kernel visits
every output position. Stride moves the window; dilation spaces its samples.
With effective kernel height $k_{eff}=(k_h-1)d_h+1$:

$$
H_{out}=\left\lfloor\frac{H+p_t+p_b-k_{eff}}{s_h}\right\rfloor+1.
$$

For `padding="same"`, the output height is $\lceil H/s_h\rceil$ and padding
can be asymmetric. For `valid`, padding is zero. Width follows the same rule.

## im2col makes the shared filter a matrix product

Flatten each local input patch into a row of $C$. Flatten each kernel into a
row of $W_f$. The forward becomes:

$$
Y_f=CW_f^T+b.
$$

`C` has shape `(N*OH*OW, input_channels*KH*KW)`. The forward reshapes the
result back into NCHW. `sliding_window_view` supplies read-only patch views;
the transpose/reshape may allocate a full patch matrix. This is a transparent
CPU implementation, not a memory-minimal or GPU-optimized kernel.

The scalar `naive_conv2d` implementation independently checks this result.
[`examples/walkthrough.py`](../examples/walkthrough.py) prints a 4×4 input,
shared 3×3 filter, every patch row, the output, pooling, and gradients.

## Convolution backward: input, kernel, and bias

Let $G_f$ be the upstream gradient in flattened output order:

$$
D_{W_f}=G_f^TC,\qquad D_b=\operatorname{rowsum}(G_f),\qquad D_C=G_fW_f.
$$

`col2im` scatter-adds $D_C$ back into its input pixel positions, then crops
padding. A pixel often belongs to several windows, so all contributions must
be **added**. Overwriting overlapping contributions is a common backward bug.

`col2im` is the adjoint of `im2col`, not its inverse:

$$
\langle\operatorname{im2col}(X),G\rangle
=\langle X,\operatorname{col2im}(G)\rangle.
$$

Applying `col2im(im2col(X))` multiplies pixels by their coverage counts.
Tests verify this identity, overlap counts, stride, dilation, asymmetric
padding, and finite-difference gradients for pixels, kernels, and bias.

## ReLU and max pooling

$$
\operatorname{ReLU}(x)=\max(0,x),\qquad
D_x=D_y\,\mathbf{1}[x>0].
$$

The derivative at zero is defined to be zero. Max pooling sends each output
gradient only to its winning input. If a pooling window has equal maxima,
the implementation chooses the first row-major winner. Overlapping pooling
windows add gradients when they select the same pixel. With valid pooling,
uncovered border pixels receive zero gradient.

## Dense layers and classification

For $Y=XW+b$:

$$
D_W=X^TD_Y,\quad D_b=\sum_n(D_Y)_n,\quad D_X=D_YW^T.
$$

The model returns logits. Softmax subtracts the maximum before exponentiating.
For mean cross entropy over $N$ images:

$$
\mathcal{L}=-\frac{1}{N}\sum_n\log p_{n,y_n},\qquad
D_{logits}=\frac{p-\operatorname{onehot}(y)}{N}.
$$

The loss uses shifted log-sum-exp directly rather than taking the log of a
possibly underflowed probability.

## Inverted dropout and optimization

Training samples a keep mask with probability $1-r$ and uses
$Y=X\odot M/(1-r)$. Backward applies the same mask. Inference is identity.
Adam maintains first/second moment estimates and corrects their initial bias.
Global L2 clipping scales all gradients together when their norm exceeds the
configured limit. Tests independently verify the first Adam update and the
clipping norm.

Training accuracy/loss are averages of training-mode minibatches as weights
change. Validation is evaluated after each epoch with dropout disabled, so
these statistics are not identical measurement conditions.

## Pixel sensitivity

The laboratory differentiates the predicted class's **logit** with respect to
the input. It displays $|\partial z_c/\partial X|$, independently normalized
to `[0,1]`. Dropout is disabled for this derivative. This is local sensitivity
to infinitesimal pixel changes; it is not a causal explanation, a calibrated
confidence measure, or a Grad-CAM implementation.

## Default architecture and costs

| Stage | Output per image | Learnable parameters |
|:--|:--|--:|
| Input | `1 × 24 × 24` | 0 |
| Conv 3×3, same + ReLU | `8 × 24 × 24` | 80 |
| MaxPool 2×2 | `8 × 12 × 12` | 0 |
| Conv 3×3, same + ReLU | `16 × 12 × 12` | 1,168 |
| MaxPool 2×2 | `16 × 6 × 6` | 0 |
| Flatten | `576` | 0 |
| Dense + ReLU + training dropout | `32` | 18,464 |
| Classifier | `6` | 198 |
| **Total** | | **19,910** |

Convolution's arithmetic work is proportional to
$N H_{out}W_{out}C_{out}C_{in}K_hK_w$. This implementation also stores
patch matrices for backward, so memory grows with batch size and local-window
size. Rectangular kernels, stride, and dilation are supported; grouped/depthwise
convolution, batch normalization, GPU execution, and automatic differentiation
are outside the implementation.

## Reference material

- LeCun et al., *Gradient-based learning applied to document recognition*,
  [DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791), 1998.
- NumPy's [official sliding-window documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html).

ConvForge is a teaching implementation of established CNN ideas, not a new
convolution algorithm or a reproduction of the original LeNet experiments.

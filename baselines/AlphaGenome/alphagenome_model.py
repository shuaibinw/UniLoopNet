def conv_block(x: Array, num_channels: int, width: int = 5) -> Array:
    x = RMSBatchNorm(x)
    x = GeLU(x)
    if width == 1:
        x = Linear(num_channels=num_channels)(x)
    else:
        x = StandardizedConv1D(num_channels=num_channels, width=width)(x)
    return x

def dna_embedder(x: Array):
    out = Conv1D(num_channels=768, width=15)(x)
    return out + conv_block(out, num_channels=768)

def downres_block(x: Array) -> Array:
    # Increase number of channels by 128 and apply skip connection by padding.
    out = conv_block(x, num_channels=x.shape[-1] + 128)
    out = out + Pad(x, [(0, 0), (0, 0), (0, 128)])
    return out + conv_block(out, num_channels=out.shape[-1])

def sequence_encoder(x: Array) -> tuple[Array, map[str, Array]]:
    intermediates = {}
    for bin_size in [1, 2, 4, 8, 16, 32, 64]:
        block = dna_embedder if bin_size == 1 else downres_block
        x = block(x)
        intermediates[f'bin_size_{bin_size}'] = x
        x = MaxPool(pool_size=2)(x)
    return x, intermediates


def mha_block(x: Array[B, S, C], attention_bias: Array[B, 8, S, S]) -> Array[B, S, C]:
    x = RMSBatchNorm(x)
    # Multi-query attention: 8 query heads, 1 shared key/value head. Each
    # query and key head has 128 channels. The value head has 192 channels.
    q = LayerNorm(Linear((8, 128), with_bias=False)(x), axis=-1)
    k = LayerNorm(Linear((1, 128), with_bias=False)(x), axis=-1)
    v = LayerNorm(Linear((1, 192), with_bias=False)(x), axis=-1)
    # Apply RoPE with max_position = S = 8192 (1 Mb / 128 bp).
    q = apply_rope(q, max_position=8192)
    k = apply_rope(k, max_position=8192)
    attention_logits = Einsum('bshc,bS1c->bhsS', q, k) / sqrt(128)
    # Add bias from pairwise activations and soft-clip logits in [-5, 5].
    attention_logits = Tanh((attention_logits + attention_bias) / 5.0) * 5.0
    attention_weights = Softmax(attention_logits, axis=-1)
    y = Einsum('bhsS,bS1c->bshc', attention_weights, v)
    # Reshape and project back to input channels C = 1536.
    y = Linear(x.shape[-1])(y.reshape(y.shape[:2] + (-1,)))
    return Dropout(RMSBatchNorm(y))

def mlp_block(x: Array[B, S, C]) -> Array[B, S, C]:
    x = RMSBatchNorm(x)
    x = Linear(2 * x.shape[-1])(x)
    x = Dropout(ReLU(x))
    x = Linear(x.shape[-1])(x)
    return Dropout(RMSBatchNorm(x))

def attention_bias_block(x: Array[B, P, P, F]) -> Array[B, 8, S, S]:
    x = GeLU(RMSBatchNorm(x))
    x = Linear(8, with_bias=False)(x)
    # Repeat attention bias to S = 8192 length (factor S/P = 8192/512 = 16).
    x = Repeat(x, 16, axis=(1, 2))
    return Moveaxis(x, 3, 1)

def transformer_tower(x: Array[B, S, C]) -> tuple[Array[B, S, C], Array[B, P, P, F]]:
    pair_x = None
    for i in range(9):
        if i % 2 == 0:
            pair_x = pair_update_block(x, pair_x)
        x = x + mha_block(x, attention_bias=attention_bias_block(pair_x))
        x = x + mlp_block(x)
    return x, pair_x

def apply_rope(x: Array, max_position: int, positions: Array | None = None):
    positions = positions or Arange(x.shape[1])
    num_freq = x.shape[-1] // 2
    freq = 1.0 / (Arange(num_freq) + Geomspace(1, max_position - num_freq + 1, num_freq))
    theta = Repeat(Einsum('...s,f->...sf', positions, freq), 2, axis=-1)
    x_rotated = Stack((-x[..., 1::2], x[..., ::2]), axis=-1).reshape(x.shape)
    return x * Cos(theta) + x_rotated * Sin(theta)

def pair_update_block(sequence_input: Array[B, S, C], pair_input: Array[B, P, P, F] | None) -> Array[B, P, P, F]:
    y = sequence_to_pair_block(sequence_input)
    x = y if pair_input is None else pair_input + y
    x += row_attn_block(x)
    x += pair_mlp_block(x)
    return x

def sequence_to_pair_block(x: Array[B, S, C]) -> Array[B, P, P, F]:
    # Downsample sequence to P=512 length (Factor S/P = 8192/512 = 16).
    x = RMSNorm(AvgPool(pool_size=16)(x))
    # 32 query and key heads with 128 feature channels each.
    q = Linear((32, 128), with_bias=False)(x)
    k = Linear((32, 128), with_bias=False)(x)
    # Generate and project the directional positional relative encodings.
    pos_features = central_mask_features(sequence_length=512, feature_size=64)
    pos_encoding = Linear((32, 128))(pos_features)
    q_bias = GetParameter('q_r_bias', (1, 1, 32, 128))
    k_bias = GetParameter('k_r_bias', (1, 1, 32, 128))
    rel_q_a = relative_shift(Einsum('bqhc,bphc->bqph', q + q_bias, pos_encoding))
    rel_k_a = relative_shift(Einsum('bkhc,bphc->bkph', k + k_bias, pos_encoding))
    a = Einsum('bqhc,bkhc->bqkh', q, k) + (rel_q_a + rel_k_a.swapaxes(1, 2))/2
    # Additional projection based on outer sum of sequence embeddings.
    y_q = Linear(128, with_bias=False)(GeLU(x))
    y_k = Linear(128, with_bias=False)(GeLU(x))
    pair_activations = Linear(128)(a) + y_q[:, :, None, :] + y_k[:, None, :, :]
    return Dropout(pair_activations)

def central_mask_features(sequence_length: int, feature_size: int):
    # `relative_positions` spans from the min to the max `i_k - i_q`, i.e. -(L-1) to +(L-1)
    relative_positions = Arange(2 * sequence_length - 1) - (sequence_length - 1)
    center_widths = Arange(feature_size // 2) + Geomspace(1, sequence_length - feature_size // 2 + 1, feature_size // 2, endpoint=False)
    embeddings = (center_widths[None, :] > Abs(relative_positions)[:, None])
    return Concatenate([embeddings, Sign(relative_positions)[:, None] * embeddings], axis=-1)

def relative_shift(x: Array[..., S, 2 * S - 1]) -> Array:
    *batch_shapes, seq_length, num_diagonals = x.shape
    x = Pad(x, [(0,0)] * (len(batch_shapes)+1) + [(1,0)])
    x = x.reshape(batch_shapes + [num_diagonals + 1, sequence_length])
    return x[..., 1:, :].reshape(batch_shapes + [sequence_length, num_diagonals])

def row_attention_block(pair_input: Array[B, P, P, F]) -> Array[B, P, P, F]:
    x = RMSNorm(pair_input)
    # Single queries, keys and values heads with 128 feature channels.
    k = Linear(128, with_bias=False)(x)
    q = Linear(128, with_bias=False)(x)
    v = Linear(128)(x)
    x = Einsum('bpPf,bpkf->bpPk', q, k) / Sqrt(128)
    x = Einsum('bpPk,bpkf->bpPf', Softmax(x, axis=3), v)
    return Dropout(x)

def pair_mlp_block(pair_input: Array[B, P, P, F]) -> Array[B, P, P, F]:
    x = RMSNorm(pair_input)
    x = Linear(2 * pair_input.shape[-1])(x)
    x = Linear(pair_input.shape[-1])(ReLU(x))
    return Dropout(x)


def upres_block(x: Array, unet_skip: Array) -> Array:
    num_channels = unet_skip.shape[2]
    out = conv_block(x, num_channels) + x[:, :, :num_channels]
    out = Repeat(out, 2, axis=1) * GetParameter('residual_scale', (1,), init=0.9)
    out += conv_block(unet_skip, num_channels, width=1)
    return out + conv_block(out, num_channels)

def sequence_decoder(x: Array, intermediates: map[str, Array]) -> Array:
    for bin_size in [64, 32, 16, 8, 4, 2, 1]:
        x = upres_block(x, intermediates[f'bin_size_{bin_size}'])
    return x

def output_embedder(x: Array, organism_index: int, skip_x: Array | None = None) -> Array:
    x = Linear(2 * x.shape[2])(x)
    if skip_x is not None:
        skip_x = Linear(x.shape[2], with_bias=False)(skip_x)
        x += Repeat(skip_x, x.shape[1] // skip_x.shape[1], axis=1)
    return GeLU(RMSBatchNorm(x) + Embedding(x.shape[2])(organism_index))

def output_pair(x: Array, organism_index: int) -> Array:
    x = (x + Swapaxes(x, 1, 2)) / 2.0 # Symmetrize.
    return GeLU(RMSNorm(x) + Embedding(128)(organism_index))

def model_embeddings(dna_sequence: Array, organism_index: int) -> tuple[Array, Array, Array]:
    trunk, intermediates = sequence_encoder(dna_sequence)
    trunk += Embedding(1535)(organism_index)
    trunk, pair_activations = transformer_tower(trunk)
    x = sequence_decoder(trunk, intermediates)
    embeddings_128bp = output_embedder(trunk, organism_index)
    embeddings_1bp = output_embedder(x, organism_index, embeddings_128bp)
    embeddings_pair = output_pair(pair_activations, organism_index)
    return embeddings_1bp, embeddings_128bp, embeddings_pair

def tracks_scaled_predictions(embeddings: Array[S, C], num_tracks: int) -> Array[S, num_tracks]:
    x = Linear(num_tracks)(x)
    scale = GetParameter('scale', (num_tracks,), init=0.0)
    return Softplus(x) * Softplus(scale)

def targets_scaling(targets: Array[S, C], track_means: Array[C], apply_squashing: bool) -> Array[S, C]:
    targets = targets / track_means
    if apply_squashing: # Applied RNA-seq tracks only.
        targets = targets ** 0.75
    return Where(targets > 10.0, 2 * Sqrt(x * 10.0) - 10.0, targets)

def predictions_scaling(x: Array[S, C], track_means: Array[C], apply_squashing: bool) -> Array[S, C]:
    x = Where(x > 10.0, (x + 10.0) ** 2 / (4 * 10.0), x)
    if apply_squashing:
        x = x ** (1.0 / 0.75)
    return x * track_means

def multinomial_loss(x: Array[S, C], targets: Array[S, C], multinomial_resolution: int) -> Array[]:
    x = x.reshape((-1, multinomial_resolution, x.shape[-1]))
    targets = targets.reshape((-1, multinomial_resolution, targets.shape[-1]))
    sum_pred = Sum(x, axis=1, keepdims=True)
    sum_target = Sum(targets, axis=1, keepdims=True)
    poisson_loss = Sum(sum_pred - sum_target * Log(sum_pred + 1e-7))
    multinomial_prob = x / (sum_pred + 1e-7)
    positional_loss = Sum(-targets * Log(multinomial_prob + 1e-7))
    return (poisson_loss / multinomial_resolution + 5.0 * positional_loss)

def tissue_scaled_rope(x: Array[S, 512], indices: Array[P]) -> Array[P, num_tissues, 512]:
    x = x[indices, :]
    scale = GetParameter('scale', (num_tissues, 512))
    offset = GetParameter('offset', (num_tissues, 512))
    x = scale[None, :, :] * x[:, None, :] + offset[None, :, :]
    return apply_rope(x, max_position=2 ** 20, positions=indices)

def splice_junctions(x: Array[S, C], donor_indices: Array[D], acceptor_indices: Array[A]) -> Array[D, A, N_tissues]:
    x = Linear(512)(x)
    donor_embedding = tissue_scaled_rope(x, donor_indices)
    acceptor_embedding = tissue_scaled_rope(x, acceptor_indices)
    return Softplus(Einsum('dtk,atk->dat', donor_embedding, acceptor_embedding))

def soft_clip(x: Array) -> Array:
    return Where(x > 10.0, 2 * Sqrt(x * 10.0) - 10.0, x)

def multinomial_cross_entropy(x: Array[D, A, N_tissues], targets: Array[D, A, N_tissues], axis: int) -> Array[]:
    pred_ratios = (x + 1e-7) / (x + 1e-7).sum(axis=axis, keepdims=True)
    target_ratios = (targets + 1e-7) / (targets + 1e-7).sum(axis=axis, keepdims=True)
    return - (targets * Log(pred_ratios)).sum()

def poisson_loss(x: Array[D, A, N_tissues], targets: Array[D, A, N_tissues], axis: int) -> Array[]:
    sum_pred = x.sum(axis=axis)
    sum_targets = soft_clip(targets.sum(axis=axis))
    return (sum_pred - sum_targets * Log(sum_pred + 1e-7)).sum()

def junctions_loss(x: Array[D, A, N_tissues], targets: Array[D, A, N_tissues]) -> Array[]:
    ratios_loss = (multinomial_cross_entropy(x, targets, axis=0) + multinomial_cross_entropy(x, targets, axis=1))
    counts_loss = (poisson_loss(x, targets, axis=0) + poisson_loss(x, targets, axis=1))
    return 0.2 * ratios_loss + 0.04 * counts_loss
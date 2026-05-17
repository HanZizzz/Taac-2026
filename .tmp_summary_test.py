from model import PCVRHyFormer

user_int_feature_specs = [(1, 0, 1)]
item_int_feature_specs = [(2, 0, 1)]
seq_vocab_sizes = {
    'seq_a': [2, 2],
    'seq_b': [2, 2],
    'seq_c': [2, 2],
    'seq_d': [2, 2],
}
user_ns_groups = [[0]]
item_ns_groups = [[0]]

model = PCVRHyFormer(
    user_int_feature_specs=user_int_feature_specs,
    item_int_feature_specs=item_int_feature_specs,
    user_dense_dim=1,
    item_dense_dim=1,
    seq_vocab_sizes=seq_vocab_sizes,
    user_ns_groups=user_ns_groups,
    item_ns_groups=item_ns_groups,
    engineered_dense_dim=66,
    seq_engineered_dim=13,
    engineered_summary_mode='core',
    engineered_summary_topk=16,
    engineered_summary_bottleneck_dim=8,
    engineered_summary_gate=True,
    d_model=16,
    emb_dim=8,
    num_queries=1,
    num_hyformer_blocks=1,
    num_heads=4,
    seq_encoder_type='transformer',
    hidden_mult=2,
    dropout_rate=0.0,
    seq_top_k=4,
    seq_causal=False,
    action_num=1,
    num_time_buckets=5,
    rank_mixer_mode='ffn_only',
    use_rope=False,
    emb_skip_threshold=0,
    seq_id_threshold=10000,
    ns_tokenizer_type='rankmixer',
    user_ns_tokens=1,
    item_ns_tokens=1,
)
print('dim', model.engineered_summary_input_dim)
print('idx', model.engineered_summary_indices)
print('gate', model.engineered_gate is not None)

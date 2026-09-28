def map_from_unilm_checkpoint(
    wavlm_state_dict: dict,
    num_layers: int = 22,
):
    # in case there are annoying prefixes
    def get_key(expected_suffix):
        for actual_key in wavlm_state_dict.keys():
            if actual_key.endswith(expected_suffix):
                return actual_key
        return None

    mapped_sd = {}

    # cnn
    for i in range(7):  # 7 conv layers
        conv_w = get_key(f"feature_extractor.conv_layers.{i}.0.weight")
        if conv_w:
            mapped_sd[f"feature_extractor.conv_layers.{i}.conv.weight"] = (
                wavlm_state_dict[conv_w]
            )

        # WavLM puts the norm at index 2 (or 2.1 if wrapped) in its sequential block
        norm_w = get_key(f"feature_extractor.conv_layers.{i}.2.weight") or get_key(
            f"feature_extractor.conv_layers.{i}.2.1.weight"
        )
        norm_b = get_key(f"feature_extractor.conv_layers.{i}.2.bias") or get_key(
            f"feature_extractor.conv_layers.{i}.2.1.bias"
        )

        if norm_w and norm_b:
            mapped_sd[f"feature_extractor.conv_layers.{i}.layer_norm.weight"] = (
                wavlm_state_dict[norm_w]
            )
            mapped_sd[f"feature_extractor.conv_layers.{i}.layer_norm.bias"] = (
                wavlm_state_dict[norm_b]
            )

    # feature proj
    proj_w = get_key("post_extract_proj.weight")
    if proj_w:
        mapped_sd["feature_projection.projection.weight"] = wavlm_state_dict[proj_w]
        mapped_sd["feature_projection.projection.bias"] = wavlm_state_dict[
            get_key("post_extract_proj.bias")
        ]

    # top-level layer norm
    for k in wavlm_state_dict.keys():
        if (
            k.endswith("layer_norm.weight")
            and "encoder" not in k
            and "conv_layers" not in k
            and "attn" not in k
        ):
            mapped_sd["feature_projection.layer_norm.weight"] = wavlm_state_dict[k]
            mapped_sd["feature_projection.layer_norm.bias"] = wavlm_state_dict[
                k.replace(".weight", ".bias")
            ]
            break

    # pos conv
    mapped_sd["pos_conv_embed.conv.bias"] = wavlm_state_dict[get_key("pos_conv.0.bias")]

    # check for weight_g/v (standard WavLM Large checkpoints)
    pos_conv_g = get_key("pos_conv.0.weight_g")
    pos_conv_v = get_key("pos_conv.0.weight_v")

    if pos_conv_g and pos_conv_v:
        mapped_sd["pos_conv_embed.conv.weight_g"] = wavlm_state_dict[pos_conv_g]
        mapped_sd["pos_conv_embed.conv.weight_v"] = wavlm_state_dict[pos_conv_v]
    else:
        mapped_sd["pos_conv_embed.conv.parametrizations.weight.original0"] = (
            wavlm_state_dict[get_key("pos_conv.0.parametrizations.weight.original0")]
        )
        mapped_sd["pos_conv_embed.conv.parametrizations.weight.original1"] = (
            wavlm_state_dict[get_key("pos_conv.0.parametrizations.weight.original1")]
        )

    # transformer encoder layers
    for i in range(num_layers):
        tgt_pfx = f"layers.{i}"
        src_pfx = f"layers.{i}"

        mapped_sd[f"{tgt_pfx}.attention.q_proj.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.q_proj.weight")
        ]
        mapped_sd[f"{tgt_pfx}.attention.k_proj.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.k_proj.weight")
        ]
        mapped_sd[f"{tgt_pfx}.attention.v_proj.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.v_proj.weight")
        ]

        mapped_sd[f"{tgt_pfx}.attention.q_proj.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.q_proj.bias")
        ]
        mapped_sd[f"{tgt_pfx}.attention.k_proj.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.k_proj.bias")
        ]
        mapped_sd[f"{tgt_pfx}.attention.v_proj.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.v_proj.bias")
        ]

        # out proj
        mapped_sd[f"{tgt_pfx}.attention.o_proj.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.out_proj.weight")
        ]
        mapped_sd[f"{tgt_pfx}.attention.o_proj.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.out_proj.bias")
        ]

        # gru
        mapped_sd[f"{tgt_pfx}.attention.gru_rel_pos_const"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.grep_a")
        ]
        mapped_sd[f"{tgt_pfx}.attention.gru_rel_pos_linear.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.grep_linear.weight")
        ]
        mapped_sd[f"{tgt_pfx}.attention.gru_rel_pos_linear.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn.grep_linear.bias")
        ]

        # relative attn bias in the absolute first layer only
        if i == 0:
            mapped_sd[f"{tgt_pfx}.attention.rel_attn_embed.weight"] = wavlm_state_dict[
                get_key(f"{src_pfx}.self_attn.relative_attention_bias.weight")
            ]

        # layernorm
        mapped_sd[f"{tgt_pfx}.layer_norm.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn_layer_norm.weight")
        ]
        mapped_sd[f"{tgt_pfx}.layer_norm.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.self_attn_layer_norm.bias")
        ]

        mapped_sd[f"{tgt_pfx}.final_layer_norm.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.final_layer_norm.weight")
        ]
        mapped_sd[f"{tgt_pfx}.final_layer_norm.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.final_layer_norm.bias")
        ]

        # FFN
        mapped_sd[f"{tgt_pfx}.feed_forward.intermediate_dense.weight"] = (
            wavlm_state_dict[get_key(f"{src_pfx}.fc1.weight")]
        )
        mapped_sd[f"{tgt_pfx}.feed_forward.intermediate_dense.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.fc1.bias")
        ]
        mapped_sd[f"{tgt_pfx}.feed_forward.output_dense.weight"] = wavlm_state_dict[
            get_key(f"{src_pfx}.fc2.weight")
        ]
        mapped_sd[f"{tgt_pfx}.feed_forward.output_dense.bias"] = wavlm_state_dict[
            get_key(f"{src_pfx}.fc2.bias")
        ]

    return mapped_sd

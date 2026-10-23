import csv
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch

_DECODER_NAMES = {
    "intermediate_prediction_heads": "intermediate_head",
    "b3_bridge_decoder": "feature_propagator",
    "trfl": "feature_propagator",
    "task_state_initializers": "preliminary_decoder",
    "task_attention_blocks": "atten_production",
    "cross_task_attention_fusion": "atten_fusion",
    "backbone_scale_projections": "scale_embed",
}
_BRIDGE_NAMES = {
    "stage_output_norms": "norm_mts",
    "stage_output_projections": "redu_chan",
    "b3_stages": "trfl_stages",
    "task_aggregation_projections": "mt_proj",
    "task_state_packaging": "mix_proj",
    "shared_reference_projection": "back_feature_embed",
    "task_state_projections": "patch_embed",
    "posterior_bridge_blocks": "blocks",
    "task_refinement": "reconv",
    "previous_bridge_projection": "reduce_channel",
    "evidence_extractors": "task_attn",
    "precision_field_estimator": "pfe",
    "prior_precision_head": "prior_logit",
    "posterior_correction_logit": "eta_refine",
    "refinement_blocks": "re_blocks",
    "contractive_dispatch_operators": "dispatch",
    "bridge_to_task_projection": "ref_to_task",
    "posterior_bridge_operator": "pbo",
}


def public_key(key):
    if key.startswith("module."):
        key = key[7:]
    elif key == "module":
        return ""
    parts = key.split(".")
    if parts[0] == "prediction_heads":
        parts[0] = "heads"
        if len(parts) > 2 and parts[2] == "projection":
            parts[2] = "linear_pred"
    if len(parts) > 1 and parts[0] == "multi_task_decoder":
        parts[1] = _DECODER_NAMES.get(parts[1], parts[1])
        if parts[1] == "feature_propagator":
            parts[2:] = [_BRIDGE_NAMES.get(part, part) for part in parts[2:]]
    return ".".join(parts)


def read_checkpoint(path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a state dict or a model/state_dict container")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Checkpoint state dict is empty or invalid")
    if any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items()):
        raise TypeError("State dict must map string names to tensors")
    metadata = {key: checkpoint.get(key) for key in ("epoch", "iter_count")}
    return state, metadata


def convert_state_dict(state):
    converted = OrderedDict()
    for old_key, value in state.items():
        new_key = public_key(old_key)
        if new_key in converted:
            raise ValueError("Checkpoint key collision: " + new_key)
        converted[new_key] = value
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        converted._metadata = OrderedDict(
            (public_key(key), value) for key, value in metadata.items()
            if not (key == "" and "module" in metadata)
        )
    return converted


def write_key_mapping(path, source, target, loaded=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint_key", "public_key", "shape", "dtype",
                         "same_source_tensor", "loaded_tensor_equal"])
        for old_key, value in source.items():
            new_key = public_key(old_key)
            equal = "not_checked" if loaded is None else torch.equal(value, loaded[new_key])
            writer.writerow([old_key, new_key, "x".join(map(str, value.shape)),
                             str(value.dtype), target[new_key] is value, equal])


def load_checkpoint_strict(model, path, mapping_path=None):
    source, metadata = read_checkpoint(path)
    converted = convert_state_dict(source)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    mismatched = [key for key in expected.keys() & converted.keys()
                  if expected[key].shape != converted[key].shape
                  or expected[key].dtype != converted[key].dtype]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint does not match this configuration. "
            "No keys were discarded and no tensors were resized.\n"
            f"Missing ({len(missing)}): {missing[:20]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:20]}\n"
            f"Shape/dtype mismatch ({len(mismatched)}): {mismatched[:20]}"
        )
    for key, tensor in converted.items():
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite checkpoint tensor: " + key)
    model.load_state_dict(converted, strict=True)
    loaded = model.state_dict()
    for key, tensor in converted.items():
        if not torch.equal(tensor, loaded[key].detach().cpu()):
            raise RuntimeError("Loaded tensor differs from checkpoint: " + key)
    if mapping_path is not None:
        write_key_mapping(mapping_path, source, converted, loaded)
    metadata.update({
        "state_dict_keys": len(converted),
        "renamed_keys": sum(public_key(key) != key for key in source),
        "strict_load": True,
        "all_loaded_tensors_equal": True,
    })
    return metadata

import argparse
import ast
import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def source_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [item for item in node.body
                         if not (isinstance(item, ast.Expr)
                                 and isinstance(item.value, ast.Constant)
                                 and isinstance(item.value.value, str))]
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description="Compare model outputs with the reference implementation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path,
                        help="Use the first validation image instead of synthetic input.")
    args = parser.parse_args()

    import torch
    from utils.checkpoint import load_checkpoint_strict
    from utils.common_config import get_model
    from utils.config import create_config

    root = Path(__file__).resolve().parents[1]
    reference = args.reference_root.resolve()

    for name, original_name in (("models/internimage.py", "models/intern_image.py"),
                                ("models/transformers/vit.py", "models/transformers/vit.py")):
        reference_source = (reference / original_name).read_text(encoding="utf-8")
        if original_name == "models/intern_image.py":
            reference_source = reference_source.replace(
                "from mmseg.models.builder import BACKBONES\n", "").replace(
                "@BACKBONES.register_module()\n", "")
        if source_ast((root / name).read_text(encoding="utf-8")) != source_ast(reference_source):
            raise RuntimeError("Backbone source differs: " + name)
    p = create_config(str(args.config), {"run_mode": "infer"})
    public = get_model(p, pretrained=False)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_checkpoint_strict(public, args.checkpoint,
                                      output_dir / "checkpoint_keys.csv")

    package_name = "b3_original_models"
    spec = importlib.util.spec_from_file_location(
        package_name, reference / "models/__init__.py",
        submodule_search_locations=[str(reference / "models")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    original_model = importlib.import_module(package_name + ".b3_model")
    original_decoder = importlib.import_module(package_name + ".transformers.b3_decoder")
    heads = torch.nn.ModuleDict({
        task: original_decoder.MLPHead(p.final_embed_dim, p.TASKS.NUM_OUTPUT[task])
        for task in p.TASKS.NAMES
    })
    original = original_model.B3Net(copy.deepcopy(p), copy.deepcopy(public.backbone),
                                   p.backbone_channels, heads)
    original.load_state_dict(public.state_dict(), strict=True)
    result = {"config": str(args.config), "checkpoint": str(args.checkpoint),
              "backbone": p.backbone, "checkpoint_loading": metadata,
              "dataset_accuracy_evaluated": False,
              "input_kind": "fixed synthetic normal tensor, seed 0",
              "rtol": 1e-6, "atol": 1e-6, "outputs": {}}
    if not torch.cuda.is_available():
        raise RuntimeError("Full forward comparison requires CUDA/DCNv3.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    if args.dataset_root is None:
        inputs = torch.randn(1, 3, *p.TEST.SCALE).cuda()
    else:
        from utils.common_config import get_test_dataset, get_transformations
        key = {"NYUD": "NYUD_MT", "PASCALContext": "PASCALContext"}[p.val_db_name]
        p.db_paths[key] = str(args.dataset_root.resolve())
        _, transforms = get_transformations(p)
        sample = get_test_dataset(p, transforms)[0]
        inputs = sample["image"].unsqueeze(0).cuda()
        result["input_kind"] = "first validation image with original transforms"
        result["image_id"] = sample["meta"]["img_name"]
    result["input_shape"] = list(inputs.shape)
    result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0), "tf32": False}
    public.cuda().eval()
    original.cuda().eval()
    with torch.no_grad():
        public_outputs = public(inputs)
        original_outputs = original(inputs)
    for name in p.TASKS.NAMES:
        for group in ("final", "intermediate"):
            a = public_outputs[name] if group == "final" else public_outputs["inter_preds"][name]
            b = original_outputs[name] if group == "final" else original_outputs["inter_preds"][name]
            result["outputs"][group + "." + name] = {
                "shape": list(a.shape),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                "bitwise_equal": torch.equal(a, b),
                "max_absolute_difference": float((a - b).abs().max()),
                "within_tolerance": torch.allclose(a, b, rtol=1e-6, atol=1e-6),
            }
    result["passed"] = all(item["finite"] and item["within_tolerance"]
                           for item in result["outputs"].values())
    serialized = json.dumps(result, indent=2, allow_nan=False)
    (output_dir / "forward_comparison.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not result["passed"]:
        raise RuntimeError("Original/public forward comparison failed")

if __name__ == "__main__":
    main()

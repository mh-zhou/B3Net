import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.checkpoint import convert_state_dict, load_checkpoint_strict, public_key


def main():
    key = "multi_task_decoder.trfl.trfl_stages.2.blocks.0.pfe.centers"
    assert public_key("module." + key) == key.replace(".trfl.", ".feature_propagator.")
    canonical = "multi_task_decoder.b3_bridge_decoder.b3_stages.2.posterior_bridge_blocks.0.precision_field_estimator.centers"
    assert public_key(canonical) == public_key(key)
    assert public_key("prediction_heads.semseg.projection.weight") == "heads.semseg.linear_pred.weight"
    tensor = torch.tensor([1.0, 2.0])
    assert convert_state_dict({key: tensor})[public_key(key)] is tensor
    try:
        convert_state_dict({"module.weight": tensor, "weight": tensor})
    except ValueError:
        pass
    else:
        raise AssertionError("Key collision was not rejected")
    with tempfile.TemporaryDirectory(prefix="b3_checkpoint_test_") as directory:
        root = Path(directory)
        model = torch.nn.Linear(2, 1)
        state = {"module.weight": torch.tensor([[1.0, 2.0]]), "module.bias": torch.tensor([3.0])}
        path = root / "test.pth"
        torch.save({"model": state, "epoch": 2}, path)
        result = load_checkpoint_strict(model, path, root / "keys.csv")
        assert result["strict_load"] and result["state_dict_keys"] == 2
        assert torch.equal(model(torch.ones(1, 2)), torch.tensor([[6.0]]))
        for invalid in ({"weight": tensor}, {"weight": torch.ones(1, 2, dtype=torch.float64), "bias": torch.ones(1)}):
            torch.save(invalid, path)
            try:
                load_checkpoint_strict(model, path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Invalid state dict was accepted")
        torch.save({"weight": torch.tensor([[float("nan"), 1.0]]), "bias": torch.ones(1)}, path)
        try:
            load_checkpoint_strict(model, path)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite tensor was accepted")
    print("Checkpoint regression checks passed.")

if __name__ == "__main__":
    main()

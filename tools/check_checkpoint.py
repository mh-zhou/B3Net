import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.checkpoint import convert_state_dict, read_checkpoint, write_key_mapping


def main():
    parser = argparse.ArgumentParser(description="Check model and checkpoint compatibility")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, metadata = read_checkpoint(args.checkpoint)
    converted = convert_state_dict(source)
    write_key_mapping(args.output, source, converted)
    metadata.update({
        "state_dict_keys": len(source),
        "mapping_rows": len(converted),
        "tensor_values_changed": False,
        "model_strict_load": "not_run",
        "validation_inference": "not_run",
    })
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()

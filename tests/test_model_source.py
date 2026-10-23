import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify_model import source_ast


def check_model_names():
    root = Path(__file__).resolve().parents[1]
    expected = {
        'models/b3net.py': {'B3Net'},
        'models/transformers/decoder.py': {'B3Decoder', 'ConvBlock', 'MLPHead'},
        'models/transformers/operators.py': {
            'PrecisionFieldEstimator', 'PosteriorBridgeOperator',
            'ContractiveDispatchOperator', 'PosteriorBridgeBlock',
            'TaskRefinementBlock', 'TaskRefinementStage', 'B3Stage', 'B3FeaturePropagator',
        },
    }
    for name, required in expected.items():
        tree = ast.parse((root / name).read_text(encoding='utf-8'))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        assert required <= classes, name
    for name in (*expected, 'models/__init__.py', 'utils/common_config.py'):
        source = (root / name).read_text(encoding='utf-8')
        for obsolete in ('B3Baseline', 'TransformerDecoder', 'CAStage', 'ContractiveRefinement'):
            assert obsolete not in source, (name, obsolete)


def main():
    original = '"""Module description."""\n# Source note\ndef forward(x):\n    """Function description."""\n    return x + 1\n'
    cleaned = 'def forward(x):\n    return x + 1\n'
    changed = 'def forward(x):\n    return x + 2\n'
    assert source_ast(original) == source_ast(cleaned)
    assert source_ast(original) != source_ast(changed)
    assert source_ast('x = 1\n"Unused section"\ny = 2') == source_ast('x = 1\ny = 2')
    assert source_ast('x = "active"') != source_ast('x = "changed"')
    assert source_ast('class Model:\n    """Class description."""\n    value = 1') == source_ast('class Model:\n    value = 1')
    check_model_names()
    print('Model-source comparison and naming checks passed.')

if __name__ == '__main__':
    main()

"""Measure supplied model shapes without allocating parameter storage or using CUDA."""
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from workloads import mlops, pytorch
from workloads.mlops._qwen_moe.model import QwenMoE

rows = []
for name in ('Llama3', 'Qwen35', 'OLMoE', 'Qwen3MoE', 'Qwen35MoE'):
    model_type = getattr(mlops, name)
    config_type = getattr(mlops, name + 'Config')
    presets = ('numerical', 'throughput') if hasattr(config_type, 'numerical') else ('throughput',)
    for preset in presets:
        config = getattr(config_type, preset)()
        with torch.device('meta'):
            model = model_type(config)
        count = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if hasattr(pytorch, name):
            with torch.device('meta'):
                reference = getattr(pytorch, name)(config)
            assert sum(p.numel() for p in reference.parameters()) == count
        rows.append(dict(model=name, preset=preset, parameters=count, trainable=trainable,
                         config=asdict(config), all_parameters_meta=all(p.is_meta for p in model.parameters())))
record = {'shadowspill_revision': '5f8dc13c2cd2ec8d2f56ae6580f02d471b3f9566',
          'counting': 'unique Parameter objects, includes embedding/output head, excludes buffers; local non-EP model',
          'models': rows, 'qwen_moe_constructor': str(inspect.signature(QwenMoE))}
path = ROOT / 'docs/internal/plans/lora_models_1008/evidence/model-catalog.json'
path.write_text(json.dumps(record, indent=2) + '\n')
for row in rows:
    print(f"{row['model']:9} {row['preset']:10} total={row['parameters']:,} trainable={row['trainable']:,}")
print(path)

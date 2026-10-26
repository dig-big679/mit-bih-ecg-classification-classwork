"""方案 B 的结构、梯度、命令参数和保存加载检查，不运行正式训练。"""
import io
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models import ECGClassifier, build_model
from train import parse_args


class NetworkBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_default_structure_and_parameter_count(self):
        model = ECGClassifier()
        self.assertEqual(model.recurrent.input_size, 300)
        self.assertEqual(model.recurrent.hidden_size, 128)
        self.assertEqual(model.recurrent.num_layers, 2)
        self.assertEqual(model.recurrent.dropout, 0.2)
        linears = [(layer.in_features, layer.out_features)
                   for layer in model.classifier if isinstance(layer, nn.Linear)]
        self.assertEqual(linears, [(128, 128), (128, 64), (64, 5)])
        self.assertEqual([layer.p for layer in model.classifier if isinstance(layer, nn.Dropout)],
                         [0.5, 0.5])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 377349)
        self.assertEqual(sum(p.numel() for p in model.recurrent.parameters()), 352256)
        self.assertEqual(sum(p.numel() for p in model.classifier.parameters()), 25093)

    def test_optimizer_updates_recurrent_and_both_hidden_layers(self):
        model = ECGClassifier()
        inputs, targets = torch.randn(8, 1, 300), torch.arange(8) % 5
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        params = [model.recurrent.weight_ih_l0, model.recurrent.weight_ih_l1,
                  model.classifier[0].weight, model.classifier[3].weight]
        before = [param.detach().clone() for param in params]
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (8, 5))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(param.grad is not None and torch.isfinite(param.grad).all()
                            for param in params))
        optimizer.step()
        self.assertTrue(all(not torch.equal(old, param) for old, param in zip(before, params)))

    def test_saved_b_configuration_restores_exact_predictions(self):
        config = {"model_type": "lstm", "hidden_size": 128, "num_layers": 2,
            "recurrent_dropout": 0.2, "classifier_dropout": 0.5,
            "input_size": 300, "pooling": "last", "classifier_hidden_sizes": [128, 64]}
        model = build_model(config).eval()
        memory_file = io.BytesIO()
        torch.save({"model_config": config, "model_state": model.state_dict()}, memory_file)
        memory_file.seek(0)
        checkpoint = torch.load(memory_file, map_location="cpu", weights_only=True)
        restored = build_model(checkpoint["model_config"]).eval()
        restored.load_state_dict(checkpoint["model_state"], strict=True)
        inputs = torch.randn(8, 1, 300)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), restored(inputs))

    def test_cli_defaults_select_b_and_keep_single_layer_usable(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_args()
        self.assertEqual((args.hidden_size, args.num_layers), (128, 2))
        self.assertEqual(args.classifier_hidden_sizes, [128, 64])
        self.assertEqual(args.recurrent_dropout, 0.2)
        self.assertEqual(args.classifier_dropout, 0.5)
        with patch.object(sys, "argv", ["train.py", "--num-layers", "1"]):
            single = parse_args()
        self.assertEqual(single.recurrent_dropout, 0.0)


if __name__ == "__main__":
    unittest.main()

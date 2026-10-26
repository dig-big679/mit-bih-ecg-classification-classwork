"""只检查输入组织、梯度和 checkpoint 兼容性，不运行正式训练。"""
import unittest

import numpy as np
import torch

from data import HeartbeatDataset, PreparedData
from models import ECGClassifier, build_model
from train import make_loader


class InputLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.data = PreparedData(
            np.arange(8 * 300, dtype=np.float32).reshape(8, 300, 1) / 300,
            np.arange(8, dtype=np.int64) % 5,
            np.array(["100"] * 8), np.array(["100"] * 8),
            np.arange(8), {})

    def test_classroom_loader_preserves_samples_and_labels(self):
        inputs, labels = next(iter(make_loader(self.data, np.arange(8), 8, torch.device("cpu"))))
        self.assertEqual(tuple(inputs.shape), (8, 1, 300))
        np.testing.assert_array_equal(inputs[:, 0].numpy(), self.data.x[:, :, 0])
        np.testing.assert_array_equal(labels.numpy(), self.data.y)

    def test_previous_layout_is_explicitly_available(self):
        sample, _ = HeartbeatDataset(self.data, np.arange(8), input_size=1)[0]
        self.assertEqual(tuple(sample.shape), (300, 1))

    def test_classroom_models_forward_backward_and_batch_independence(self):
        inputs = torch.randn(8, 1, 300)
        labels = torch.arange(8) % 5
        for kind in ("rnn", "lstm"):
            with self.subTest(model=kind):
                model = ECGClassifier(kind, hidden_size=8, classifier_dropout=0).eval()
                self.assertEqual(model.recurrent.input_size, 300)
                with torch.no_grad():
                    logits = model(inputs)
                    individual = torch.cat([model(row[None]) for row in inputs])
                self.assertEqual(tuple(logits.shape), (8, 5))
                torch.testing.assert_close(logits, individual, atol=1e-6, rtol=1e-5)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                before = model.recurrent.weight_ih_l0.detach().clone()
                loss = torch.nn.functional.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(loss))
                self.assertFalse(torch.equal(before, model.recurrent.weight_ih_l0))
                with self.assertRaises(ValueError):
                    model(torch.randn(8, 300, 1))

    def test_new_config_roundtrip(self):
        config = {"model_type": "lstm", "hidden_size": 8, "input_size": 300}
        original = build_model(config).eval()
        restored = build_model(config).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        inputs = torch.randn(4, 1, 300)
        torch.testing.assert_close(original(inputs), restored(inputs))

    def test_legacy_configs_and_pooling_keep_original_structure(self):
        for pooling in ("last", "meanmax"):
            with self.subTest(pooling=pooling):
                config = {"model_type": "lstm", "hidden_size": 8, "pooling": pooling}
                legacy = ECGClassifier(**config, input_size=1, num_layers=1,
                    recurrent_dropout=0.0, classifier_dropout=0.3,
                    classifier_hidden_sizes=[64]).eval()
                restored = build_model(config).eval()
                restored.load_state_dict(legacy.state_dict(), strict=True)
                self.assertEqual(restored.input_size, 1)
                inputs = torch.randn(4, 300, 1)
                torch.testing.assert_close(legacy(inputs), restored(inputs))
                self.assertNotIn("input_size", config)


if __name__ == "__main__":
    unittest.main()

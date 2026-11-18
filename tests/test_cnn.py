"""Unit tests for CNN model architectures (Phase 2).

Owner: Vipul Sharma (@vs-code06)
Runs via: .venv/bin/python3 -m unittest discover -s tests -v
"""

import unittest
import torch
import torch.nn as nn

from dl.models.cnn import SimpleCNN, DeepCNN, count_parameters


class CNNArchitectureTests(unittest.TestCase):
    """Test suite for SimpleCNN and DeepCNN architectures."""

    def setUp(self):
        torch.manual_seed(42)

    def test_simple_cnn_forward_shape(self):
        """Verify SimpleCNN produces (B, 10) output for variable batch sizes."""
        model = SimpleCNN()
        for batch_size in (1, 4, 16, 64):
            x = torch.randn(batch_size, 1, 28, 28)
            out = model(x)
            self.assertEqual(out.shape, (batch_size, 10))
            self.assertTrue(torch.isfinite(out).all())

    def test_deep_cnn_forward_shape(self):
        """Verify DeepCNN produces (B, 10) output for variable batch sizes."""
        model = DeepCNN()
        for batch_size in (1, 4, 16, 64):
            x = torch.randn(batch_size, 1, 28, 28)
            out = model(x)
            self.assertEqual(out.shape, (batch_size, 10))
            self.assertTrue(torch.isfinite(out).all())

    def test_parameter_counts(self):
        """Verify model parameter counts match expected budgets (~410K-440K)."""
        simple = SimpleCNN()
        deep = DeepCNN()

        simple_params = count_parameters(simple)
        deep_params = count_parameters(deep)

        # SimpleCNN: ~421K
        self.assertGreater(simple_params, 400_000)
        self.assertLess(simple_params, 450_000)

        # DeepCNN: ~390K (smaller linear head because 3 pooling layers yield 3x3 instead of 7x7)
        self.assertGreater(deep_params, 370_000)
        self.assertLess(deep_params, 440_000)

    def test_gradient_flow_simple_cnn(self):
        """Verify gradients flow back to all layers in SimpleCNN during backprop."""
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertGreater(
                    param.grad.abs().sum().item(), 0.0, f"Zero gradient for {name}"
                )

    def test_gradient_flow_deep_cnn(self):
        """Verify gradients flow back to all layers in DeepCNN during backprop."""
        model = DeepCNN()
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertGreater(
                    param.grad.abs().sum().item(), 0.0, f"Zero gradient for {name}"
                )

    def test_ablation_toggles_no_batchnorm(self):
        """Verify DeepCNN cleanly supports turning off BatchNorm for ablations."""
        model_no_bn = DeepCNN(use_batchnorm=False)
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model_no_bn.modules())
        self.assertFalse(has_bn, "BatchNorm2d found when use_batchnorm=False")

        x = torch.randn(4, 1, 28, 28)
        out = model_no_bn(x)
        self.assertEqual(out.shape, (4, 10))

    def test_ablation_toggles_no_dropout(self):
        """Verify DeepCNN cleanly supports turning off dropout for ablations."""
        model_no_drop = DeepCNN(conv_dropout=0.0, fc_dropout=0.0)
        has_drop = any(
            isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model_no_drop.modules()
        )
        self.assertFalse(has_drop, "Dropout layer found when dropout=0.0")

        x = torch.randn(4, 1, 28, 28)
        out = model_no_drop(x)
        self.assertEqual(out.shape, (4, 10))

    def test_eval_mode_determinism(self):
        """Verify that in eval() mode, forward pass is completely deterministic."""
        model = DeepCNN(conv_dropout=0.5, fc_dropout=0.5)
        model.eval()

        x = torch.randn(8, 1, 28, 28)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        self.assertTrue(torch.allclose(out1, out2))


if __name__ == "__main__":
    unittest.main()


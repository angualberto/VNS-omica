import torch

from vnn_pixelwise_mammo import VNNPixelMammo


def test_shape_cpu():
    model = VNNPixelMammo(in_ch=8).cpu()
    x = torch.randn(4, 8, 256, 256)
    y = model(x)
    assert y.shape == (4, 1, 256, 256)


def test_backward_cpu():
    model = VNNPixelMammo(in_ch=8).cpu()
    x = torch.randn(2, 8, 256, 256)
    mask = torch.rand(2, 1, 256, 256)
    loss = torch.nn.BCEWithLogitsLoss()(model(x), mask)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_shape_gpu_if_available():
    if not torch.cuda.is_available():
        return
    model = VNNPixelMammo(in_ch=8).cuda()
    x = torch.randn(4, 8, 256, 256, device="cuda")
    y = model(x)
    assert y.shape == (4, 1, 256, 256)

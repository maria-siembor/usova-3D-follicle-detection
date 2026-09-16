"""Optional inference adapter for the Kaggle-trained 2D follicle U-Net."""

from pathlib import Path

import numpy as np
from scipy import ndimage


_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "kaggle_outputs" / "follicle_unet" / "follicle_unet.pt"


def _build_model(torch):
    nn = torch.nn

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.net(value)

    class UNet(nn.Module):
        def __init__(self, base=32):
            super().__init__()
            self.enc1, self.enc2 = ConvBlock(1, base), ConvBlock(base, base * 2)
            self.enc3, self.enc4 = ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(base * 8, base * 16)
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.dec4, self.dec3 = ConvBlock(base * 16, base * 8), ConvBlock(base * 8, base * 4)
            self.dec2, self.dec1 = ConvBlock(base * 4, base * 2), ConvBlock(base * 2, base)
            self.out_conv = nn.Conv2d(base, 1, 1)

        def forward(self, value):
            e1 = self.enc1(value)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            bottleneck = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out_conv(d1)

    return UNet()


def predict_labels(volume, spacing, checkpoint_path=_DEFAULT_CHECKPOINT, threshold=0.5):
    """Predict a 3D binary mask and relabel connected follicle components."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Deep learning requires PyTorch. Install requirements-dl.txt.") from error

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"U-Net checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    image_size = int(checkpoint.get("img_size", 256))
    model = _build_model(torch)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predicted_slices = []
    with torch.no_grad():
        for image in volume:
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            image_tensor = image_tensor[None, None]
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
            prediction = torch.sigmoid(model(image_tensor))[0, 0]
            prediction = torch.nn.functional.interpolate(
                prediction[None, None], size=image.shape, mode="nearest"
            )[0, 0].numpy()
            predicted_slices.append(prediction > threshold)

    binary = np.stack(predicted_slices)
    labels, _ = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 1))
    voxel_volume = float(np.prod(spacing))
    minimum_voxels = (4 / 3 * np.pi * 1.0 ** 3) / voxel_volume
    sizes = ndimage.sum(binary, labels, range(1, labels.max() + 1))
    keep = [index + 1 for index, size in enumerate(sizes) if size >= minimum_voxels]
    return np.where(np.isin(labels, keep), labels, 0).astype(np.int32)

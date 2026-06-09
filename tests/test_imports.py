def test_public_convolution_imports():
    from ml_suite.models.convolution import (
        ConvBlock,
        ConditionedConvBlock,
        ConvNet,
        ConditionedConvNet,
    )

    assert ConvBlock is not None
    assert ConditionedConvBlock is not None
    assert ConvNet is not None
    assert ConditionedConvNet is not None


def test_public_unet_imports():
    from ml_suite.models.unet import UNet, ConditionedUNet

    assert UNet is not None
    assert ConditionedUNet is not None

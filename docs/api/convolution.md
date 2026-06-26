# Convolution models

`ml_suite.models.convolution` contains blocks and multi-stage backbones that work identically for 1D signals, 2D images, and 3D volumes. The `conv_dim` parameter is the only thing that changes between them.

**ConvNet vs UNet.** For classification and feature extraction, use `ConvNet`. For generation and reconstruction tasks where output spatial size needs to match input size, use `ConditionedUNet` from `models.unet`. The U-Net has skip connections that preserve spatial detail through the decoder. The ConvNet is encoder-only.

**ConvNet vs ConditionedConvNet.** Use `ConvNet` for pure discriminative tasks like classification. Use `ConditionedConvNet` when you need to inject a global signal, such as a class embedding or a diffusion timestep, into every convolutional layer via FiLM.

**Receptive field diagnostic.** `ConvNet` exposes a `print_receptive_field` method. When you are tuning `stage_channels` and `blocks_per_stage` and want to know how large a spatial region each output feature represents, call this method. It prints the cumulative receptive field after each layer so you can verify that the field covers your target structure size.

---

::: ml_suite.models.convolution.blocks

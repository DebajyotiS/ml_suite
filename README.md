# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/DebajyotiS/ml_suite/blob/coverage-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/ml\_suite/\_\_init\_\_.py                    |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/\_\_init\_\_.py             |        5 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/convolution/\_\_init\_\_.py |        2 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/convolution/blocks.py       |      310 |       28 |      144 |       18 |     89% |80, 135-\>137, 229, 231, 233, 274-\>276, 278, 285-286, 298-306, 325, 343, 345, 350, 365, 459, 461, 465, 626-627, 633-642, 685 |
| src/ml\_suite/models/generative/\_\_init\_\_.py  |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/generative/diffusion.py     |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/generative/drifting.py      |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/generative/flowmatching.py  |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/linear/\_\_init\_\_.py      |        3 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/linear/blocks.py            |      111 |        6 |       50 |        6 |     93% |101-102, 139-\>149, 151, 168, 170, 174 |
| src/ml\_suite/models/linear/vad.py               |       29 |        0 |        2 |        0 |    100% |           |
| src/ml\_suite/models/transformer/\_\_init\_\_.py |       11 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/transformer/attention.py    |      127 |        7 |       48 |        5 |     93% |16, 21, 84, 213, 215, 226, 268 |
| src/ml\_suite/models/transformer/blocks.py       |       52 |        0 |       24 |        0 |    100% |           |
| src/ml\_suite/models/transformer/conditioning.py |       88 |        6 |       52 |        3 |     94% |68-71, 98, 108 |
| src/ml\_suite/models/transformer/decoders.py     |       58 |        4 |       24 |        4 |     90% |33, 64, 97, 101 |
| src/ml\_suite/models/transformer/heads.py        |       48 |        1 |       18 |        1 |     97% |        65 |
| src/ml\_suite/models/transformer/models.py       |      119 |        4 |       28 |        4 |     95% |385, 491, 505, 533 |
| src/ml\_suite/models/transformer/pooling.py      |       44 |        7 |       24 |        1 |     82% |     64-71 |
| src/ml\_suite/models/transformer/positional.py   |       98 |        3 |       36 |        3 |     96% |90, 212, 237 |
| src/ml\_suite/models/transformer/presets.py      |       15 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/transformer/stacks.py       |       20 |        0 |        6 |        0 |    100% |           |
| src/ml\_suite/models/transformer/tokenization.py |       60 |        0 |       22 |        0 |    100% |           |
| src/ml\_suite/models/transformer/types.py        |        7 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/transformer/utils.py        |       51 |        0 |       28 |        0 |    100% |           |
| src/ml\_suite/models/unet/\_\_init\_\_.py        |        5 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/unet/attention.py           |      114 |        9 |       56 |        7 |     89% |29, 39, 41, 50-52, 54, 176, 181 |
| src/ml\_suite/models/unet/base.py                |      167 |       23 |       90 |       22 |     82% |144, 146, 150, 152, 154, 164, 166, 168, 173, 181, 183, 185, 187, 189, 191, 220, 326, 334, 345, 352, 381, 383, 412 |
| src/ml\_suite/models/unet/conditioning.py        |       87 |       16 |       52 |       11 |     81% |30, 32, 36, 38, 42, 68, 77-81, 95, 109, 114, 125, 129 |
| src/ml\_suite/models/unet/models.py              |       30 |        3 |        2 |        1 |     88% |164, 255-256 |
| src/ml\_suite/models/unet/stages.py              |       94 |       10 |       36 |        8 |     86% |33, 35, 127, 141, 174, 178, 180, 215, 268, 282 |
| src/ml\_suite/models/unet/types.py               |        9 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/models/unet/utils.py               |       47 |        8 |       18 |        6 |     75% |14, 69, 81-86, 89, 100, 106 |
| src/ml\_suite/utils/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/utils/activations.py               |       23 |        0 |       18 |        0 |    100% |           |
| src/ml\_suite/utils/conditioning.py              |       52 |        1 |       22 |        1 |     97% |        52 |
| src/ml\_suite/utils/numpy\_utils.py              |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/utils/runtime\_utils.py            |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/utils/samplers/\_\_init\_\_.py     |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/utils/samplers/ode.py              |        0 |        0 |        0 |        0 |    100% |           |
| src/ml\_suite/utils/samplers/sde.py              |        0 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                        | **1889** |  **136** |  **800** |  **101** | **91%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/DebajyotiS/ml_suite/coverage-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/DebajyotiS/ml_suite/blob/coverage-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DebajyotiS/ml_suite/coverage-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/DebajyotiS/ml_suite/blob/coverage-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FDebajyotiS%2Fml_suite%2Fcoverage-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/DebajyotiS/ml_suite/blob/coverage-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
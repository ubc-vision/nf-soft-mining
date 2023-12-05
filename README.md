# Accelerating Neural Field Training via Soft Mining

[![Project Page](https://img.shields.io/badge/-Project%20Page-blue?style=flat-square)](https://ubc-vision.github.io/nf-soft-mining/) [![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-orange?style=flat-square)](https://arxiv.org/abs/2312.00075)


## Overview
This repository contains the implementation and resources for our research paper ["Accelerating Neural Field Training via Soft Mining"](https://arxiv.org/abs/2312.00075). 

The paper presents a novel approach to accelerate the training of Neural Fields (NeRFs) by introducing a soft mining strategy. This strategy dynamically selects a subset of rays for each iteration, focusing on those that contribute most to the learning process based on the loss gradient. 

For a detailed understanding of the methodology, results, and more, please refer to the [full paper](https://ubc-vision.github.io/nf-soft-mining/).

<img src="./docs/static/videos/mic_good_video.gif" height=300>


## Installation

Please refer to [NerfAcc Repository](https://github.com/KAIR-BAIR/nerfacc) for installation instructions.

## Usage
After installing Nerfacc, make sure you have NeRF Synthetic or LLFF dataset downloaded. 

To run the baseline with **Uniform** Sampling:
``` python examples/train_ngp_nerf_prop.py --sampling_type uniform --data_root /path/to/your/dataset --scene scene_name ```

To run **LMC** sampling, simply run:
``` python examples/train_ngp_nerf_prop.py --sampling_type lmc --data_root /path/to/your/dataset --scene scene_name ```


## Acknowledgements

This project is built upon the work found in [**Nerfacc Repository**](https://github.com/KAIR-BAIR/nerfacc). Special thanks to all the contributors of the original repository for laying the groundwork that has enabled us to advance this initiative.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{kheradsoftmining2023,
        author    = {Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi},
        title     = {Accelerating Neural Field Training via Soft Mining},
        journal   = {Arxiv},
        year      = {2023},
        }
```



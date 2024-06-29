<p align="center">

  <h1 align="center">Accelerating Neural Field Training via Soft Mining</h1>
  <p align="center">
        <a href="https://shakibakh.github.io/">Shakiba Kheradmand</a>,
        <a href="http://drebain.com/"> Daniel Rebain</a>,
        <a href="https://hippogriff.github.io/"> Gopal Sharma</a>,
        <a href="https://wsunid.github.io/"> Weiwei Sun</a>,
        <a href="https://scholar.google.com/citations?user=1iJfq7YAAAAJ&hl=en"> Yang-Che Tseng</a>,
        <a href="http://www.hossamisack.com/">Hossam Isack</a>,
        <a href="https://abhishekkar.info/">Abhishek Kar</a>,
        <a href="https://taiya.github.io/">Andrea Tagliasacchi</a>,
        <a href="https://www.cs.ubc.ca/~kmyi/">Kwang Moo Yi</a>
  </p>
  <h2 align="center">CVPR 2024</h2>

  <h3 align="center"><a href="https://arxiv.org/abs/2312.00075">arXiv</a> | <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Kheradmand_Accelerating_Neural_Field_Training_via_Soft_Mining_CVPR_2024_paper.pdf">CVPR</a> | <a href="https://ubc-vision.github.io/nf-soft-mining/">Project Page</a>  </h3>
  <div align="center"></div>
</p>

## Overview
This repository contains the implementation and resources for our research paper ["Accelerating Neural Field Training via Soft Mining"](https://arxiv.org/abs/2312.00075) which was accepted at CVPR 2024. 

The paper presents a novel approach to accelerate the training of Neural Fields (NeRFs) by introducing a soft mining strategy. This strategy dynamically selects a subset of rays for each iteration, focusing on those that contribute most to the learning process based on the loss gradient. 

For a detailed understanding of the methodology, results, and more, please refer to the [full paper](https://ubc-vision.github.io/nf-soft-mining/).

<!-- <img src="./docs/static/videos/mic_good_video.gif" height=300> -->


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



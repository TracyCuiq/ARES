# ARES
Implamentation of [Cui, Q., Zhou, Z., Meng, R., Wang, S., Yan, H. and Wu, Q.M.J., ARES: On Adversarial Robustness Enhancement for Image Steganographic Cost Learning. IEEE Transactions on Multimedia, vol. 26, pp. 6542-6553, 2024.](https://ieeexplore.ieee.org/abstract/document/10398515/)

## Introduction
We propose a novel GAN-based steganographic approach, ARES, in which the Diversified Inverse-Adversarial Training (DIAT) strategy and the Steganalytic Feature Attention (SteFA) structure are designed to train a robust steganalytic discriminator. 
Specifically, the DIAT strategy provides the steganalytic discriminator with an expanded feature space by generating diversified adversarial stego-samples; the SteFA structure enables the steganalytic discriminator to capture more various steganalytic features by employing the channel-attention mechanism on higher-order statistics. Consequently, the steganalytic discriminator can build a more precise decision boundary to make it more robust, which facilitates learning a superior steganographic cost function. 

## Run 

- Train ARES
  ```sh
  python main.py 
  ```

## Citation

If you use ARES in your research or wish to refer to the results published here, please cite our paper with the following BibTeX entry.

```BibTeX
@article{cui2024ares,
  title={ARES: On Adversarial Robustness Enhancement for Image Steganographic Cost Learning},
  author={Cui, Qi and Zhou, Zhili and Meng, Ruohan and Wang, Shaowei and Yan, Hongyang and Wu, QM Jonathan},
  journal={IEEE Transactions on Multimedia},
  volume={26},
  pages={6542--6553},
  year={2024},
  publisher={IEEE}
}
```

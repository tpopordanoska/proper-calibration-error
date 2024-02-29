This is the code that accompanies the paper ["Consistent and Asymptotically Unbiased Estimation of Proper
Calibration Errors"](https://arxiv.org/pdf/2312.08589.pdf), published at AISTATS 2024.

The paper proposes a consistent and asymtotically unbiased estimator of *any* proper calibration error, which is a general class of calibration
errors derived from risk minimization via proper scores. 

## Installation

```
conda env create -f conda_env.yml
conda activate proper-ce
```


## To use it in your project

To evaluate proper calibration error using our estimator, you can copy the file `src/calibration_error/bregman_ce.py` in your repo.
The bandwidth of the kernel can either be manually set, or chosen by maximizing the leave-one-out likelihood with the method `get_bandwidth`.
For example, an estimate of $L_2$ and KL calibration error can be obtained with:

```
from calibration_error.bregman_ce import get_bregman_ce, l2_norm, negative_entropy

# Generate dummy probability scores and labels
f = torch.rand((50, 3))
f = f / torch.sum(f, dim=1).unsqueeze(-1)
y = torch.randint(0, 3, (50,))
bandwidth = 0.001

kde_l2 = get_bregman_ce(l2_norm, f, y, bandwidth)
kde_kl = get_bregman_ce(negative_entropy, f, y, bandwidth)
```

## To reproduce our results

### Downloading the pretrained models

Download the pretrained models from [Zenodo](https://zenodo.org/records/10724791), and place them in `trained_models`.

### Running our experiments

To reproduce Table 2 and Table 3, run the `calibration_methods.py` script.

The code for generating Figure 2 can be found in `synthetic_experiment.py`, whereas Figure 3 is generated in `notebooks/mnist_mlp_monotonicity.ipynb`.

The Figures and Tables in the Appendix are generated with the code in the `visualization` folder. 
Make sure to download the `results.zip` file from [Zenodo](https://zenodo.org/records/10724791) and place it in `trained_models`. 
The result files were generated using the scripts in `visualization/helper`.

## Reference
If you found this work or code useful, please cite:

```
@inproceedings{popordanoska2023consistent,
      title={Consistent and Asymptotically Unbiased Estimation of Proper Calibration Errors}, 
      author={Teodora Popordanoska and Sebastian G. Gruber and Aleksei Tiulpin and Florian Buettner and Matthew B. Blaschko},
      booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
      year={2024}
}
```

```
@inproceedings{popordanoska2022consistent,
  title={A Consistent and Differentiable $L_p$ Canonical Calibration Error Estimator},
  author={Popordanoska, Teodora and Sayer, Raphael and Blaschko, Matthew B.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

```
@inproceedings{gruber2022better,
   title={Better Uncertainty Calibration via Proper Scores for Classification and Beyond},
   author={Sebastian Gregor Gruber and Florian Buettner},
   booktitle={Advances in Neural Information Processing Systems},
   year={2022},
}
```


## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).

# Neural Eikonal Solver
[**Neural Eikonal Solver (NES)**](https://github.com/sgrubas/NES) is framework for solving factored eikonal equation using physics-informed neural network, for details see our paper: [early arXiv version](https://arxiv.org/abs/2205.07989) and [published final version](https://authors.elsevier.com/a/1g9st_W0q-3gK).

## Description
See quick introduction on [Google Colab](https://colab.research.google.com/github/sgrubas/NES/blob/main/notebooks/NES_Introduction.ipynb)

NES has two solvers:
1.   **One-Point NES (NES-OP)** is to solve conventional one-point eikonal ([NES-OP tutorial](https://github.com/sgrubas/NES/blob/main/notebooks/NES-OP_Tutorial.ipynb))
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=|\nabla \tau|=\displaystyle\frac{1}{v}">
</p>

3.   **Two-Point NES (NES-TP)** is to solve generalized two-point eikonal ([NES-TP tutorial](https://github.com/sgrubas/NES/blob/main/notebooks/NES-TP_Tutorial.ipynb))
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=|\nabla_r T|=\displaystyle\frac{1}{v_r}"> &nbsp;&nbsp;   
<img src="https://render.githubusercontent.com/render/math?math=|\nabla_s T|=\displaystyle\frac{1}{v_s}">
</p>

So far, NES outperforms all existing neural-network based solutions. Table shows average performance results on a smoothed part of Marmousi model (NES-OP vs. PINNeik and NES-TP vs. EikoNet). RMAE is relative mean-absolute error with respect to the reference solution (second-order factored Fast Marching Method). The tests were performed on GPU Tesla P100-PCIE.

|Solver   	|RMAE, %   	|Training time, sec   	|Network size   	|
|---	|---	|---	|---	|
|**NES-OP**   	|**0.2**  	|**240**   	|**7856**   	|
|PINNeik   	|12.4   	|330   	|4061   	|
|**NES-TP**   	|**0.4**   	|**300**   	|**51308**   	|
|EikoNet   	|5.4   	|9600  	|7913249   	|

For detailed comparisons see our colab notebooks [EikoNet](https://github.com/sgrubas/NES/blob/main/notebooks/EikoNet_NES-TP_Marmousi.ipynb) and [PINNeik](https://github.com/sgrubas/NES/blob/main/notebooks/PINNeik_NES-OP_Marmousi.ipynb).

## Installation
```python
pip install git+https://github.com/sgrubas/NES.git
```

# Quick example
```python
import NES

Vel = NES.velocity.MarmousiSmoothedPart()
Eik = NES.NES_TP(velocity=Vel)
Eik.build_model()
h = Eik.train(x_train=100000, epochs=1000, batch_size=25000)

grid = NES.utils.RegularGrid(Vel)
Xs = grid((5, 5)); Xr = grid((100, 100))
X = grid.sou_rec_pairs(Xs, Xr)
T = Eik.Traveltime(X)
```

# 2D examples of NES-OP
Isochrones of solutions. RMAE is shown above each figure. The NES solutions are *white dashed isochrones*, the reference solutions are *black isochrones*. 

<img src="https://github.com/sgrubas/NES/blob/main/NES/data/NES_OP_Sinus_0.06.png" alt="0.06%" width="400"/> <img src="https://github.com/sgrubas/NES/blob/main/NES/data/NES_OP_GaussianPlus_0.12.png" alt="0.12%" width="400"/>

<img src="https://github.com/sgrubas/NES/blob/main/NES/data/NES_OP_Flower_0.42.png" alt="0.42%" width="400"/> <img src="https://github.com/sgrubas/NES/blob/main/NES/data/NES_OP_Boxes_0.28.png" alt="0.28%" width="400"/>

<img src="https://github.com/sgrubas/NES/blob/main/NES/data/NES_OP_Layered_0.33.png" alt="0.33%" width="400"/> <img src="https://github.com/sgrubas/NES/blob/main/NES/data/NES_OP_LayeredBoxGauss_0.34.png" alt="0.34%" width="400"/>

# Citation
If you find NES useful for your research, please cite our paper:
```
@article{grubas2023NES,
title = {Neural Eikonal solver: Improving accuracy of physics-informed neural networks for solving eikonal equation in case of caustics},
journal = {Journal of Computational Physics},
volume = {474},
pages = {111789},
year = {2023},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2022.111789},
url = {https://www.sciencedirect.com/science/article/pii/S002199912200852X},
author = {Serafim Grubas and Anton Duchkov and Georgy Loginov},
keywords = {Physics-informed neural network, Eikonal equation, Seismic, Traveltimes, Caustics}
}
```

# Future plans
*  Anisotropic eikonal
*  Ray tracing
*  Wave amplitudes
*  Earthquake localization
*  Traveltime tomography

# Developers
Serafim Grubas (serafimgrubas@gmail.com) <br>
Nikolay Shilov <br>
Anton Duchkov <br>
Georgy Loginov

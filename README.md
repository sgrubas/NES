# Neural Eikonal Solver
**Neural Eikonal Solver (NES)** is framework for solving eikonal equation using neural networks. NES incorporates special features helping to solve the eikonal relatively fast, for details see our [paper](https://github.com/sgrubas/NES).

## Tutorials
See quick introduction on [Google Colab](https://colab.research.google.com/github/sgrubas/NES/blob/main/notebooks/NES_Introduction.ipynb)

NES has two versions:
1.   **One-Point NES (NES-OP)** is to solve conventional one-point eikonal ([NES-OP tutorial](https://github.com/sgrubas/NES/blob/main/notebooks/NES-OP_Tutorial.ipynb))
2.   **Two-Point NES (NES-TP)** is to solve generalized two-point eikonal ([NES-TP tutorial](https://github.com/sgrubas/NES/blob/main/notebooks/NES-TP_Tutorial.ipynb))

For comparison with existing neural-network solutions see [EikoNet](https://github.com/sgrubas/NES/blob/main/notebooks/EikoNet_NES-TP_Marmousi.ipynb) and [PINNeik](https://github.com/sgrubas/NES/blob/main/notebooks/PINNeik_NES-OP_Marmousi.ipynb)

## Installation
```python
!pip install git+https://github.com/sgrubas/NES.git
```

# Quick example
```python
import NES
import numpy as np

Vel = NES.misc.MarmousiSmoothedPart()
Eik = NES.NES_TP(velocity=Vel)
Eik.build_model()
h = Eik.train(x_train=200000, tolerance=7e-3, 
              epochs=10, verbose=0,
              batch_size=40000)

grid = NES.misc.RegularGrid(Vel)
Xs = grid((5, 5)); Xr = grid((200, 100))
X = grid.sou_rec_pairs(Xs, Xr)
T = Eik.Traveltime(X)
```

# Future plans
*  Anisotropic eikonal
*  Earthquake localization
*  Traveltime tomography

# Contributors
Serafim Grubas (serafimgrubas@gmail.com) <\br>
Nikolay Shilov <\br>
Anton Duchkov <\br>
Georgy Loginov

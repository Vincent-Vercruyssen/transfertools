# transfertools

`transfertools` is a small Python package containing recent **transfer learning algorithms**.
Transfer learning strives to transfer information from one dataset, the *source domain*, to a related dataset, the *target domain*. Several constraints and assumptions can be placed on the domains, inspiring different algorithms to do the information transfer.
The package contains four transfer learning algorithms.


## Installation

Install the package directly from PyPi with the following command:
```bash
pip install transfertools
```
OR install the package using the `setup.py` file:
```bash
python setup.py install
```
OR install it directly from GitHub itself:
```bash
pip install git+https://github.com/Vincent-Vercruyssen/transfertools.git@master
```


## Contents and usage

Transfer learning aims to transfer information from a *source domain* **Ds** to a related *target domain* **Dt**. A domain consists of a dataset with attributes **X** and labels *Y*. Thus, the source domain is **Ds** = {**Xs**, *Ys*} and the target domain is **Dt** = {**Xt**, *Yt*}.  The fundamental assumption is that the source and target domain live in the same feature space.
Different flavors of transfer learning methodologies exist.
*Unsupervised transfer* learning, for instance, disregards label information and only uses **Xs** and **Xt** to determine what information to transfer.
*Supervised transfer* learning uses the full domains **Ds** and **Dt** to do the transfer.
*Semi-supervised transfer* learning uses the full source domain **Ds** and the target attributes **Xt** to do the transfer.

The actual information that is transferred also differs between methods.
Domain adaptation techniques transform the source (and target) domains such that they match more closely (according to different criteria) and then combine all the data points to construct **Dcombo**.
Instance selection techniques select a subset of the source data that should be transferred to the target data to construct **Dcombo**.
After transfer, a classifier can be constructed using **Dcombo**.


### Instance selection techniques

The `transfertools` package contains two instance selection transfer techniques tailored to anomaly detection:

1. The **LocIT** (*localized instance transfer*) algorithm works in a completely unsupervised manner. It transfers the instances in **Ds** that have matching localized distributions in both domains [1]. This algorithm can also be used in other applications than anomaly detection.
2. The **CBIT** (*cluster-based instance transfer*) algorithm works in a semi-supervised manner. It transfer the instances in **Ds** that fall inside a cluster defined on the target data [2].

Given a source domain {**Xs**, *Ys*} and a target domain {**Xt**, *Yt*}, the algorithms are applied as follows:
```python
from transfertools.models import LocIT, CBIT

# train
transfor = LocIT()
transfor.fit(Xs, Xt)

# predict
Xs_trans = transfor.transfer(Xs)

# ... or immediately
Xs_trans = transfor.fit_transfer(Xs, Xt)
```

### Domain adaptation techniques

The `transfertools` package contains two instance domain adaptation techniques:

1. The **CORAL** (*correlation alignment*) algorithm is an unsupervised transfer learning technique that aligns the first and second order statistics of the source and target data [[3](https://arxiv.org/abs/1511.05547)].
2. The **TCA** (*transfer component analysis*) algorithm is an unsupervised transfer learning technique that projects the source and target data onto a lower-dimensional subspace [[4](https://ieeexplore.ieee.org/document/5640675)].

Given a source domain {**Xs**, *Ys*} and a target domain {**Xt**, *Yt*}, the algorithms are applied as follows:
```python
from transfertools.models import TCA, CORAL

# train
transfor = CORAL()
transfor.fit(Xs, Xt)

# predict
Xs_trans = transfor.transfer(Xs)

# ... or immediately
Xs_trans = transfor.fit_transfer(Xs, Xt)
```

## Package structure

The transfer learning algorithms are located in: `transfertools/models/`

For further examples of how to use the algorithms see the notebooks: `transfertools/notebooks/`


## Dependencies

The `transfertools` package requires the following python packages to be installed:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Scikit-learn](https://scikit-learn.org/stable/)


## Contact

Contact the author of the package: [vincent.vercruyssen@kuleuven.be](mailto:vincent.vercruyssen@kuleuven.be)


## References

[1] Vercruyssen, V., Meert, W., and J. Davis. (2020) *Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection.* In 34th AAAI Conference on Artificial Intelligence, New York. *To be published*

[2] Vercruyssen, V., Meert, W., and Davis, J. (2017) *Transfer learning for time series anomaly detection.* In CEUR Workshop Proceedings, vol. 1924, pp. 27-37.
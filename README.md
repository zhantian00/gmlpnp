C++ implementation of Generalized Maximum Likelihood Estimation for Perspective-n-Point Problem

To build our demo:

Install python 3.9 and

```bash
pip install -r requirements.txt
```

Build [g2o](https://github.com/RainerKuemmerle/g2o), [opengv](https://laurentkneip.github.io/opengv/page_installation.html#sec_installation_2) in the `/thirdparty`. Make sure to use the same version of g2o ([commit 95a177](https://github.com/RainerKuemmerle/g2o/tree/95a177f22cc1aba160a5659d83a9e3ad95ed5c7a)).

We use [pybind11](https://pybind11.readthedocs.io/en/stable/installing.html) to link the core algorithm with python scripts. Run

```bash
mkdir build
cd build
cmake ..
make
touch __init__.py
```

to get a `.so` in `/build` directory.

Then run the python scripts to evaluate the performance of our algorithm.

```bash
cd ..
python eval_synthetic.py
```
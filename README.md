# calypso

This repository contains accompanying code for "CalyPSO: An Enhanced Search Optimization based Framework to Model Delay Based PUFs" (to be) presented at CHES 2024.

## Setup

Clone the repository to the destination directory (say `CALYPSO_CLONE_DIR`) and setup packages through `pip install -r requirements.txt`.

## Running calypso

1. Navigate through `cd CALYPSO_CLONE_DIR/pypuf`

2. Execute `python3 calypso.py --target-degree 1 --challenge-num 50000 --cut-length 2 --proc 1 --population 500` to launch calypso as per relevant parameterization:
    - `--target-degree`: The value of `k` in a `k`-XOR PUF. Example: `--target-degree 1` mounts the attack on a APUF
    - `--challenge-num`: The number of challenge response pairs to use in training.
    - `--cut-length`: The number of delay stages to change for each generation in the evolutionary algorithm. For a `k`-XOR PUF, valid range is from `2` to `(64*k - 1)`
    - `--proc`: The number of processors to spread the workload on
    - `--population`: Initial size of the population

3. Additional optional flags:
    - `-landscape-evolution`: Whether to include landscape evolution or not. By default, it is disabled. Upon enabling this switch, in `10` generations, `25%` of the CRPs are evolved to smooth out the problem landscape over a period of time.
    - `-aeomebic-reproduction`: Whether to enable aeomebic reproduction or not. By default, it is disabled.

Sample command to enable both: `python3 calypso.py --target-degree 5 --challenge-num 50000 --cut-length 64 --proc 1 --population 500 -landscape-evolution -aeomebic-reproduction`

## Running calypso++

Similar to running calypso, except now, `calypso++.py` must be used. Consequently, `--target-degree k` represents an attack on `k` LP-PUF using a cross-architectural model of a `k`-XOR PUF.

## Jaccard distance

Both `calypso.py` and `calypso++.py` rely upon accuracy and bias metrics to construct their fitness functions. However, Jaccard distance could also be used as an alternative. We provide a sample in `calypso_jaccard.py`, which uses Jaccard distance to construct its fitness function. Rest all semantics of `calypso_jaccard.py` are similar to `calypso.py`

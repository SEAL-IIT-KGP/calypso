# calypso

This repository contains accompanying code for "CalyPSO: An Enhanced Search Optimization based Framework to Model Delay Based PUFs" (to be) presented at CHES 2024.

## Setup

Clone the repository to the destination directory (say `CALYPSO_CLONE_DIR`) and setup packages through `pip3 install -r requirements.txt`.

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

Sample command to enable both: `python3 calypso.py --target-degree 1 --challenge-num 50000 --cut-length 64 --proc 1 --population 500 -landscape-evolution -aeomebic-reproduction`

## Running calypso++

Similar to running calypso, except now, `calypso++.py` must be used. Consequently, `--target-degree k` represents an attack on `k` LP-PUF using a cross-architectural model of a `k`-XOR PUF. For example, to model `1` LP-PUF using a population of APUFs, execute `python3 calypso++.py --target-degree 1 --challenge-num 50000 --cut-length 2 --proc 1 --population 500 -landscape-evolution -aeomebic-reproduction`

## Jaccard distance

Both `calypso.py` and `calypso++.py` rely upon accuracy and bias metrics to construct their fitness functions. However, Jaccard distance could also be used as an alternative. We provide a sample in `calypso_jaccard.py`, which uses Jaccard distance to construct its fitness function. Rest all semantics of `calypso_jaccard.py` are similar to `calypso.py`

## Hardware data

We have also uploaded the hardware data we have collected and used to test CalyPSO/CalyPSO++ on real devices. Under `CALYPSO_CLONE_DIR/datasets`, there are two subsections: `CALYPSO_CLONE_DIR/datasets/in-house` (holding the collected challenge-response tuples for in-house generated hardware) and `CALYPSO_CLONE_DIR/datasets/online` (holding data from https://pypuf.readthedocs.io/en/latest/data/datasets.html)

The user can specify hardware data of a *new* PUF of their choice, with the restriction that both the challenge file and the response file *must* have only a single subdatabase. To elaborate, `calypso_hardware.py` shall load the challenge data as `challenge_data = np.load(args.challenge_file)` and then extract the *first* subdatabase as `challenge_data.files[0]` to load the actual challenges. The challenges can be present as a bitstring sampled from `{0, 1}^n` (where `n` is the
challenge length; `n=64` in our experiments). We internally convert such challenges to `{-1, 1}^n`.

We give some more examples here:

### BR-PUF 

We have also added a dedicated version of calypso (named `calypso_hardware.py`) to aid in testing out the hardware data. A sample run command: `python3 calypso_hardware.py --target-degree 4 --cut-length 64 --challenge-num 100000 --proc 1 --population 500 -aeomebic-reproduction --challenge-file $CALYPSO_CLONE_DIR/datasets/in-house/BRPUF/Challenge/chal_64_nChal_200000_bi.npz --response-file $CALYPSO_CLONE_DIR/datasets/in-house/BRPUF/GoldenResponses/respG_BRPUF_64_NChal_200000_5_meas_Br_10_all.npz`. This loads the challenge-response data related to BR-PUF and models it (cross-architecturally) using a `4`-XOR PUF. There are 3 main changes wrt. the simulation attacks mentioned before:
    - We have removed landscape evolution, since in some cases, we will not have the expansive dataset to consider only a subset of it for training.
    - `--challenge-file` flag takes the absolute path of the .npz file containing the challenge set
    - `--response-file` flag takes the absolute path of the .npz file containing the response set

### (11-11) iPUF

Some datasets are larger than what GitHub allows in its normal storage. To avoid using Git Large File Storage, we ship such data as zipped. One example is the challenge set of `(11-11) iPUF`. To run CalyPSO on it, perform the following steps:

1. `unzip $CALYPSO_CLONE_DIR/datasets/in-house/11_11_iPUF/chal_64_300000_bi.npz.zip` to unzip the challenge set (say to `$CALYPSO_CLONE_DIR/datasets/in-house/11_11_iPUF/`)
2. Run calypso as `python3 calypso_hardware.py --target-degree 11 --cut-length 64 --challenge-num 100000 --proc 1 --population 500 -aeomebic-reproduction --challenge-file $CALYPSO_CLONE_DIR/datasets/in-house/11_11_iPUF/chal_64_300000_bi.npz --response-file $CALYPSO_CLONE_DIR/datasets/in-house/11_11_iPUF/respG_11_11_iPUF_64_NChal_300000_5_means_Br_54.npz` to model the `(11-11) iPUF` cross-architecturally using a `11`-XOR PUF.

### 10XOR-APUF

1. `unzip $CALYPSO_CLONE_DIR/datasets/in-house/10XOR-APUF/chal_64_300000_bi.npz.zip` to `$CALYPSO_CLONE_DIR/datasets/in-house/10XOR-APUF`.
2. Run calypso as `python3 calypso_hardware.py --target-degree 10 --cut-length 64 --challenge-num 100000 --proc 1 --population 500
 -aeomebic-reproduction --challenge-file $CALYPSO_CLONE_DIR/datasets/in-house/10XOR-APUF/chal_64_300000_bi.npz --response-file $CALYPSO_CLONE_DIR/datasets/in-house/10XOR-APUF/respG_10_XORPUF_64_NChal_300000_5_meas_Br_98.npz` to model the `10XOR-APUF` cross-architecturally using a `10`-XOR PUF.


![calypso](https://github.com/SEAL-IIT-KGP/calypso/blob/main/calypso.jpg)

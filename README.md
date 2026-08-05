# MD-WGAN: Coordinated Distributed Training of Multi-Discriminator GANs under Mobility

Official implementation and simulation environment for:

> F. Jafari Kaleibar, A. M. Zaki, and M. Litoiu, "Coordinated Distributed
> Training of Multi-Discriminator GANs under Mobility: A Reinforcement-
> Learning and Game-Theoretic Framework," *IEEE Transactions on Parallel
> and Distributed Systems* (under review), 2026.

This repository contains two components:

1. **Vehicular network simulation** (ns-3 + SUMO) — models the IoV
   communication environment, mobility, and V2I links.
2. **MD-WGAN training pipeline** <!-- TODO: confirm folder name/path --> —
   the reinforcement-learning and game-theoretic coordination framework,
   the multi-discriminator WGAN, and the baselines evaluated in the paper.

---

## Repository Structure

```
MD-WGAN/
├── ns-3.43/                       # ns-3 network simulator core
│   ├── scratch/
│   │   └── dsrc-simulations4.cc   # main DSRC/V2I simulation entry point
│   └── GANs/
│       └── WGAN/                  # Python training code (RL + WGAN + baselines)
│           ├── requirements.txt
│           └── ...
├── netanim-3.109/                 # NetAnim visualization tool
├── bake/                          # ns-3 build tool
├── build.py                       # builds ns-3 + NetAnim
├── constants.py
├── util.py
└── README.md
```

---

## 1. Setting Up and Running the ns-3 Simulation

This repo follows the standard `ns-3-allinone` layout. To build ns-3 and
NetAnim:

```bash
./build.py --enable-examples --enable-tests
```

This builds `ns-3.43` and `netanim-3.109`. Build time is typically
15–30 minutes depending on your machine.

### Running the DSRC/V2I simulation

The main simulation entry point is:

```
ns-3.43/scratch/dsrc-simulations4.cc
```

From the `ns-3.43` directory, run it with:

```bash
cd ns-3.43
./ns3 run scratch/dsrc-simulations4
```

Or, to pass simulation parameters (adjust flags to match what the script
accepts):

```bash
./ns3 run "scratch/dsrc-simulations4 --numVehicles=40 --simTime=150 --numRSU=8"
```

Simulation output (trace files, mobility logs, resource/latency metrics)
is written to `<TODO: output directory>`, which is consumed by the Python
training pipeline described below.

### Visualizing with NetAnim

```bash
cd netanim-3.109
./NetAnim
```

Open the `.xml` animation trace produced by the ns-3 run.

---

## 2. Setting Up the Python Training Pipeline (MD-WGAN)

The RL controllers, game-theoretic coordination, multi-discriminator
WGAN, and baseline methods (HVS-GAN, Early Update MD-GAN, Random
Parameter Selection) are implemented in Python under `ns-3.43/GANs/WGAN`.

### Requirements

```bash
cd ns-3.43/GANs/WGAN
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

See [`requirements.txt`](./ns-3.43/GANs/WGAN/requirements.txt) for pinned
versions.

### Running training

<!-- TODO: fill in the actual entry-point script and CLI flags, e.g.: -->
```bash
python train.py \
  --data <path-to-ngsim-processed-data> \
  --config configs/full.yaml \
  --seed 0
```

### Reproducing the baselines

<!-- TODO: fill in per-baseline commands, e.g.: -->
```bash
python train.py --config configs/random_selection.yaml
python train.py --config configs/hvs_gan.yaml
python train.py --config configs/early_update_mdgan.yaml
```

### Reproducing the ablation study

<!-- TODO: fill in once ablation configs exist -->
```bash
python train.py --config configs/rl_only.yaml
python train.py --config configs/game_only.yaml
```

---

## Dataset

Experiments use an enhanced version of the [NGSIM Next-Generation
Simulation dataset](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/),
augmented with synthetic V2X communication data. See the paper (Section
V-B) for preprocessing details.

<!-- TODO: add data download/preprocessing instructions if the raw data
     isn't included in the repo -->

---

## Citation

If you use this code, please cite:

```bibtex
@article{jafarikaleibar2026coordinated,
  title   = {Coordinated Distributed Training of Multi-Discriminator GANs
             under Mobility: A Reinforcement-Learning and Game-Theoretic
             Framework},
  author  = {Jafari Kaleibar, Farhoud and Zaki, Amr M. and Litoiu, Marin},
  journal = {IEEE Transactions on Parallel and Distributed Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

<!-- TODO: add license (e.g. MIT) -->

## Contact

Farhoud Jafari Kaleibar — farhoud [at] yorku.ca
Department of Electrical and Computer Engineering, York University

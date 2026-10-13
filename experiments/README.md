# Experiments

This folder contains the simulation code accompanying *Proto-HAL: Semantic Saturation and Task Drift in AI Agents*.

The simulations explore how task-relevant learning may become less attractive as a domain becomes familiar, even when the weight assigned to information gain remains constant.

They are **synthetic models**, not tests of deployed language models.

## What is included?

| Option | Experiment |
| --- | --- |
| `pilot` | Initial demonstration |
| `study1` | Semantic saturation, task drift, empowerment, and task-loss envelope |
| `study2a` | Learned representations and critical task cost |
| `study2b` | Matched comparison including frozen-control conditions |

## Run the simulations

You need Python 3 and the packages listed in `requirements.txt`.

Install them:

```bash
python -m pip install -r requirements.txt
```

Show the available studies:

```bash
python proto_hal_simulations.py --list
```

Run the pilot first:

```bash
python proto_hal_simulations.py --study pilot
```

The results are written to a `proto_hal_results` folder by default. To choose a different destination, add `--output-dir YOUR_FOLDER`.

The other studies can take substantially longer. Run each separately by replacing `pilot` with `study1`, `study2a`, or `study2b`.

## Status and limitations

The code is retained from the original supplied simulation file; the study logic has not been rewritten. The pilot has been smoke-tested. Full reproduction of the longer studies, numerical result matching, and cross-platform compatibility have **not** yet been verified.

The script packages several earlier study scripts into one command-line runner. Some study sections contain successive analysis variants. Results should therefore be checked against the corresponding publication before making reproducibility claims.

**Publication:** [Proto-HAL on Zenodo](https://zenodo.org/records/22975822).

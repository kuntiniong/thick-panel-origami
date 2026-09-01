# Computational Design & Fabrication Pipeline for Thick-Panel Origami

An end-to-end design & fabrication pipeline for converting zero-thickness origami structures into thick panels, powered by a simulator optimization loop using evolutionary algorithm (CMA-ES) to search for the optimal thickness in rigid-foldable (i.e. energy-minimizing deployable) designs.

> *Please refer to https://github.com/kuntiniong/thick-panel-origami/tree/develop for the lastest updates.*

## How to use
1. Create a venv with Anaconda

```bash
conda env create -f environment.yml
```

2. Run the code according to your tasks in this table:

| Task | Path | Results Path |
|-|-|-|
| Simulation | `/phys_sim_pd14.py` | \ |
| Optimization | `/optimization/run.py` | `/physResult/` |
| Removing intersecting geometry | `/panel_trimming/run_panel_trimming.py` | `/panel_trimming/trimmedData/*trimmed.json` |
| Cleaning trimmed .json | `/panel_trimming/clean/clean.py` | `/panel_trimming/trimmedData/*-cleaned.json` |
| .json visualization | `/panel_trimming/visualize/visualize.py` or `/panel_trimming/visualize/visualize_3d.py` | `/panel_trimming/visualize/output/` |
| .json to .stl conversion | `/panel_trimming/json_to_stl/run.py` | `/panel_trimming/trimmedData/stl-*/`  |

> *Note i: All the configuration params in this project follow this structure:*
> ```bash
> /config.example.yml 
> # configs template
> /config.yml 
> # this file is hidden by .gitignore when you first clone. it's a good pratice to create on your own by copying the template file and remove ".example", though the code uses .example.yml as fallback if this file doesn't exist
> ```

> *Note ii: Most results files like `/physResult/ & */*/output/*` are hidden by .gitignore for more efficient version control. You could adjust it to your own preference.*

## Folder structure
```bash
/legacy-visualization/ 
# legacy methods & visualizations to improve optimization performance (e.g. symmetry grouping, bias matrix initialization, etc.)

/optimization/
  /algorithms/
  # 5 algs: bayesian, cma-es-elitist, cma-es-margin, cma-es, differential evolution
  /framework.py
  # the main component orchestrating the alg & the simulation loop
  /manual.py
  # input heights manually and get the reward
  /run.py
  # entry point

/panel_trimming/
# main logic for eliminating intersecting geometry & fabrication
  /clean/
  # clean up messy nodes after running /run_panel_trimming.py & export files as json with suffix "-cleaned"
  /json_to_stl/
  # convert json to printable stl files
  /visualize/
  # 2d + 3d visualization for trimmed or cleaned json files (NOT stl) 
  /run_panel_trimming.py
  # entry point for trimming & export files as json with suffix "-trimmed" 

/cdf_thick_panel.py
# legacy optimization loop

/collision_util.py
/collision.py
# collision detection

/ori_sim_sys.py
/phys_sim_pd14.py
/plot_fitness.py
/spatialhash.py
/utils.py
# simulator
```

# DDFenics
A (model-free) Data-driven implementation based on fenics (https://github.com/felipefr/ddfenics).

## Getting started
### Tutorials DDFenics 
Aim: Solve a simple 2D bar problem using standard Fenics and DDFenics.
- 1) 2D bar (linear elastic) in FEniCs: tutorial/linear/main_bar.ipynb
- 2) 2D bar (linear elastic) in DDFenics (Hands-on): tutorial/linear/main_bar_dd_to_fill.ipynb
  - Complete the "missing lines" (commented in the notebook) 
  - Plot the convergence (with data) curves
  - Run the sanity-check (last block of notebook) and redo DDCM
  - Change C = some isotropic elastic tensor (hookean) for changed (E', nu') ?
- 3) 2D bar (nonlinear elastic) in FEniCs: tutorial/nonlinear/main_bar_nonlinear.ipynb
- 4) 2D bar (nonlinear elastic) in DDFenics: tutorial/nonlinear/main_bar_nonlinear_dd.ipynb

## Installation 
Please run the steps below (subsection Steps) that installs the requirements (listed in subsection Requirements).

### Steps:
- Download the install.sh script
- Change the initial paths accordingly to your system/preferences
- run: sh install.sh 1
- activate the conda environment: conda activate <ddfenics_environment>
- run: sh install.sh 2
- launch jupyter: jupyter-lab in the desired parent folder

Obs: You can run step by step the bash script in order to have full control of eventual errors in the installation.

Obs: Note that the script automatically add into the PYTHONPATH the root directory in which you cloned DDFenics. This is done by adding a .pth file (any name) with a list of directories in ~/miniconda3/envs/ddfenics/lib/python3.8/site-packages. You can also add the directories ''by hand'' in spyder (Tools > PYTHONPATH) or sys.path.insert(..., '...folder...') in your source files.

Obs: Command to convert python notebooks to python files (if you prefer not use jupyter-lab): jupyter nbconvert --to script file.ipynb 

### Requirements
DDFenics relies on the following libraries/packages (some others are installed automatically altogether with the principal ones):
- library  /        version
- python   /        3.8 (conda-forge) 
- fenics   /        2019.1.0   (conda-forge)
- scikit-learn /  latest or no restriction (conda-forge)
- matplotlib	/  latest or no restriction (conda-forge)

Optionally for mesh generation and postprocessing (with Paraview):
- library   /    version
- h5py      /    2.10.0 (conda-forge)
- meshio    /    3.3.1  (pypi)
- pygmsh    /    6.0.2  (pypi)
- gmsh      /    4.6.0   (pypi)

Optionally for an interactive run of the tutorial:
- library  /  version
- jupyterlab / latest or no restriction (conda-forge)
- ipykernel	 /  latest or no restriction (conda-forge)

Obs: the default repository is conda-forge, otherwise pypi from pip. Recommended versions should be understood only as guideline and sometimes the very same version is not indeed mandatory.

Obs: We included in the "external" folder a lite version of fetricks (https://github.com/felipefr/fetricks), that implements some auxiliary routines for computational mechanics using fenics. However, you can decide to use your own functions for this job. 

Obs: We recommend the use of miniconda (https://docs.conda.io/en/latest/miniconda.html) or your preferred Anaconda-like variants.

Obs: For Windows users, unfortunately Fenics is not available in the Anaconda repositories for Windows. As alternative, we recommend: i) to use the the Linux (Ubuntu) subsystem (https://learn.microsoft.com/en-us/windows/wsl/install) and use the instructions as below; ii) set some virtual machine (e.g. Virtual Box) or iii) use the Docker version of Fenics (not tested!) (https://fenicsproject.org/download/archive/).

## Documentation
1. Map between Galerkin-like variational approximation and FEniCs objects.
![FenicsContinuum](./docs/FenicsContinuum.png)

2. Map between FEniCs and the corresponding objects in DDFenics.
![FenicsDDFenics](./docs/FenicsDDFenics.png)

3. Map between (Model-free) Data-driven formulation and the corresponding objects in DDFenics.
![DDFenics](./docs/DDFenics.png)

### Basic Usage (a little deprecated) 

The usage mimetises the basic framework of fenics by defining Data-driven equivalents of the LinearVariationalProblem
and LinearVariationalSolver objects (see https://fenicsproject.org/pub/tutorial/html/._ftut1018.html), respectively DDProblem and DDSolver.
Additionally the DDProblem object depends on a Data-driven material, which is defined by an instance of a DDMaterial. The output of the DD solver also contains
the mechanical and neighrest projections (in the material database) states, which are instances of DDFunction (just a derived class of the dolfin Function to facilitate 
some needed domain-specific operations )

#### Fenics

0. Definition of standard constitutive equations. 
1. Definition of mesh, FE spaces, boundary conditions, variational forms, etc. 
2. Variational problem definition: problem = LinearVariationalProblem(a, b, uh, bcs)
3. Solver definition: solver = LinearVariationalSolver(problem, solver_args) 
4. Solve the problem: solver.solve()


#### DDFenics

0. Definition of Data-driven constitutive equations : loading of material datasets and definition of an approximative metric ==> ddmat = DDMaterial(DB, Metric) 
1. Definition of mesh, FE spaces, boundary conditions, standard constitutive equations, variational forms, etc. (idem)
2. Definition of Gauss-Point spaces where the material states live : Sh0 = df.VectorFunctionSpace(Uh.mesh(), 'DG', degree = 0 , dim = 3). 
Stresses and strains are instances of DDFunction(Sh0). 
3. DD Variational problem definition: problem = DDProblem(a, b, uh, bcs, ddmat, ddmat, state_mech, state_db) (almost idem)
4. Solver definition: solver = DDProblem(problem, solver_args) (idem) 
5. Solve the problem: solver.solve() (idem)

## Citing
Please cite this repository if this library has been useful for you.
[![DOI](https://zenodo.org/badge/545056382.svg)](https://zenodo.org/badge/latestdoi/545056382)

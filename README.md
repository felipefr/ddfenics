
# Tutorial DDFenics at CISM's minicourse Data-driven Mechanics tutorial at (2022) 

## DDFenics
A (model-free) Data-driven implementation based on fenics.

## Installation 

Among other libraries, DDFenics relies on the following ones (some of them are installed automatically in the installation of previous ones):

- library          version  
- python           3.8 (recommended) 
- fenics           2019.1.0   (conda-forge)
- scikit-optimize  0.8.1  (conda-forge)
- meshio           3.3.1  (pypi)
- pygmsh           6.0.2  (pypi)
- gmsh             4.6.0   (pypi)
- pytest           6.2.5.  (pypi)

Obs: the default repository is conda-forge, otherwise pypi from pip. Recommended versions should be understood only as guideline and sometimes the very same version is not indeed mandatory. 

We recommend the use of miniconda (https://docs.conda.io/en/latest/miniconda.html)

- To create a fresh environment with fenics (note that the default conda-forge repository version of fenics has became recently. The use of conda-forge/label/cf202003 fix the issue, although this version is not compatible with python>=3.9):
```
conda create -n ddfenics_tutorial -c conda-forge/label/cf202003 fenics python=3.8.13
```

- To activate the environment:
```
conda activate ddfenics_tutorial
```

- To install additional packages:
```
conda install -c conda-forge <name_of_the_package>=<version>
```
or use pip
```
pip install <name_of_the_package>==<version>
```

or at once

```
conda install -c conda-forge scikit-optimize
pip install pytest gmsh==4.6.0 meshio==3.3.1 pygmsh==6.0.2
```

- Make sure your PYTHONPATH variable contains the root directory in which you cloned DDFenics. By default, the anaconda installation does not take into consideration the OS path. You can add a .pth (any name) file listing the directories into ~/miniconda/envs/ddfenics_tutorial/lib/python3.8/site-packages. You can also add the directories you want  into spyder (Tools > PYTHONPATH), if you are using it.  

## Testing  (not implemented yet)
```
cd tests
pytest test_*    
or pytest --capture=no test_file.py::test_specific_test  (for detailed and specific test)      
```

## Usage

The usage mimetises the basic framework of fenics by defining Data-driven equivalents of the LinearVariationalProblem
and LinearVariationalProblem objects (see https://fenicsproject.org/pub/tutorial/html/._ftut1018.html), respectively DDProblem and DDSolver.
Additionally the DDProblem object depends on a Data-driven material, which is defined by an instance of a DDMaterial. The output of the DD solver also contains
the mechanical and neighrest projections (in the material database) states, which are instances of DDFunction (just a derived class of the dolfin Function to facilitate 
some needed domain-specific operations )

### Fenics

0. Definition of standard constitutive equations. 
1. Definition of mesh, FE spaces, boundary conditions, variational forms, etc. 
2. Variational problem definition: problem = LinearVariationalProblem(a, b, uh, bcs)
3. Solver definition: solver = LinearVariationalProblem(problem, solver_args) 
4. Solve the problem: solver.solve()


### DDFenics

0. Definition of Data-driven constitutive equations : loading of material datasets and definition of an approximative metric ==> ddmat = DDMaterial(DB, Metric) 
1. Definition of mesh, FE spaces, boundary conditions, standard constitutive equations, variational forms, etc. (idem)
2. Definition of Gauss-Point spaces where the material states live : Sh0 = df.VectorFunctionSpace(Uh.mesh(), 'DG', degree = 0 , dim = 3). 
Stresses and strains are instances of DDFunction(Sh0). 
3. DD Variational problem definition: problem = DDProblem(a, b, uh, bcs, ddmat, ddmat, state_mech, state_db) (almost idem)
4. Solver definition: solver = DDProblem(problem, solver_args) (idem) 
5. Solve the problem: solver.solve() (idem)

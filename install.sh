CONDA_PATH=/home/ffiguere/miniconda3
TUTORIAL_PATH=/home/ffiguere/ECN/handsOnCISM
DDFENICS_ENV=ddfenics
DDFENICS_GIT=https://github.com/felipefr/ddfenics_tutorial.git

# run
# 1) sh install.sh 1
# 2) conda activate $DDFENICS_ENV
# 3) sh install.sh 2
# 4) Run tutorial with jupyter-lab (optionally --notebook-dir=/home/...)

printf "options:\n 1 - conda environment creation and git clone\n 2 - installing of additional packages\n chosen option: $1\n"

if [ $1 -eq 1 ]; then
	conda create -n $DDFENICS_ENV -c conda-forge fenics python=3.8 h5py=2.10.0 jupyterlab ipykernel scikit-learn matplotlib
	git clone $DDFENICS_GIT $TUTORIAL_PATH/$DDFENICS_ENV
	printf "$TUTORIAL_PATH/\n$TUTORIAL_PATH/$DDFENICS_ENV/external/" > extra_python_folders.pth
	cp extra_python_folders.pth $CONDA_PATH/envs/$DDFENICS_ENV/lib/python3.8/site-packages/extra_python_folders.pth
fi 

if [ $1 -eq 2 ]; then
	python -m ipykernel install --user --name=$DDFENICS_ENV
	pip install gmsh meshio==3.3.1 pygmsh==6.0.2
fi

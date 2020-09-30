#!/bin/bash
clear

eval "$(conda shell.bash hook)"
conda activate Py37
conda env list

python encoders/BackwardDifferenceEncoder.py
python encoders/BaseNEncoder.py
python encoders/BinaryEncoder.py
python encoders/CatBoostEncoder.py
python encoders/CountEncoder.py
python encoders/DropEncoder.py
python encoders/GLMMEncoder.py
python encoders/HelmertEncoder.py
python encoders/JamesSteinEncoder.py
python encoders/LeaveOneOutEncoder.py
python encoders/MEstimateEncoder.py
python encoders/OneHotEncoder.py
python encoders/OrdinalEncoder.py
python encoders/PolynomialEncoder.py
python encoders/SumEncoder.py
python encoders/TargetEncoder.py
python encoders/HashingEncoder.py
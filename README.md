# traj_calc
Planetary entry and orbital propagation simulation using Python.

Citation: [![DOI](https://zenodo.org/badge/21491/Nate28/traj_calc.svg)](https://zenodo.org/badge/latestdoi/21491/Nate28/traj_calc)

Requires: NumPy, SciPy, Cantera, matplotlib, aerocalc (for US76 atmosphere model only)

Trajectories are simulated using the Fortran ODE solvers packaged with Scipy, with array handling courtesy of NumPy.  Cantera is used to calculate gas properties, and aerocalc is used for the US76 atmospheric model.  

Capabilities include:
+ 3DoF lifting trajectory model (rotating, non-inertial reference frame)
	+ Lift/drag forces considered (set lift coefficient to 0 for ballistic simulation)
	+ Spherical, non-rotating planet assumption
+ 3DoF orbit propagation model (inertial reference frame w. rotating force transformations)
	+ Lift/drag/lateral forces considered
	+ Spherical, non-rotating planet assumption
+ Choice of atmospheric models 
	+ US76 
	+ Jacchia77
	+ NRLMSISE_00
+ Function to allow automatic recalculation of aerodynamic coefficients during simulation (aerodynamic database interface)
+ Stagnation heat flux correlation library (still experimental, with known errors)
	
Future additions planned:
+ 6 DoF dynamics model
+ Spherical harmonics for gravity modelling
+ Ablation model for heat shields
+ Aerodynamic coefficient calculator and database generator

NRLMSISE_00 Python model by Deep Horizons
https://github.com/DeepHorizons/Python-NRLMSISE-00

J77 Python model by Deep Horizons
https://github.com/DeepHorizons/Python-Jacchia77

Main author: Nathan Donaldson, Department of Aeronautics and Astronautics, University of Southampton

Contributions: Hilbert van Pelt, Australian Defence Force Academy

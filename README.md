# MagPI 


[Documentation](https://schaffer9.github.io/MagPI/)

MagPI was developed in the course of my PhD project at _WPI_ and the _Research Platform MMM_ at University of Vienna within the research group of Dr. Lukas Exl. It is a collection of useful algorithms and methods for the training of Physics Informed Neural Networks (PINNs). This is considered to be a starting point to a larger modular PINN framework for large scale full 3d micromagnetic simulations.

Here is a short summary of the features which are offered by MagPI at this point:
- Preliminary implementation of a CSG modelling framework using R-Functions
- A Trust Region optimizer with CGSteihaug solver
- Differential operators using forward mode AD
- A basic differentiable integration module
- Efficient Hessian vector products
- An differentiable ode solver
- Implementation of Quaternion rotation which can be used for CSG
- Some simple domains and useful transformations
- Simple importance sampling algorithm


<center><img src="examples/pacman.png" alt="pacman" width="250"/></center>

## Acknowledgements
Financial support by the Austrian Science Fund (FWF) via project P-31140 ”Reduced Order Approaches for Micromagnetics (ROAM)” and project P-35413 ”Design of Nanocomposite Magnets by Machine Learning (DeNaMML)” is gratefully acknowledged.
Simulation of the Blood–Brain Barrier with PINN
📖 Introduction

The blood–brain barrier (BBB) is a highly selective interface between the circulatory system and the brain. It protects neurons by regulating the entry of molecules while making drug delivery to the brain a major challenge.

This project builds a computational simulation of the BBB using reaction–diffusion equations and Physics-Informed Neural Networks (PINNs). The goal is to visualize and analyze how drugs move from the bloodstream into the brain, considering both diffusion and active transport mechanisms.

🧬 Biological Background

The BBB is composed of several structural layers:

Endothelial cells – lining of brain capillaries, joined by tight junctions.

Basement membrane – structural support around endothelial cells.

Astrocytic end-feet – projections of glial cells surrounding capillaries.

Neurons – brain cells that ultimately interact with delivered drugs.

Drug transport across the BBB involves:

Passive diffusion.

Carrier-mediated transport.

Efflux pumps removing unwanted molecules.

Enzymatic metabolism at the barrier.

⚙️ Methods

Numerical Simulation

3D diffusion of drug molecules through BBB layers.

Adjustable permeability and metabolism rates.

PINN Approximation

Neural networks learn the solution to the PDEs.

Model parameters can be inferred from synthetic data.

GPU acceleration ensures efficient computation.

Visualization

2D concentration maps across barrier layers.

3D isosurface rendering of drug penetration.

Interactive exploration of drug distribution.

📊 Results

Visuals show how concentration drops from blood → brain.

Diffusion-only transport is slow, whereas active transport accelerates entry.

PINNs can infer hidden parameters like permeability and transport efficiency.

🖼️ Schematic


Blood vessel → Endothelium → Basement membrane → Astrocyte → Neuron

🚀 Applications

Preclinical drug delivery modeling.

Testing neuropharmacology hypotheses.

Educational demonstration of the BBB.

Foundation for multi-scale brain models.

📌 Future Work

Add heterogeneous enzyme activity in basement membrane.

Simulate multiple drug molecules simultaneously.

Link BBB simulation to neural activity models.

Compare with experimental datasets.

✨ Conclusion

This project combines biological insight with modern AI methods to model the BBB. By uniting diffusion physics with PINNs, it provides a tool that is both educational and useful for research directions in drug discovery and neuroscience.

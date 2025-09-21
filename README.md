# Blood-Brain-Barrier
This project builds a computational model of the BBB using Physics-Informed Neural Networks (PINNs) and GPU acceleration. By combining biological principles with physics-based simulations, it aims to visualize and understand how different drugs cross (or fail to cross) the BBB and interact with brain tissue.


Simulation of the Blood–Brain Barrier (BBB) with Physics-Informed Neural Networks (PINNs)
🌌 Overview

This project models how drugs move from the bloodstream into the brain using Physics-Informed Neural Networks (PINNs), accelerated with CUDA.

The aim is to understand one of the most challenging problems in neuroscience and medicine: how substances cross the blood–brain barrier (BBB). By combining biology, physics, and deep learning, this work simulates the interactions between drugs and the brain’s natural defense systems.

🧠 Biological Foundations
The Blood–Brain Barrier (BBB)

Built from endothelial cells tightly joined together.

Acts as the brain’s primary filter, blocking harmful compounds.

Uses efflux pumps (like P-glycoprotein) to actively remove many drugs.

Basement Membrane (BM)

A thin extracellular layer between vessels and brain tissue.

Made of proteins like collagen IV and laminins.

Adds a second filtering step for molecules trying to reach neurons.

Astrocytic End-Feet

Astrocytes extend “end-feet” that wrap around blood vessels.

Regulate ion balance, neurotransmitters, and drug transport.

Serve as a metabolic and signaling bridge between blood and neurons.

Neuron Uptake

Lipid-soluble drugs may diffuse directly.

Some molecules (like L-DOPA) require transporters to cross.

Final action happens at the neuron, where drugs can have therapeutic or harmful effects.

🔬 Analogy: The Drug’s Journey

Crossing into the brain is like passing through multiple layers of security:

Airport security → Endothelial cells (BBB).

Customs checkpoint → Basement membrane.

Local police check → Astrocytic regulation.

Final recipient → Neurons.

Only drugs with the right properties (size, solubility, charge) can successfully make it through.

⚙️ Computational Model
Features

PINN-based solver for diffusion and transport.

CUDA acceleration to fully use modern GPUs (tested on RTX 5080).

Multi-layer modeling of:

Endothelial barrier.

Basement membrane.

Astrocytic interactions.

Neuronal uptake.

Why PINNs?

They enforce physical laws like diffusion equations.

They work with limited experimental data.

They provide biologically realistic simulations at scale.

📊 Visualizations

2D cross-sections → clearer view of gradients.

3D volumetric plots → capture the full structure of diffusion.

These show how drug concentration changes as it moves from the blood vessel, through the barrier, and toward neurons.

🚀 Roadmap

 Baseline PINN diffusion solver.

 GPU acceleration.

 3D visualizations.

 Add basement membrane filtering.

 Add astrocytic regulation.

 Simulate multiple drugs with different properties.

 Compare results with real-world drug cases (e.g., L-DOPA vs dopamine).

 Write up for publication.

📚 Philosophy

“Algebra is like sheet music. It’s not whether you can read music, it’s whether you can hear it.” — Oppenheimer

This project is about hearing the music of biology through mathematics. It brings together theory, computation, and neuroscience to simulate one of the brain’s most fundamental protective systems.

🤝 Contribution

Contributions are welcome from both:

Computational scientists → improving solvers, scaling on GPUs.

Biologists & neuroscientists → refining parameters, providing validation data.


I. Quick roadmap — what we’ll cover

BBB anatomy and functional units.

Transport routes (passive, paracellular, transcellular, carrier, receptor, transcytosis, efflux).

Drug properties that matter and why.

Governing equations and how each biological process becomes a mathematical term.

How those terms appear in your finite-difference solver and in the PINN.

Typical parameter values / orders of magnitude (so your code’s placeholders make sense).

Limitations, validation data needed, next biological refinements.

1. Anatomy and the functional unit

The BBB is not a single membrane. At the microscale, the minimal functional unit is the capillary endothelial cell layer plus supporting structures:

Endothelial cells with very close tight junctions (claudin, occludin, ZO proteins) forming a continuous barrier that prevents paracellular leak.

Basement membrane (extracellular matrix) under the endothelium.

Pericytes embedded in the basement membrane that regulate blood flow and barrier integrity.

Astrocyte end-feet that ensheath the vessel and release factors regulating BBB phenotype.

Physically: capillary lumen radius ~1–5 µm, endothelial thickness ~0.2–1 µm, spacing between capillaries ~20–40 µm. The barrier thickness (endothelium) is very small relative to tissue, which motivates modeling it as a thin interface (surface) in many PK models.

Functional consequence: paracellular pathways are almost sealed; transcellular (through cells) and specialized transport mechanisms dominate.

2. Transport routes — biology → short descriptions

Passive transcellular diffusion

Small, lipophilic molecules dissolve into and cross the endothelial cell membrane by partitioning into lipid bilayer and diffusing across. Speed depends on lipophilicity and membrane diffusivity.

Paracellular transport

Movement between cells through tight junctions. At the BBB, tight junctions are tight — paracellular permeability is extremely low for most solutes. Some small hydrophilic solutes or pathological states (inflammation) can increase paracellular leak.

Carrier-mediated transport

Solute carriers (SLC family) such as GLUT1 (glucose), LAT1 (large neutral amino acids) actively transport specific substrates across the endothelium via facilitated diffusion or active transport.

Receptor-mediated/transcytosis

Large molecules (e.g., insulin, transferrin) bind receptors and are internalized and shuttled across via vesicles (endocytosis → exocytosis). Slow but can transport large ligands.

Adsorptive transcytosis

Non-specific electrostatic interactions can cause uptake of cationic proteins/peptides.

Active efflux transporters

ABC transporters (P-glycoprotein / ABCB1, BCRP / ABCG2, MRPs) actively pump many xenobiotics and drugs back into blood, often ATP-dependent. Key for opioids and many CNS drugs — they reduce net brain uptake.

Metabolic enzymes

Endothelial cells express Phase I/II enzymes that can metabolize drugs during transit, reducing net flux.

Clearance pathways (brain side)

Interstitial fluid flow, perivascular drainage, and glymphatic clearance remove drug from brain extracellular space.

3. Drug properties that control transport

Molecular weight (MW): larger molecules penetrate more slowly; >400–500 Da reduces passive permeation substantially.

Lipophilicity (logP or logD): higher lipophilicity increases membrane partitioning and transcellular diffusion, but extreme lipophilicity can increase tissue binding and lower free fraction.

Polar surface area (PSA): high PSA decreases diffusion through lipid.

Hydrogen bond donors/acceptors: more H-bonds generally reduce permeation.

Charge at physiological pH: neutral molecules cross membranes easier than charged ones.

Plasma protein binding (fu): only the unbound fraction in plasma crosses the BBB; total concentration is not what drives flux. So free fraction (fu) is critical.

Transporter substrate status: whether a drug is a P-gp/BCRP substrate. This can dominate brain exposure independent of passive permeability.

Practical takeaway: to predict brain exposure you need (a) free plasma concentration, (b) intrinsic permeability (P or PS), (c) transporter kinetics (Vmax, Km), (d) clearance from brain.

4. Governing equations — mapping biology to math
A. Bulk transport PDE (diffusion + advection + reactions)

For concentration 
𝐶
(
𝑥
,
𝑡
)
C(x,t) in tissue/plasma space, the advection–diffusion–reaction equation:

∂
𝐶
∂
𝑡
+
𝑢
⋅
∇
𝐶
=
∇
⋅
(
𝐷
(
𝑥
)
∇
𝐶
)
−
𝑅
(
𝐶
,
𝑥
)
∂t
∂C
	​

+u⋅∇C=∇⋅(D(x)∇C)−R(C,x)

𝑢
u is fluid velocity (advection) — in capillary lumen this is blood flow; in brain ECS advection is minor.

𝐷
(
𝑥
)
D(x) is spatially varying diffusion coefficient (plasma, barrier interior if modeled as volume, brain interstitium).

𝑅
(
𝐶
,
𝑥
)
R(C,x) is reaction/clearance term: metabolism, uptake, or efflux sinks/sources.

In our code we used this form. Efflux inside barrier is modeled as a sink: 
𝑅
=
efflux
(
𝐶
)
R=efflux(C).

B. Efflux as Michaelis–Menten (surface or volumetric)

Efflux via transporter can be represented as:

Surface flux per area (more correct for transporters localized to membrane):

𝐽
efflux
=
𝑉
max
⁡
 
𝐶
free
𝐾
𝑚
+
𝐶
free
[
mol
/
(
m
2
⋅
s
)
]
J
efflux
	​

=
K
m
	​

+C
free
	​

V
max
	​

C
free
	​

	​

[mol/(m
2
⋅s)]

Convert to concentration change in neighboring compartment by dividing by cell volume / area.

Volumetric sink (what we used as simplification):

𝑅
(
𝐶
)
=
1
barrier
⋅
𝑉
max
⁡
 
𝐶
free
𝐾
𝑚
+
𝐶
free
[
mol
/
(
m
3
⋅
s
)
]
R(C)=1
barrier
	​

⋅
K
m
	​

+C
free
	​

V
max
	​

C
free
	​

	​

[mol/(m
3
⋅s)]

where 
1
barrier
1
barrier
	​

 selects barrier region.

When modeling more precisely you must scale Vmax (often measured in nmol/min/mg protein) to flux per area using protein density and membrane area.

C. Interface (thin barrier) permeability — PS or P formulation

Because endothelial thickness 
𝑑
d is small, a thin-interface model is often clearer:

Net flux across barrier interface:

𝐽
=
𝑃
⋅
(
𝐶
blood,free
−
𝐶
brain,free
)
[
mol
/
(
m
2
⋅
s
)
]
J=P⋅(C
blood,free
	​

−C
brain,free
	​

)[mol/(m
2
⋅s)]

where 
𝑃
P is permeability (m/s) and 
𝐶
free
C
free
	​

 are free (unbound) concentrations.

Relation to membrane properties (solubility-diffusion):

𝑃
=
𝐾
 
𝐷
𝑚
𝑑
P=
d
KD
m
	​

	​


𝐾
K = partition coefficient (membrane↔water), 
𝐷
𝑚
D
m
	​

 membrane diffusion, 
𝑑
d membrane thickness.

In finite-volume discretization, update concentrations adjacent to interface by:

Δ
𝐶
left
=
−
𝐽
𝐴
Δ
𝑡
𝑉
left
,
Δ
𝐶
right
=
𝐽
𝐴
Δ
𝑡
𝑉
right
ΔC
left
	​

=−
V
left
	​

JAΔt
	​

,ΔC
right
	​

=
V
right
	​

JAΔt
	​


with area 
𝐴
A, cell volume 
𝑉
V.

This interface approach is more physically accurate than making the barrier simply a low-D block, and is recommended as you refine.

D. Plasma protein binding

Only unbound concentration drives flux. If total plasma concentration is 
𝐶
tot
C
tot
	​

 and fraction unbound is 
𝑓
𝑢
f
u
	​

:

𝐶
free
=
𝑓
𝑢
⋅
𝐶
tot
.
C
free
	​

=f
u
	​

⋅C
tot
	​

.

On the brain side there is also an unbound fraction 
𝑓
𝑢
,
𝑏
𝑟
𝑎
𝑖
𝑛
f
u,brain
	​

 (free in brain ECF) which affects measurement and effect.

5. How the biology maps to your software (both FD solver and PINN)
Finite-difference code

Domain split in x into vessel → barrier → brain.

D_map = spatially varying diffusion coefficient approximates different media. Pores/patches are implemented by local D increases.

Efflux modeled as local sink in barrier (efflux[mask_barrier] = Vmax * C / (Km + C)). This is a volumetric approximation.

Dirichlet blood source: vessel region is fixed to source concentration (represents well-mixed blood reservoir).

Visualization: slices (x–y mid-z) show blood→brain gradient and localized pore-mediated leakage.

Limitations: volumetric efflux approximates a surface transporter; boundary-interface PS flux is more physically correct.

PINN (PyTorch) implementation

The neural network represents 
𝐶
(
𝑥
,
𝑦
,
𝑧
,
𝑡
)
C(x,y,z,t).

The PDE residual computed by autograd enforces:

res
=
𝐶
𝑡
+
𝑢
(
𝑥
)
𝐶
𝑥
−
𝐷
(
𝑥
)
Δ
𝐶
+
efflux
(
𝐶
)
≈
0
res=C
t
	​

+u(x)C
x
	​

−D(x)ΔC+efflux(C)≈0

using collocation points.

Boundary/inlet conditions and initial conditions are added as loss terms.

Mixed precision and GPU use accelerate training.

Mapping notes:

The PINN allows you to enforce PDE laws continuously in space–time and makes parameter estimation (e.g., Vmax, Km) possible by adding a parameter loss term.

The PINN uses volumetric efflux; switching to interface PS flux requires introducing surface collocation and treating jumps — doable but requires more careful residual terms.

6. Typical parameter ranges and units (orders of magnitude)

These are approximate ranges — use literature values where available.

Diffusion coefficient 
𝐷
D (water, small molecules): 
10
−
9
10
−9
 — 
10
−
10
 
m
2
/
s
10
−10
 m
2
/s. Brain ECS effective D ≈ 
0.5
 ⁣
−
 ⁣
0.8
×
0.5−0.8× free D due to tortuosity.

Membrane diffusion 
𝐷
𝑚
D
m
	​

: 
10
−
11
 ⁣
−
 ⁣
10
−
12
 
m
2
/
s
10
−11
−10
−12
 m
2
/s (depends on lipid).

Membrane thickness 
𝑑
d: 
2
 ⁣
−
 ⁣
8
 nm
2−8 nm for lipid bilayer; effective endothelial thickness is larger (~0.2–1 µm) if you include cytoplasm.

Permeability 
𝑃
P: varies widely. Small lipophilic drugs: 
10
−
5
 ⁣
−
 ⁣
10
−
7
 m/s
10
−5
−10
−7
 m/s. Hydrophilic drugs: much lower. PAMPA/PAMPA-BBB and cell monolayer assays report P in these ranges.

Vmax (per area) for efflux: depends on assay. Expressed as mol/(m²·s) when scaled. Raw in vitro often nmol/min/mg protein — needs conversion. Orders of magnitude are highly variable; treat as tunable.

Km: often in µM–mM (10^{-6}–10^{-3} mol/m^3) depending on transporter.

Plasma unbound fraction fu: from ~0.05 (highly protein bound) to ~0.9 (mostly unbound). Fentanyl ~0.1–0.2; morphine ~0.3–0.7 depending on source.

Units consistency: choose SI (m, s, mol/m^3) and keep D in m^2/s, P in m/s, concentrations in mol/m^3. If you keep dimensionless demo units, record conversion factors.

7. Modeling choices, pros/cons, and what you need for “medical-grade” predictions
Key modeling choices and trade-offs

Volumetric low-D barrier vs thin-interface PS flux

Volumetric low-D is simple but conflates barrier thickness with diffusivity.

Thin-interface with explicit P is more realistic; you can compute P from MD or measure P in vitro and plug directly.

Efflux as volumetric sink vs surface flux

Volumetric sink is easier inside FD grids and PINN residuals.

Surface flux is physiologically correct (transporters are membrane-bound); requires surface terms or matching conditions.

Fixed blood reservoir vs dynamic perfusion

Fixing vessel concentration is simple (Dirichlet). For time-resolved PK, couple with an advective 1D blood model / perfusion term.

Explicit FD vs implicit solvers vs PINN

Explicit FD is simple but stability-limited by dt and grid spacing.

Implicit (Crank–Nicolson, ADI) allows larger dt at cost of solving linear systems.

PINNs offer mesh-free approximation and parameter inference but require substantial compute and careful training.

Data required for higher-fidelity (clinical-level) predictions

Measured P (permeability) or PS from in vitro BBB assays or MD-derived PMF→P computations for each drug.

Transporter kinetics: Vmax and Km for P-gp/BCRP measured under comparable conditions, scaled to flux per membrane area.

Unbound fraction fu in plasma and fu,brain (unbound fraction in brain interstitial fluid).

Perfusion / flow rates for capillary segments (for advection modeling).

Brain clearance rates (glymphatic / interstitial clearance) if modeling long times.

Geometry / surface area of capillary bed (to convert surface fluxes to compartment changes).

Validation datasets: brain/plasma time series from microdialysis, PET, or in vivo PK.

8. Practical examples of improvements you can make (actionable)

Replace volumetric efflux with interface flux + membrane transporter: treat transporter flux at the vessel/barrier face and implement J_efflux as part of boundary conditions.

Add plasma protein binding explicitly and track both total and free concentrations. Use C_free = fu * C_total in flux terms.

Couple 3D tissue with a 1D advective blood channel so inlet concentration evolves with time instead of being fixed.

If you want to infer unknown parameters (P, Vmax, Km), extend the PINN to treat them as trainable parameters and include experimental observations in the loss.

For speed and scale, port the FD solver to PyTorch tensors and run on GPU — that's faster than pure NumPy and lets you integrate with model inversion.

# Modelling biochemical processes: kinetics, CSTR reactors, bioreactors and spectral speciation

Parameter estimation and model selection for chemical and biochemical process models in Python.
MSc student project, Gdańsk University of Technology, 2026. Course: *Modelling of Biochemical Processes*.

![Bioreactor fit](results/figures/lab3_bioreactor_default_fit_XSP.png)

## What's inside

- **Reaction kinetics (lab 2):**
  - least-squares fitting of rate constants for A → C and A → B → C from noisy absorbance data
  - bootstrap confidence intervals
  - model comparison of free vs constrained (k<sub>max</sub>) fits
- **CSTR reactors (lab 3):**
  - steady-state and dynamic models for A → P and A → B → C
  - conversion vs residence time
  - fitting rate constants to outlet concentrations
- **Bioreactor (lab 3):** Monod-kinetics chemostat (biomass X, substrate S, product P):
  - fitting μ<sub>max</sub> and K<sub>S</sub> with bootstrap
  - effect of measurement error
  - washout analysis vs dilution rate D
- **Spectral speciation (labs 4–5):**
  - PCA to determine the number of acid–base forms
  - fitting species fractions and **pKa** values, with leave-one-out validation, for 3-, 4- and 5-form systems (Ledakrin, Symadex, C-2045)
  - self-association and complexation models

<p>
<img src="results/figures/lab2_ABC_C_fit.png" width="49%">
<img src="results/figures/lab4_example-Symadex_fractions_4form.png" width="49%">
</p>

## Repository structure

```
mpb_compute.py   batch driver: runs all exercises, fits models, saves figures/tables/JSON
results/         generated data (CSV), figures (PNG), results*.json
reports/         lab reports (PL): kinetics + reactors, bioreactors, spectral speciation
```

## How to run

`mpb_compute.py` imports the lab programs that the lecturer provided for the course. Those programs are not redistributed here. To reproduce the results, put them in a `Laby/` folder (or point `MPB_LABS_DIR` at them) and run:

```bash
pip install -r requirements.txt
python mpb_compute.py
```

All generated outputs are already committed in `results/`.

**Data:** none of the data are sensitive.
- The kinetics, reactor and bioreactor "measurements" are synthetic: the lab programs simulate them with added measurement noise.
- The spectral example datasets (Ledakrin, Symadex, C-2045) and the lab programs are course materials for *Modelling of Biochemical Processes* at Gdańsk University of Technology. They are available to course participants from the instructors.

**Tools:** Python (NumPy, SciPy `least_squares` / `solve_ivp`, pandas, matplotlib).

## AI assistance

The code in this repository was written with the help of AI tools (large language models). Defining the tasks, running the analyses, and checking and interpreting the results were my part of the work.

---

## 🇵🇱 Opis po polsku

Ćwiczenia z *Modelowania procesów biochemicznych*:
- kinetyka reakcji (dopasowanie stałych szybkości, bootstrap, wybór modelu)
- reaktory CSTR (stan ustalony i dynamika)
- bioreaktor z kinetyką Monoda (μ<sub>max</sub>, K<sub>S</sub>, wymywanie)
- specjacja z danych spektrofotometrycznych (PCA, pKa, walidacja LOO)

Sprawozdania są w folderze `reports/`.

Projekt studencki (studia II stopnia), Politechnika Gdańska, 2026.

**Wsparcie AI:** kod w tym repozytorium powstał z pomocą narzędzi AI (dużych modeli językowych). Określenie zadań, uruchamianie analiz oraz sprawdzenie i interpretacja wyników były moją częścią pracy.

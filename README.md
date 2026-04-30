## Official code for *"Whole-Brain Modeling Reveals Subtype-Specific Regulation for Adolescent Major Depressive Disorder"*.

---

### Project Structure 📁


```text
├── datas/              # input datasets
│   ├── dep1/
│   └── dep2/
├── modeling/           # modeling code
│   └── functions/
├── regulation/         # regulation analysis
│   └── dataSave/       # intermediate outputs (provided via GitHub Releases)
├── results/            # figures and outputs
├── spm/                # SPM-related scripts
└── subtyping/          # subtype analysis
```

---

### Quick Start 🚀

This section provides a minimal pipeline to reproduce the main results.

**1. Run Whole-Brain Modeling**

To perform large-scale whole-brain modeling for the two identified subtypes, run:

```
modeling/Modeling_dep1.m
modeling/Modeling_dep2.m
```

These scripts will generate the core modeling results for each subtype.

**2. Run Regulation Analysis**

After modeling, run the scripts in the *regulation* folder with the prefix:

```
fic_*.m
```

These scripts implement different regulatory strategies and will produce the corresponding intervention results.

**3. Reproduce Figures**

The scripts in the *results* folder can be used to reproduce the figures reported in the paper.

⚠️ Notes

Make sure all required data files are placed in:

```
regulation/dataSave/
```

Some scripts depend on intermediate results generated in previous steps.
Due to dataset restrictions, full end-to-end reproduction may require the original data.

---

### Data Availability & Usage Notes

Some files are excluded via `.gitignore` due to GitHub size limitations. 

They are provided via GitHub Releases:

https://github.com/xinzeng1996/AMDDsubtypes/releases

* The excluded files mainly correspond to **intermediate outputs from the `regulation` module**.
* To **reproduce figures quickly**, please download these files and place them into:

```
regulation/dataSave/
```

---

### BANDA Dataset Restrictions ⚠️

This project is based on the **BANDA dataset**, which requires a formal application for access.

* To comply with data usage regulations, **raw BANDA data is NOT included** in this repository.
* Instead, we provide:

  * Average **Functional Connectivity (FC)** matrices
  * Average **Structural Connectivity (SC)** matrices
    for:

    * Healthy controls
    * Two identified subtypes

---

### Reproducibility Notes

* Due to the absence of raw data, **some scripts may not run end-to-end**.
* If you want to test the pipeline:

  * You can generate **synthetic data with matching dimensions**
  * This allows the code structure to run correctly

⚠️ However, results obtained from synthetic data **will NOT reflect real findings**.


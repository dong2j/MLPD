### This repository contains the code associated with the manuscript:

**"Nonlinearity of the Post-Spinel Transition and its Expression in Slabs and Plumes Worldwide"**

#### Manuscript Authors
- Junjie Dong
- Rebecca A. Fischer
- Lars P. Stixrude
- Matthew C. Brennan
- Kierstin Daviau
- Terry-Ann Suer
- Katlyn M. Turner
- Yue Meng
- Vitali B. Prakapenka

#### Code Author
- Junjie Dong

## Summary
This repository consists of two main parts:
1. **Model Selection with Supervised Learning**
2. **Phase Diagram Construction**

The code analyzes phase equilibria observations and computes a globally optimized phase diagram through multi-class logit regression and supervised learning. The code can be used generally, but the construction of phase relations in Mg<sub>2</sub>SiO<sub>4</sub> at mantle transition zone conditions is used as a benchmark case to ensure reproducibility and ease of use, and the data on phase equilibria observations of Mg<sub>2</sub>SiO<sub>4</sub> are provided.

## Usage

Follow the instructions below or in the Python scripts to reproduce the analyses discussed in the manuscript.

## Prerequisites
The primary packages used in this code are:

- `matplotlib=3.5.0`
- `numpy=1.19.5`
- `pandas=1.3.5`
- `python=3.9.20`
- `scikit-learn=0.23.2`
- `scipy=1.10.0`

For full package compatibility, you can use the provided conda environment file or requirements file.

## Getting Started
1. Set up the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate mlpd

2. Alternatively, you can install dependencies using pip:

   ```bash
   pip install -r requirements.txt

3. To tune the hyperparameters for the model, run 'logit-reg-model-selection.py':

      ```bash
   python logit-reg-model-selection.py

4. You can find the optimized degree and other hyperparameters in the 'log.txt' file. After updating the optimized degree and other hyperparameters, run 'logit-reg-fit.py':

      ```bash
   python logit-reg-fit.py

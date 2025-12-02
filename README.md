# CFDP: Causal Front-Door Prompting for Debiasing Large Language Models

This repository contains the implementation of **CFDP (Causal Front-Door Prompting)** used in our paper:

The repository includes prompt templates, CFDP inference code, a small set of toy examples, and scripts to reproduce the main tables and figures in the paper.

---

## 📁 Repository Structure

<pre>
project_root/
├── CFDP                  
│   ├── demo.py/           
│   ├── functions.py/   
│   ├── generate_input.py/ 
│   └── generate_cot_wrong.py/      
│
├── data/                  # Toy examples 
│
├── scripts/
│   └── run_bbq.sh         
│
├── baseline/               
│
├── README.md
└── requirements.txt
</pre>


## 📦 Datasets
We evaluate CFDP on:

BBQ (Parrish et al., 2022)
StereoSet (Nadeem et al., 2021)

Due to licensing, datasets are not included.
We provide toy samples under data/ for quick functional testing.

## 🚀 run Demo
bash run.sh

# 🚉 Victorian Travel Mode Prediction 🚚
**Group data science project analysing the ability to predict the travel mode of Victorians based on socio-demographic features. Complete ML pipeline with business-style presentation and academic report.**

## 📋 Overview
This project investigates the socioeconomic drivers of travel mode choice in Victoria using the 2023–2024 Victorian Integrated Survey of Travel and Activity (VISTA). Drawing on a sample of 2,404 individuals, we analysed 27 socioeconomic and demographic variables to determine which factors most strongly predict transport behaviour.

Key research questions:
1. What socioeconomic factors are most important in predicting travel mode choice?
2. Are there distinct workforce profiles — by occupation, income, vehicle ownership, and suburb — that predict transport mode choice?

Key findings: Vehicle ownership, household income, and home location emerged as the strongest predictors. Random Forest achieved the best predictive performance, and clustering revealed four distinct workforce profiles with sharply different transport behaviours.

## 💻 My Role and Contributions
1. Implemented supervised machine learning models namely Logistic Regression, Decision Tree and Random Forest models; performed hyperparameter tuning with k-fold cross-validation
2. Developed visual presentation through Canva and Microsoft Powerpoint through visual tools, and Seaborn generated distribution graphs, Gini Heatmaps, and classification reports
3. Discussion of Machine Learning considerations and dicussion in report
4. Team Collaboration: Organised regular meetings and catchups to discuss project expecataions, deadlines and task division 

## 📊 Key Results
| Model | Accuracy (macro avg) | Best F1-Score | 
|-------|----------|-----------|
| Random Forest (Tuned) | 0.796 | 0.73 (weighted)
| Decision Tree | 0.38 | - | 
| Logistic Regression | 0.45 | - |

Strongest predictors: Vehicle ownership (Cramér's V = 0.29), home location (0.26), household income (0.24)

Clustering: 4 distinct workforce profiles; public transport use highest among lower‑income service workers, lowest among high‑income professionals and trade workers

Key insight: Private transport dominates all socioeconomic groups — a finding with direct policy implications for sustainable transport planning

## ⚒️ Limitations and Considerations Considered 
1. Dataset imbalance (private transport overrepresented) limited prediction of public/active modes
2. Categorical binning (e.g., income as Low/Medium/High) sacrificed some granular detail
3. K‑means assumes spherical clusters; real behaviour is more fluid
4. Future work: continuous variables, multi‑modal trip capture, larger dataset

## 📝 References 
This project was completed on behalf of a project for a Data Science University of Melbourne Subject (COMP20008)
The VISTA dataset is provided by Transport Victoria: opendata.transport.vic.gov.au

DeepSeek – Assisted with code structuring, debugging, and README formatting

ChatGPT – Helped refine methodology explanations and presentation structure

All analysis, model choices, interpretations, and final conclusions are our own.

## 🧱 Repository Structure

```plaintext
victorian-travel-mode-prediction/
├── README.md
├── data/                              # VISTA source data (see link above)
│   ├── Raw data
│   ├── Data Stage 1 
│   ├── Data Stage 2                     
├── files/
│   ├── vista_filter_merge.py
│   ├── Code Stage 1/
│   |   ├── ML_tree+random_forest.py
│   |   ├── ses_vs_mode_analysis.py  
│   ├── Code Stage 2/
│   |   ├── Machine_Learning_students2.py
│   |   ├── Machine_Learning_workforce.py
├── outputs/
│   ├── presentation.pdf
│   └── report.pdf
└── requirements.txt
```

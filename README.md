# Crohn's Disease Diagnosis Using Metabolomics & Machine Learning

This study explores a non-invasive approach to diagnosing **Crohn's disease (CD)** using metabolomics and machine learning. **Gas chromatography-mass spectrometry (GC-MS)** was applied to different biological samples (**breath, blood, urine, and faeces**) and classified using **Support Vector Machines (SVMs) and Random Forest (RF)** models.

## 🧪 Key Findings
- **Faecal samples** provided the most reliable diagnostic signature.
- **Random Forest (RF)** performed best, achieving an **AUROC score of 0.824**.
- **Faecal metabolomics** shows strong potential for non-invasive CD diagnostics.

## 🏗️ Methodology
- **Data**: GC-MS chromatograms of different sample types.
- **Models**: SVM & RF trained on preprocessed data.
- **Evaluation Metrics**: Accuracy, specificity, sensitivity, F1-score, AUROC.
- **Validation**: Bootstrap resampling & permutation testing.

## 📊 Results Summary
| Sample Type | Best Model | Accuracy | AUROC |
|-------------|------------|----------|--------|
| **Faecal**  | RF         | 74%      | **0.824** |
| **Breath**  | RF         | 67%      | 0.774  |
| **Blood**   | SVM        | 40%      | 0.598  |
| **Urine**   | RF        | 55%      | 0.505  |

- **Faecal samples**: Best diagnostic accuracy.
- **Breath samples**: Moderate diagnostic potential.
- **Blood & Urine samples**: Poor diagnostic performance.

## ✅ Conclusion
This study highlights **faecal metabolomics + machine learning** as a promising tool for **non-invasive Crohn’s disease diagnosis**. Future improvements include:
- Feature selection & dimensionality reduction.
- Integration of multiple sample types (e.g., faecal + breath).
- Inclusion of transcriptomic/proteomic data.

## ⚙️ Installation & Usage
To run this analysis, ensure you have **Python 3.13** and the required libraries installed.

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## 📜 References

Application of gas chromatography mass spectrometry (GC–MS) in conjunction with multivariate classification for the diagnosis of gastrointestinal diseases (Cauchi et al., 2014)

## ⚠️ License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
See https://creativecommons.org/licenses/by-nc/4.0/ for details.


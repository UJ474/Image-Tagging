# AI Assistant Prompt Template for Team Members

When working with an AI coding assistant (such as Antigravity, Claude, or Copilot) to implement your part, provide this prompt:

```text
Read docs/team/README.md and my individual guide:
<INSERT_YOUR_GUIDE_PATH_HERE>

Guide references:
- Anant: docs/team/01_Anant_Setup_Data_MasterNotebook.md
- Kartik: docs/team/02_Kartik_Feature_Extraction_HOG.md
- Krishna: docs/team/03_Krishna_Logistic_Regression.md
- Vipul: docs/team/04_Vipul_KNN_Classifier.md
- Aditya: docs/team/05_Aditya_Random_Forest.md
- Utkarsh: docs/team/06_Utkarsh_SVM_Error_Analysis.md

Instructions:
1. Implement my assigned work on the branch specified in my guide.
2. Follow the shared interfaces, data shapes, and file ownership rules.
3. If a teammate's module is not yet merged, use synthetic data or scikit-learn mock data to develop and test independently without blocking.
4. Run all verification checks specified in my guide.
5. Summarize changed files, test outputs, and prepare my contribution for PR review.
```
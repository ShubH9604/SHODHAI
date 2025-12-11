# SHODHAI – Loan Approval Optimization (ML + RL)

This project focuses on improving loan approval decisions using the LendingClub 2007–2018 dataset. The objective is to predict loan default risk using a Deep Learning model and then optimize loan approval decisions through a reward-based policy that maximizes financial return.

### 📌 Project Summary
Using borrower financial attributes and loan history, a supervised Deep Learning model was trained to predict the probability of default.  
Because prediction alone does not maximize profit, an RL-inspired threshold policy was developed to choose when a loan should be approved based on expected monetary gain.

### 📊 Key Results
- **Supervised Deep Learning Model:**  
  - AUC: **0.7294**  
  - F1 Score: **0.0390**  
  (Low F1 is expected due to heavy class imbalance.)

- **RL-Inspired Policy (Profit Optimization):**  
  - Best Approval Threshold: **0.70**  
  - Total Profit on Test Set: **27,888.17**  
  - Average Reward per Loan: **0.06168**

### 🎯 Conclusion
The Deep Learning model provides strong risk ranking capability, while the RL-style policy directly improves financial returns by approving loans with positive expected value. This approach reflects real-world credit decision optimization used in fintech and lending systems.

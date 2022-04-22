# Capstone Project Team Falcon

<style>
    p {align: center;
    font-size:30px}
</style>

<p> Peer-to-Peer-Lending: Predicting Bad Loans with Machine Learning </p>

## Table of Contents
# 1.	Introduction 
### a. Background 
Peer-to-peer lending is an innovative FinTech product that disrupts the entire banking industry. Traditionally, banks play an intermediary role between borrowers and lenders. Banks collect money from lenders as deposits or savings at a lower rate, then issue loans to borrowers at a higher rate. To protect lenders’ money, banks execute the professional due diligence that distinguishes good borrowers from bad ones. Recently, peer-to-peer lending fintech firms invented a platform that connects lenders with borrowers without these intermediary banks. It gives an opportunity for lenders to earn more returns and for borrowers to get loans in a cheaper and quicker way. 
### b.Audience and Motivation 
Even though peer-to-peer lending platforms give tremendous opportunities to both parties, the solution itself has its own drawbacks. The professional due diligence work is on investors' shoulders.  Not all investors have the professional knowledge like banks to distinguish good borrowers from bad borrowers. Investors are regular people who want to earn a high return on their loans.  If they invest in a bad loan, they would lose their money.  
Prosper Lending enables investors to browse consumer loan applications containing the applicants loan details, credit history, etc in order to make determinations as to which loans to fund. Loans from applicants deemed at higher risk of default, such as those with lower FICO scores, will typically carry higher interest rates and therefore the potential to yield a higher return on investment for the investor. Alternatively, lower risk of default loans will usually carry lower interest rates. Therefore, a Machine Learning model that could accurately predict the default risk of a loan using the available data on Prosper Lending could help investors maximize their investment returns by identifying the loans worth investing. 

# 2.	Data Acquisition 
### a. Data source
For our initial datasets, we use two separate datasets from https://www.prosper.com. The first data is the Listing data.  The listing data includes information about all applicants. It includes personal features that can estimate the credit quality of borrowers. The are 886 qualitative and quantitative features per customer. The second data is the loan data.  The loan data includes actual loan information about applicants whose listings are approved and who received a loan from the peer-to-peer platform. The loan data has information about loan characteristics such as loan interest rate, loan amount, loan maturities, principal payments, balance, etc. Additionally, the loan data include the loan status if the loan is defaulted or not. Some of these features in the data are only relevant after loan issuance and therefore not available to investors at the time of investing. We used the Prosper Data Dictionary (https://www.prosper.com/Downloads/Services/Documentation/ProsperDataExport_Details.html) to better undertsnad the features available to investors.

# 3.	Methodology 
### a.	Data Preprocessing 
To begin, we executed a simple exploratory analysis using the data dictionary to find unique key identifiers to merge the data. Both datasets have their unique key identifiers however, there is no direct connection between the unique identifiers. We decided to merge the two datasets with 6 common variables (loan_origination_date, amount_borrowed, borrower_rate, prosper_rating, term, and co_borrower_application). We kept only unique merged data as there might be two same borrowers who requested the same amount of loan and received the same loan interest and maturity. After merging we had more that 80% unique rows.

 After merging the data, we dropped some variables based on these conditions;
  - variables with >50% of missing data 
  - variables that would introduce data leakage (eg age_in_months)
  - variables unavailable until loan issuance (eg days_past_due, late_fees_paid)
  - variables that include only one feature (eg scorex)

How we replaced null values in other variables like stated_montly_income, monthly_payment, etc can be found in our code. After all this our dataframe had shape of (62673, 28) 

### b. Data Exploration
We created several exploratory data analysis plots using seaborn and plotly's Python library in order to build high-level intuition of some of the relationships between different variables.

A plot of distribution of borrowed amount by loan term
(*** add image)
The distribution is right skewed and also consistent with the fact that more people tend to pay smaller loans quickly

A plot of defaulting customers
(*** add image)
We can see that a small percentage of customers are defaulting. This is a typical imbalanced dataset where the default rate is much lower than the successful completed rate. Machine learning algorithms applied to imbalanced classification datasets can produce biased predictions with misleading accuracies. We will use the SMOTE library to try mitigate this problem.

A correlation matrix plot
(*** add image)
To avoid multicollinearity in the data, both numeric and categorial variables exhibiting high degrees of multicollinearity >0.85 were dropped from the dataset

### c. SMOTE - Synthetic Minority Oversampling Technique
We use the Synthetic Minority Oversampling Technique (SMOTE), which is a widely adopted approach, to address the class imbalance dataset. SMOTE uses bootstrapping and k-nearest neighnors to construct new minority class instances by transforming data based on feature space (rather than data space) similarities from minority samples. SMOTE performs a combination of oversampling and undersampling to construct a balanced dataset.

For our puposes we oversmapled the minority class to have 10% the number of examples of the majority class. We then used random undersampling to reduce the number of examples in the majority class to have 80% more than the minority class. Ratios are what works best for your data. We believe this ratio works best for us. As shown below we successfully generated a balanced dataset using smote.
(*** add image)
![image](https://user-images.githubusercontent.com/86815494/164077554-686aa158-50af-4e2d-a786-be5c4e3da62a.png) 
![image](https://user-images.githubusercontent.com/86815494/164077567-3fcfddd5-946a-421a-b243-3df151d6571b.png)

### d.	Feature Engineering
We created two new features EMI (Equated Monthly Installment) and Balance_Income.

EMI is the monthly amount to be paid by the applicant to repay the loan. The idea behind creating this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate the EMI by taking the ratio of the loan amount with respect to the loan amount term.
**EMI = amount_borrowed / term**

Balance Income — This is the income left after the EMI has been paid. The idea behind creating this variable is that if this value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval. The distribution of this variable was highly skewed so we took the log transformation of it before feeding it to the machine learning model.
**balance_income = monthly_income - EMI**

![image](https://user-images.githubusercontent.com/86815494/164076511-57593284-dbff-43e6-94b6-e8e8af208269.png)
![image](https://user-images.githubusercontent.com/86815494/164076538-d2ce5778-67f9-4bd6-beb6-f01507bf0893.png)
![image](https://user-images.githubusercontent.com/86815494/164076559-d2b8e42a-fda8-475d-a409-30aa8e1d6de5.png)


As we see from the distribution, the engineered variables can be a good explainer in our model.

### c.	Machine Learning Models 
To be able to truly understand and then improve our model’s performance, we established a baseline for the data that we have. We employed a **dummy classifier** as our baseline model. This classifier serves as a simple baseline to compare against other more complex classifiers. The other models we were interested in testing for this problem were **Logistic Regression, Random Forest, and XGBoost classifier.** For our Random Forest and XGBoost classifier we used grid search to get the optimized values of hyper parameters. We observed the performance of our models through the confusion matrix plot and the model classification report (our interest was Accuracy, Precision, Recall, and F1-score). We primarily care about correctly identifying **bad loans**, so the **recall score** will be of much importance. To show the performance of our model, we graphically depicted the Area under the ROC Curve (AUC). By analogy, the Higher the AUC, the better the model is at distinguishing between customers who will default or not.

# 4.	Results and Discussion 
### Dummy Classifier
As we stated earlier, the dummy classifier was to just have a baseline to compare our model with. When loans are predicted to be bad they are bad 86% of the time. Of all the true bad-loans this model identified 50% of them. We observed an F1-score of 63% for bad loans.

### Logistic regression 
Of all the true bad-loans the logistic regression model identified 68% of them and when the loans are predicted to be bad they are bad 92% of the time. By comparing with our baseline model we saw an improvement in the model performance. An F1-score of 78% was observed for bad loans. The model had an accuracy of 68%.

![image](https://user-images.githubusercontent.com/86815494/164074647-c8cc1a54-3131-4d91-b648-3c1c15b62591.png)
![image](https://user-images.githubusercontent.com/86815494/164074679-85f01aef-8403-4098-812b-20b3e8566281.png)


### Random Forest
Our Random Forest model identified 70% of true bad-loans with a 92% precision. There was a slight improvement in the model accuracy (69%). We also observed a slight improvement in the F1-score (79%). The Random Forest model seems to be the best performing model so far.

![image](https://user-images.githubusercontent.com/86815494/164074753-78550b0b-fbd2-4ee3-a654-8c396564bfe6.png)
![image](https://user-images.githubusercontent.com/86815494/164074774-40fc4eae-030f-420e-a854-da4c2557d63a.png)

### XGBoost Classifier
The XGBoost model reported an accuracy of 71% (best performing model out of the lot). Of all the true bad-loans this model identified 72% of them and when the loans are predicted to be bad they are bad 92% of the time. The harmonic-mean of the Precision and Recall was 81% of bad-loans. We therefore concluded that the XGBoost is our best performing model to do a model explainability.

![image](https://user-images.githubusercontent.com/86815494/164074816-5db724bc-2f91-41a8-9fa0-f4cbc967133a.png)
![image](https://user-images.githubusercontent.com/86815494/164074843-0915d4de-9266-47f9-a5c5-fb6cf7e0b6b5.png)

 # 5.   Model Explainability (SHapley Additive exPlanations - SHAP)
We used the SHAP Python package to interpret our model. SHAP is an increasingly popular method used for interpretable machine learning.
### Feature Contributions (force plot)
We used a force plot to summarize how each feature contributes to an individual prediction. (insert force plot image and add explanation)

### Global Explanations and Feature Importance
We put local explanations described above together to get a **global explanation**. And because of the axiomatic assumptions of SHAP, global SHAP explanations can be more reliable that other measures. (insert global plot image and add explanation)

# 6.	Conclusion and Future Work 
Predicting the occurrences of loan default in a peer-to-peer lending platform is crucial and challenging task. More accurate prediction models would be highly beneficial since the failure of a peer-to-peer lending platform could trigger a series of financial risks. Our project shows that machine learning methods have broad application prospects in the prediction of P2P loan default.

For future work, we would wnat to deploy our model and have a real-time machine learning predictions

Refrences: 
https://towardsdatascience.com/p2p-lending-platform-data-analysis-exploratory-data-analysis-in-r-part-1-32eb3f41ab16




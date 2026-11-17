# Predicting Self-Reported Diagnosed Diabetes in Adults from 2024 BRFSS Data

## Comparing Logistic Regression, Random Forest, and Gradient Boosting

**Author:** Jiadi Wu

**Course:** DSCI 7000, M.S. in Artificial Intelligence, Hawaii Pacific University

**Data:** CDC 2024 Behavioral Risk Factor Surveillance System (BRFSS), combined landline and cell-phone public file (LLCP2024XPT.zip, released August 2025)

## Abstract

**Objective:** To evaluate how well seven routine survey features identify self-reported diagnosed diabetes among adult respondents in the 2024 BRFSS, and to compare logistic regression, random forest, and gradient boosting. **Methods:** Analysis sample 441,934; stratified 70/30 train–test split by outcome; hyperparameters tuned by five-fold cross-validation on average precision (AP); thresholds selected by out-of-fold F1. Description and association analyses use the complex survey design; the main prediction analysis is unweighted, with a weighted sensitivity analysis. **Results:** The three models' AP values are 0.332887, 0.335560, and 0.337267, with precision around 30%. Gradient boosting exceeds logistic regression by an absolute AP difference of 0.004380 (95% interval 0.002800–0.005962); under weighted evaluation, the design intervals for all three pairwise model differences include 0. The design-weighted prevalence is 12.98%; the adjusted odds ratios for obesity, age 65 and older, and no leisure-time physical activity are 3.51, 9.95, and 1.55, respectively. **Conclusions:** Seven features carry moderate ability to identify the label; the numerical gain from more complex algorithms is small and insufficient to demonstrate a substantive advantage. The models must not be used for clinical diagnosis, causal inference, or prediction of future onset.

**Keywords:** BRFSS; self-reported diagnosed diabetes; average precision; complex survey sampling; machine learning

## 1. Introduction

### 1.1 Background and Research Problem

Using survey information to identify population characteristics associated with a history of diabetes diagnosis is a concrete problem in public-health data analysis. Compared with blood tests and complete clinical histories, information such as age, BMI category, and physical activity is easier to obtain by questionnaire. Existing studies have built simple diabetes risk scores from such information, but different studies target different things: some predict future onset, some screen for undiagnosed diabetes, and others describe the distribution of people already diagnosed. Even when models use similar variables, their outputs are only comparable when the outcome definition and the target population match. Bang et al. (2009) used U.S. National Health and Nutrition Examination Survey data to develop a self-assessment score for identifying undiagnosed diabetes, showing that a small amount of non-laboratory information has some identifying value. Noble et al. (2011), in a systematic review of diabetes risk models, showed that developing a model is not the same as having a tool ready for practice — the validation population, measurement methods, and implementation conditions still matter. These studies motivate a more data-constrained question: with only a few routine survey features, how well can we identify whether a respondent reports a history of diabetes diagnosis?

The U.S. Behavioral Risk Factor Surveillance System (BRFSS) provides large-scale cross-sectional data for this question. The CDC's 2024 combined landline and cell-phone file contains 457,670 records and 345 variables, covering 49 states, the District of Columbia, and three territories; Tennessee was not included in that year's public file (CDC, 2025a). The standardized questionnaire and survey-design information make it suitable for model comparison, while also requiring researchers to handle self-reported measurement, geographic coverage, and survey weights carefully.

This report compares logistic regression, random forest, and gradient boosting on the same seven survey features. The focus is the identifying power of this information itself, and how much gain additional algorithmic complexity brings. If complex models improve only slightly over simple ones, their higher maintenance cost and less transparent model form should also enter the method choice; if the improvement is substantial, it must then be judged for stability and for whether it also improves probability calibration and performance across groups.

### 1.2 Outcome and Its Boundaries

The target variable is "self-reported diagnosed diabetes" — whether the respondent answered that a doctor had told them they have diabetes. It is simultaneously affected by actual disease status, access to medical care, prior diagnosis, memory, and willingness to answer. Answering "no" does not rule out undiagnosed diabetes, and answering "yes" is not a clinical re-examination by this study. What is predicted is therefore the diagnosis-history label in the survey record; the model's ability to discover truly undiagnosed disease cannot be computed from it.

The BRFSS diabetes question does not distinguish type 1 from type 2 diabetes in this report's outcome coding. Although much research on obesity, age, and physical activity focuses on type 2 diabetes, this report cannot equate those study populations with the current mixed self-reported label. Related literature is used to justify variable selection and evaluation methods; it does not support claiming that a dedicated type 2 diabetes prediction tool has been built.

The cross-sectional design also limits judgments about time order. Weight and activity are measured at the time of the survey, while diagnosis may have occurred earlier; patients may also have changed their lifestyle after diagnosis. Current features may therefore be related to disease onset, or may reflect changes after diagnosis. Throughout, predictive performance, conditional statistical association, and causal effect are kept distinct: higher feature importance is not interpreted as greater intervention benefit, and a current classification probability is not called future onset risk.

### 1.3 Research Questions and Hypotheses

The first research question asks whether the information is sufficient: among adult respondents with a valid outcome in the 2024 BRFSS, how well do seven survey features identify self-reported diagnosed diabetes? Baselines are set at random ranking, age alone, and age plus BMI category, to separate "better than a majority-class rule" from "better than a simple feature combination." The corresponding H1 is that the full-feature models' main evaluation metric beats random ranking and the age-plus-BMI-category baseline; whether it is supported must be judged together with the differences and their uncertainty.

The second research question concerns algorithm choice: under identical inputs, data splits, and evaluation rules, do logistic regression, random forest, and gradient boosting differ in performance, and does performance vary across population groups? H2 is that the three models' predictive performance differs. This report does not assume tree models must beat logistic regression, nor does it automatically interpret a statistically detectable small difference as a practically meaningful improvement. Positive rates and sample sizes across groups must also be reported, so that changes in PR metrics are not attributed entirely to model ability.

The third research question concerns interpretation: after adjusting for the other survey features, what are the conditional associations of BMI category, age group, and leisure-time physical activity with self-reported diagnosed status, and how much do model predictions depend on these features? H3 is that obesity, older age, and no leisure-time physical activity are associated with higher odds of self-reported diagnosis, and that shuffling these features degrades model performance. Adjusted odds ratios and permutation importance correspond to association direction and model dependence respectively; the two kinds of evidence cannot substitute for each other. The direction in the hypothesis does not require every age or BMI category to show a strictly monotone relationship.

### 1.4 Value of the Study and Report Structure

The value of this report is to place a low-information-cost feature set in a unified comparison framework and to state clearly what the results can and cannot answer. The seven features are BMI category, age group, leisure-time physical activity, sex, race or ethnicity, education, and household income. BMI is computed from height and weight information, so the inputs cannot simply be called "seven questions." Keeping this wording helps describe the potential data-collection burden accurately.

The study also keeps two distinct analytic goals: design-weighted descriptive statistics and association analyses address the covered eligible target population; the unweighted main prediction analysis addresses the distribution of the observed sample. Their difference is examined further through a weighted sensitivity analysis. Chapter 2 reviews the literature on simple risk scores, self-reported data, model comparison, and performance evaluation accordingly; Chapter 3 describes data processing, training and validation, and robustness checks; later chapters present results and discussion. This report proposes no new algorithm and does not use internal test results to claim the model is ready for clinical deployment.

## 2. Literature Review

### 2.1 Scope and Organization

This chapter is a narrative literature review organized around the research questions, covering simple diabetes assessment tools, BRFSS measurement quality, common classification models, and evaluation methods for prediction studies. The focus is on comparing study goals, data sources, and methodological limitations, and then explaining this report's design choices. No pre-registered systematic search, dual screening, or quantitative synthesis was conducted, so this chapter is not called a systematic review and does not claim to cover all machine-learning research on diabetes.

Comparing existing results first requires distinguishing outcome types. Predicting future incident diabetes needs follow-up and a clear time origin; screening for undiagnosed diabetes usually relies on test results to establish the outcome; self-reported diagnosis history depends on questionnaires and prior medical experience. The three kinds of studies can share ideas for variable selection, but they cannot be ranked by a single AUC value. The discussion below uses this distinction to assess which conclusions can support this report and which still need re-testing.

### 2.2 Simple Diabetes Assessment Tools and Their Conditions of Use

Lindström and Tuomilehto (2003) proposed a score using easily obtained information to predict type 2 diabetes risk, a representative example of non-laboratory assessment. Its significance is that it brings information-collection cost into model design: a model need not use every possibly relevant indicator to form a useful score around a limited set of routine features. However, the time order required for prospective risk prediction differs from identifying an existing diagnosis history in a single cross-sectional survey. This report borrows the low-burden design idea without transferring its scores, thresholds, or future-onset probabilities.

Bang et al. (2009) more directly show the development and validation path of a self-assessment tool in a U.S. sample. The study developed a model on NHANES 1999–2004 data and validated it on later NHANES and community-study data; age, sex, family history of diabetes, history of hypertension, obesity, and physical activity were associated with undiagnosed diabetes. Cross-data validation is closer to testing generalizability than a single random split, but its label includes test information and cannot be equated with the BRFSS self-reported diagnosis label. This report also does not include all of the same variables, so its validation conclusions cannot be inherited directly.

Noble et al. (2011) systematically reviewed risk models for future type 2 diabetes in adults, analyzing 94 models or scores in detail. The review makes the gap between the number of models and their actual usability an important issue: different studies' populations, predictors, and validation procedures are inconsistent. For this report, this means that adding one more performance table is not enough to constitute a research contribution. The more meaningful approach is to show the gain on the same label, the same features, and a shared test set, and to clearly bound how far it generalizes.

Together, these literatures support studying simple features, but they do not prove that any simple model can be used for diabetes screening. This report examines age, BMI category, and leisure activity as the key features, adding sociodemographic information, to test their added identifying value for the survey label. Without laboratory outcomes, family history, and similar information, even limited model performance cannot be used to declare non-laboratory assessment invalid overall; the conclusions apply only to the inputs and task defined here.

### 2.3 Measurement Quality and Selection Mechanisms of Self-Reported Data

Pierannunzi et al. (2013) reviewed studies assessing the reliability and validity of the BRFSS from 2004–2011. The results show that many survey items have usable test–retest consistency, but agreement with external standards such as physical measurements varies by item, with height, weight, and physical activity especially sensitive to measurement method. It follows neither that "the BRFSS is usable overall" implies "every variable is error-free," nor that consistent repeated answers mean the true state was measured accurately.

This matters directly for this report. BMI is computed from self-reported height and weight, and misreporting can move respondents across category boundaries; converting BMI to categories does not remove the original measurement error. EXERANY2 records whether there was any physical activity or exercise outside work in the past month; it does not fully measure exercise intensity, frequency, or occupational activity. This report therefore uses wording tied to the variables — "BMI category" and "any leisure-time physical activity" — rather than overstating them as precise BMI or total activity.

Diagnosis history also involves contact with medical care. People with the same disease may receive different labels because of different opportunities for testing, and undiagnosed people may enter the negative group. Models may therefore learn both disease-related features and differences related to prior diagnosis and reporting. The predictive contributions of income, education, or race/ethnicity variables thus cannot be directly interpreted as biological effects, let alone as evidence of essential group differences. This is a limitation on interpretation imposed by how the label is formed, not a claim that specific mechanisms have been identified.

Survey coverage and non-response also shape the sample. The CDC's 2024 documentation continues the cell-phone sample and iterative proportional fitting weighting design, and notes that Tennessee did not meet the minimum requirement for inclusion in the public data (CDC, 2025a). Weights can adjust for some known sampling and composition differences but cannot automatically correct all measurement error and unobserved selection. This report therefore uses the survey design information while retaining the limits on the target population, missing values, and label interpretation.

### 2.4 Basis for Comparing Logistic Regression and Tree Ensembles

Logistic regression provides a clear parametric expression for binary probability modeling. With one-hot encoding of categorical variables, each category can have its own coefficient independent of other categories, so it is wrong to simply say logistic regression can only describe a "linear age effect." This report's grouped age and BMI already allow differences between groups; its main limitation is that a main-effects model does not automatically learn complex interactions. Regularization can constrain estimation, but a penalty coefficient in a prediction model should not directly substitute for interpreting a design-weighted association model.

Random forests combine many randomized decision trees, can represent interactions and piecewise relationships between categories, and averaging across trees helps reduce the instability of a single tree (Breiman, 2001). Gradient boosting adds base learners sequentially to optimize a chosen loss function, with complexity governed by learning rate, tree depth, and iteration count (Friedman, 2001). Both offer more flexible functional forms than main-effects logistic regression, but flexibility also raises the demands of tuning and overfitting control.

Christodoulou et al. (2019), comparing machine learning with logistic regression in the clinical prediction literature, found no stable overall advantage for machine learning in low-risk-of-bias comparisons, and noted problems such as insufficient calibration evaluation. That conclusion does not prove the two approaches are equivalent on every task, but it is enough to reject the assumption that "a more complex algorithm is necessarily more accurate." This report therefore treats logistic regression as a formal candidate model, not a weak baseline included only to make tree models look good.

With this report's seven grouped features, the gain from complex models could go either way. If the main information is already expressed by the category differences of age and BMI, the extra interactions available to tree models may be limited; if stable interactions exist across age, activity, and sociodemographic groups, tree models may improve prediction. Which case holds requires empirical comparison. Fair comparison requires the three models to use the same train–test split, the same available features, and the same metrics, with the search ranges disclosed; if tuning budgets differ, conclusions should be limited to the comparison under the stated configurations.

### 2.5 Evaluating Performance under Class Imbalance

When positives are only part of the sample, accuracy can hide identification problems — for example, predicting everything as negative can still yield high accuracy with zero recall. Saito and Rehmsmeier (2015) show in imbalanced classification settings that precision–recall curves highlight how many of the samples identified as positive are truly positive, and how many of all positives are retrieved. This report therefore uses an area-type PR-curve metric as the main comparison criterion, while also reporting other metrics.

Precision is TP divided by TP plus FP; recall is TP divided by TP plus FN. Both depend on the classification threshold, and the PR curve describes the trade-off across thresholds; the area under the curve is not a count of "people retrieved" at any single threshold. ROC-AUC summarizes the relationship between the true positive rate and the false positive rate; under the common definition it can be understood as the probability that a randomly drawn positive case receives a higher predicted score than a randomly drawn negative case, with ties handled accordingly.

PR metrics must be interpreted together with the positive rate. The reference precision for random ranking is approximately the positive rate of the evaluation sample, and the PR baseline changes when the positive rate changes across groups or after weighting. A higher PR metric in a subgroup therefore does not by itself prove better discrimination in that group. This report presents positive rates, ROC-AUC, and threshold metrics together in comparisons, and does not rank fairness directly by cross-group PR values.

This report uses average precision (AP) as the area-type PR-curve metric, computed with scikit-learn's `average_precision_score`. AP weights precision by increments in recall and is not the same as trapezoidal integration of the PR curve. Models, baselines, and the weighted sensitivity analysis use the same calculation. AP labels in figures are all interpreted by the AP definition.

### 2.6 Probability Calibration and Applied Value

A model's ranking of samples and the accuracy of its probabilities are different problems. Van Calster et al. (2019) emphasize that calibration of predicted probabilities is often neglected, while miscalibrated probabilities affect real decisions. A model can rank positives ahead of negatives yet systematically over- or under-estimate event probabilities. This report therefore examines the Brier score and calibration curves alongside ROC-AUC and AP, rather than declaring a model reliable on discrimination alone.

The Brier score is the mean squared error between predicted probabilities and the binary outcome; lower usually means smaller overall probability error, but it combines several aspects of performance and cannot substitute for a calibration curve. Grouped calibration plots can show the relationship between each group's mean predicted probability and its actual positive rate, but may hide within-group bias; a curve close to the diagonal does not prove that all groups or all probability intervals are adequately calibrated. This report therefore limits the strength of its calibration conclusions and discusses calibration separately from discrimination.

Threshold choice also depends on the use case. Maximizing F1 gives a uniform rule for model comparison, but it encodes neither the real costs of false negatives and false positives, nor follow-up testing resources or patient preferences. This report's F1-optimal thresholds serve only research comparison. If the model were later used for referral or screening, a new target definition, external validation, and evaluation of decision consequences would be needed; the current thresholds cannot be carried over directly.

### 2.7 Prediction, Explanation, and Complex Sampling

Shmueli (2010) notes that explaining and predicting have different research goals, and variable selection and model evaluation should change accordingly. This report centers on out-of-sample prediction while also providing conditional association analysis to help understand the data. The size of an odds ratio cannot decide which model predicts best, and better predictive performance cannot prove a variable has a causal effect. Keeping the two apart avoids substituting one significant coefficient for a complete performance check.

The "Table 2 fallacy" discussed by Westreich and Greenland (2013) further warns that coefficients of covariates in a multivariable model cannot all be interpreted as the same kind of causal effect. This report discusses adjusted associations in detail only for BMI, age, and leisure-time physical activity, with the other variables as joint adjustment terms. Even so, uncontrolled confounding, behavior change after diagnosis, and measurement error may still affect the results, so adjusted odds ratios are stated only as conditional odds ratios — not as risk multiples or intervention effects.

Permutation importance answers a different question: how much does predictive performance drop when a feature is shuffled in a fixed model? Its value is affected by feature correlation, the evaluation metric, and the permutation method; correlated features can substitute for each other, and shuffling can create combinations rarely seen in reality. This report calls it model dependence and cross-references it with association direction, without treating agreement between the two analyses as confirmation of a causal relationship.

Complex sampling requires stating the inferential target clearly. The CDC's 2024 technical documentation requires core-variable analyses to use the corresponding final weights, stratification, and PSU information (CDC, 2025c). For prediction research, whether to weight should also be tied to the intended evaluation distribution: unweighted results mainly represent the observed sample, while design-weighted results are closer to the eligible population the survey weights represent. This report uses unweighted main analysis and tests through weighted analysis whether the conclusions change with this target; "predicting individuals" is not treated as sufficient reason not to weight.

### 2.8 Synthesis and Positioning

Existing research provides three concrete starting points. Simple assessment tools support studying routine survey features, but different outcomes prevent direct transfer of scores and performance numbers; model-comparison research supports keeping both simple and flexible methods, but does not support assuming complex algorithms win; measurement and evaluation research requires stating the self-reported label, sampling target, probability calibration, and uncertainty together. This report implements these requirements within a single BRFSS analysis framework.

The research space addressed here is to test three common classification methods on the same seven features in the 2024 public survey data, and to show how much information beyond age and BMI they add. Its contribution is an application comparison with clear boundaries — not a claim to fill a gap where nobody has studied diabetes prediction before. If model differences are small, the conclusion should return to the information limits of the features and the label; if differences exist, they must pass robustness analysis and external validation before broader application is discussed.

**Table 1 — How existing findings map to this report's design**

| Finding from existing research | Source | What this report does |
| --- | --- | --- |
| Overweight/obesity, increasing age, and insufficient physical activity are associated with diabetes | ADA (2025); CDC | Takes these three as the main features |
| BRFSS is self-reported data: broadly usable, but with self-report bias and telephone-survey coverage issues | Pierannunzi et al. (2013) | Calls the outcome "self-reported diagnosed" and discusses bias in the limitations |
| Prediction and explanation have different goals; observational data cannot be treated directly as causal; not every coefficient in a multivariable model is interpretable | Shmueli (2010); Hernán (2018); Westreich & Greenland (2013) | Prediction-led; interprets only the three main features; feature importance is called "model dependence" only |
| Under class imbalance, PR curves help evaluate the precision–recall trade-off and depend on the positive rate | Davis & Goadrich (2006); Saito & Rehmsmeier (2015) | AP is the main metric, reported together with the prevalence baseline |
| Random forests and gradient boosting capture nonlinearity and interactions and perform robustly on tabular data | Breiman (2001); Friedman (2001); Grinsztajn et al. (2022) | Compared with logistic regression; features are few categories, so the gap may be small and needs testing |
| BRFSS is a complex sample: population estimates need weights, variance needs strata | Lumley (2010); Heeringa et al. (2017) | Description and aOR use design weighting; main prediction analysis is unweighted, weighted analysis is the sensitivity check |

## 3. Data and Methods

### 3.1 Data Source and Study Design

This study uses a cross-sectional observational design on the CDC's 2024 BRFSS combined landline and cell-phone public data. The official annual page provides SAS Transport and ASCII files, a codebook, calculated-variable documentation, a data-quality report, and complex-sampling technical files. The data file specified for this study is LLCP2024XPT.zip, released in August 2025; the annual page lists 457,670 records and 345 variables (CDC, 2025a). The 2024 questionnaire and calculated-variable version 4 documentation were also consulted to interpret variable meanings (CDC, 2024; CDC, 2025b).

The study scope is limited to eligible adult respondents in the areas covered by the public file. Tennessee was not included; Guam, Puerto Rico, and the U.S. Virgin Islands are included in the combined file, so population descriptions cannot unconditionally be called complete U.S. national estimates. The CDC also notes that the public data were modified and that some missingness may relate to removed questions. This report treats the data version and outlier checks as part of analysis provenance and does not assume all missingness is random.

The analysis is based on 12 columns extracted from the BRFSS public file — the outcome, the seven prediction features, and four design and auxiliary variables — covering 457,670 records. The extracted file's SHA-256, software versions, and run order are listed in Appendix A. The extracted file's hash locks the analysis input; it does not substitute for the hash of the complete original XPT file.

### 3.2 Sample Inclusion and Outcome Definition

The target variable comes from DIABETE4. Respondents answering 1 ("Yes") are coded positive (1); those answering 3 ("No") are coded negative (0). Those diagnosed only during pregnancy, with prediabetes or borderline status, "don't know," refusals, and missing records are excluded from the main analysis. A total of 15,736 records were excluded, leaving an analysis sample of 441,934, of whom 65,809 report a diagnosis — an unweighted positive rate of about 14.89%. This rate describes the eligible analysis sample and does not substitute for the design-weighted population rate.

Excluding prediabetes has a clear definitional consequence: the negative group in the main analysis does not contain all non-diabetic respondents, only those who explicitly answered "No." The prediabetes group may be closer in feature distribution to diagnosed cases, and excluding them may change classification difficulty. This report therefore includes them in the negative group as a sensitivity analysis (S3), to judge whether results depend on the composition of the control group. That analysis still uses the questionnaire classification and does not treat prediabetes status as a laboratory-confirmed result.

The sample is randomly split 70% training and 30% test, stratified by outcome — 309,353 and 132,581 people respectively, with random seed 20260920. The training set carries preprocessing fitting, hyperparameter selection, and threshold selection; the test set is used for final evaluation only after those steps are complete. Stratification by outcome helps keep the positive rate similar in both parts but does not automatically guarantee balance on geography, survey design, or all covariates.

### 3.3 Prediction Features and Encoding Rules

All prediction models use the same seven features. BMI category uses _BMI5CAT: underweight, normal, overweight, and obese, with normal as the reference category in the association model. Age uses _AGEG5YR: original codes 1–5 are merged into 18–44 years, 6–9 into 45–64 years, and 10–13 into 65 years and older; code 14 and missing values form an "unknown" category, with 18–44 as the reference. Merging categories improves brevity of expression but loses continuous differences within age and BMI groups.

The model age variable is named age3, with categories 18–44, 45–64, and 65 and older, plus an unknown category. The three prediction models, baselines B1 and B2, and the association model use the same mapping, and the encoder's actual categories are the four listed. The original 13 levels of _AGEG5YR are used only for codebook comparison or fine-grained description, not as additional model inputs; the unknown category has no age ordering.

Leisure-time physical activity uses EXERANY2: any vs. no physical activity or exercise outside work in the past month, with "any" as the reference. It does not indicate whether any exercise guideline is met, nor does it include complete occupational activity. Sex uses _SEX (male, female), with male as the reference; the variable is used as measured in the data file and does not represent a complete measurement of gender identity.

Race or ethnicity uses _RACE (eight valid categories), with White non-Hispanic as the reference; education uses _EDUCAG (four valid groups), with college graduates as the reference; household income uses _INCOMG1 (seven valid groups), with $50,000 to under $100,000 as the reference. The income bands are under $15,000; $15,000 to under $25,000; $25,000 to under $35,000; $35,000 to under $50,000; $50,000 to under $100,000; $100,000 to under $200,000; and $200,000 or more. This is household income, not equivalent to personal wages or income adjusted for household size.

"Don't know," refusals, and missing values in the features are grouped into "unknown" by each variable's official special codes and are not used as continuous magnitudes. The unknown category retains records and reflects response patterns, but it does not recover the missing true values; if the missingness mechanism changes in new data, the model may also become inaccurate. A sensitivity analysis (S1) drops records with any unknown prediction feature and compares with the main analysis that keeps the unknown categories.

Categorical features are one-hot encoded, with fitting done only on training data or within the training folds of cross-validation. Category mappings pre-specified in the official codebook may be applied to all records; any processing parameters learned from data must not use the test set. _LLCPWT, _STSTR, _PSU, and _STATE are used only for survey design, evaluation, and grouped validation — not as prediction features.

**Table 2 — Concepts, variables, encoding, and use**

| Concept | BRFSS variable | Encoding | Use |
| --- | --- | --- | --- |
| Self-reported diagnosed diabetes (outcome) | DIABETE4 | 1 = Yes → 1; 3 = No → 0; 2 (pregnancy only), 4 (prediabetes/borderline), 7, 9, missing → excluded | Prediction target |
| BMI category | _BMI5CAT | 1 underweight, 2 normal (reference), 3 overweight, 4 obese; missing → "unknown" | Prediction feature; interpretation (main) |
| Age structure | _AGEG5YR | Original 1–5 → 18–44; 6–9 → 45–64; 10–13 → 65+; 14 and missing → unknown | Prediction feature; interpretation (main) |
| Any leisure-time physical activity | EXERANY2 | 1 = yes (reference), 2 = no; 7, 9, missing → "unknown" | Prediction feature; interpretation (main) |
| Sex | _SEX | 1 male (reference), 2 female | Prediction feature; adjustment variable only |
| Race or ethnicity | _RACE | Categories 1–8 (White non-Hispanic as reference); 9, missing → "unknown" | Prediction feature; adjustment variable only |
| Education | _EDUCAG | Categories 1–4 (college graduate as reference); 9, missing → "unknown" | Prediction feature; adjustment variable only |
| Household income | _INCOMG1 | Categories 1–7 ($50k–<100k as reference); 9, missing → "unknown" | Prediction feature; adjustment variable only |
| Survey design (not features) | _LLCPWT, _STSTR, _PSU, _STATE | Not entered into models; used for weighting, variance, and grouped validation | Design and robustness |

### 3.4 Complex Sampling and Inferential Targets

Population description and conditional association inference for the BRFSS core data use the final weight _LLCPWT, the stratum _STSTR, and the PSU variable _PSU (CDC, 2025c). Weights determine each record's contribution; strata and PSUs are used for variance. Passing `sample_weight` to a prediction model does not by itself complete complex-survey inference. The main prediction analysis is unweighted and addresses the observed sample; S2 examines the evaluation target implied by the design weights. Implementation and computation checks for each step are in Table 3.

In the complete 12-column extracted data, H = (_STATE, _STSTR) and G = (H, _PSU). The 457,670 records correspond to 457,670 distinct G values, exactly one record per G; the original _PSU has only 43,913 distinct values and cannot serve as a global identifier without the design stratum. _STSTR is not reused across states. The complete design has 2,393 strata, of which 110 have a single PSU; the eligible analysis domain contains 2,390 strata, of which 112 retain a single PSU — 107 from originally single-PSU strata and 5 from originally multi-PSU strata reduced to one PSU within the domain; another 3 originally single-PSU strata have no records in the domain. Design variables have no missing values and all weights are positive.

Description and association analyses first build the design on all 457,670 records, then take the analysis domain with DIABETE4 equal to 1 or 3. R uses `survey.lonely.psu = "adjust"` and `survey.adjust.domain.lonely = TRUE`, preserving the difference between the original design and the domain's lonely-PSU handling and all warnings. To avoid huge identifier cross-tabulations, the audited-unique G is mapped one-to-one to the original row number; `svydesign` uses `nest = FALSE` and `check.strata = FALSE`, with nesting verified by an independent data audit — this numeric relabeling does not change any PSU or the design.

With the fixed random seed 20260920, the 70/30 split stratified by outcome gives 309,353 in training (46,066 positive) and 132,581 in test (19,743 positive). The intersection of the composite PSU key G between training and test is 0; in five-fold StratifiedKFold, the training–validation G intersection within each fold is also 0. Since each G has exactly one record, record-level splitting never splits a PSU. The state-grouped S4 examines performance in unseen regions and has a different goal from the main analysis's internal validation.

S2 refits with fixed main-analysis parameters, rescaling training weights by the training-set mean; each out-of-fold fit uses the current training fold's weight mean. S2's threshold is set by the maximum weighted out-of-fold F1, and test metrics use the original survey weights. This analysis changes fitting and evaluation weights without additional weighted tuning. Test weights are used for normalized metrics on the fixed holdout sample and are not interpreted as full-population counts.

**Table 3 — Complex-sampling implementation**

| Step | Units and grouping keys | Weight use | Software implementation | Computed checks |
| --- | --- | --- | --- | --- |
| Design and description | All 457,670 records; stratum H, composite PSU G; design built first, domain taken after | Original _LLCPWT; Taylor design variance | R survey 4.2.1; G mapped to original row number; nest = FALSE / check.strata = FALSE, nesting separately audited | 2,393 strata; 110 originally single-PSU strata; 112 single-PSU strata in domain; weights complete and all positive |
| Train–test | 70/30 by outcome; one record per G | Main analysis unweighted | train_test_split, seed 20260920 | 309,353 / 132,581 people; G intersection 0; original row-number list |
| Five-fold tuning | StratifiedKFold within training; G does not cross sets | Main-analysis AP unweighted | Pipeline, dense one-hot; five folds; LR 4 / RF 5 / GB 8 candidates | Per-candidate scores, best parameters; all five G intersections 0 |
| S2 and thresholds | Refits with fixed main-analysis parameters; five-fold out-of-fold F1 thresholds | Each fold weighted by that fold's mean; S2 threshold and test by original weights | sample_weight passed explicitly to each out-of-fold fit; no weighted re-tuning | Thresholds, confusion matrices, weighted AP/ROC/Brier; calibration tables by group |
| Main-analysis intervals | Fixed test records, 1,000 outcome-stratified resamples | Unweighted; sample-conditional intervals | Tied scores combined before AP; shared resampling indices | Percentile intervals for each AP and model/baseline differences; no training variation |
| S2 design intervals | Complete H/G design; non-test contributions zero | Weight derivatives of weighted AP; Taylor design variance | Python linearization; R svytotal independent covariance cross-check; original single-PSU adjust | Derivatives pass finite-difference checks; covariance cross-check; fixed-score conditional intervals |
| Conditional associations | Complete design, then eligible domain; shared seven categorical features | Original weights; quasibinomial with design intervals | R svyglm; Python independent coefficient check | aOR matches rounded study values; coefficient difference < 1e-7; log contains single-PSU warnings |
| S4 exploratory validation | State-grouped, unseen-region performance | Unweighted AP convention | Grouped validation; complete nested tuning not confirmed | Interpreted as exploratory evidence, not clinical or cross-year external validation |

### 3.5 Model Training and Hyperparameter Selection

The analysis uses scikit-learn 1.8.0, with three models: LogisticRegression, RandomForestClassifier, and HistGradientBoostingClassifier. One-hot encoding explicitly outputs dense matrices to suit histogram gradient boosting. Five-fold tuning within the training set uses average precision; all preprocessing is fit within training folds, and the test set plays no part in parameter or threshold selection.

The hyperparameter search ranges are: logistic regression C ∈ {0.01, 0.1, 1, 10}; random forest with 300 trees, max_depth ∈ {8, 12, None}, min_samples_leaf ∈ {5, 20, 50}, five candidates drawn with a fixed seed; gradient boosting with learning_rate ∈ {0.03, 0.1}, max_depth ∈ {3, 5}, max_iter ∈ {200, 400}, eight candidates. Five-fold scores for each candidate and the complete best parameters are saved.

The selected configurations are: logistic regression C = 0.01; random forest min_samples_leaf = 50, max_depth = None, n_estimators = 300; gradient boosting learning_rate = 0.03, max_depth = 5, max_iter = 200. The three models' search budgets differ, so algorithm comparison is limited to the stated search conditions and does not claim exhaustive optimization of every model.

### 3.6 Baselines and Performance Metrics

Three baselines are set. B0 is the random-ranking reference, whose PR baseline precision is approximately the evaluation sample's positive rate and whose ROC-AUC is 0.5; the constant-probability baseline for probability error should give the test set the training-set positive rate as a fixed probability. B1 is a logistic regression using age only, and B2 uses age and BMI category. B1 and B2 use the same train–test split to help judge the gain from adding features.

Evaluation addresses three distinct decisions. First, ranking ability: AP is used for hyperparameter selection, ROC-AUC as a supporting description; neither specifies a threshold. AP is computed by `average_precision_score` and is affected by the evaluation sample's positive rate, so AP values cannot be used directly to judge model quality across groups or weighting targets. Second, threshold classification: after the configuration is chosen, five-fold out-of-fold probabilities are generated on the training set, the threshold maximizing F1 is selected (the smallest threshold at the maximum), locked, and applied to the test set. Per-model thresholds with TP, FP, TN, FN are in Table 5. S2 fits each fold explicitly weighted and selects the threshold by weighted out-of-fold F1. F1 does not express the real costs of false positives and false negatives. The out-of-fold process happens within training after hyperparameter selection and cannot serve as a nested, unbiased evaluation of the full pipeline; the final test set is independent of parameter and threshold selection. Third, probability error: the Brier score is the mean squared difference between predicted probabilities and the binary outcome, with the design-weighted version Σw(p−y)²/Σw. It is affected by discrimination, calibration, and the outcome distribution together and cannot alone prove reliable calibration. Ten-group calibration plots should report the grouping rule, group sizes, and positive counts together; a figure alone cannot prove full calibration across all groups. If a calibrator were added, it would have to be fit within training and locked before testing.

Thresholds for future applications should be chosen in development data based on resource capacity, acceptable misses, or explicit costs, then externally validated. This report has no decision net-benefit or cost-utility analysis, so screening usability cannot be inferred from a better AP, F1, or Brier score.

### 3.7 Uncertainty and Model Comparison

Uncertainty is quantified with 1,000 outcome-stratified bootstrap resamples of the test records, positives and negatives resampled separately with replacement, with all models and baselines sharing the same resampling counts and tied scores combined correctly. Since each composite PSU has only one record, record resampling never splits a PSU. But the intervals remain sample-conditional intervals for fixed models, thresholds, and an approximately fixed positive composition — not survey-design intervals — and they do not include training or tuning variation.

S2 uses Taylor linearization of the weight derivatives of weighted AP at fixed predicted scores: tied scores are combined before differentiation, linearized contributions are placed back into the complete H/G design, and domain-outside and non-test records contribute zero. Originally single-PSU strata use adjust. Derivatives pass finite-difference checks with tied scores and a weight scale-invariance test; covariance is cross-checked independently with R survey::svytotal. t-approximate intervals use the complete design's 455,277 degrees of freedom; all models share the same design, and paired differences use the covariance term. These intervals are conditional on fixed models and the given holdout labels and do not include uncertainty from training, tuning, threshold selection, or holdout randomization, so they are not interpreted as full-pipeline or clinical external uncertainty.

H1 is judged by the paired gain relative to B2, and H2 by absolute model differences and paired intervals — not by whether two separate intervals overlap. Intervals without multiple-comparison correction are nominal 95% intervals only. Statistical detectability, substantive importance, and decision benefit are judged separately; no substantive-importance threshold was pre-specified, so excluding zero cannot be used to declare that a complex model has applied advantage.

### 3.8 Conditional Associations and Model Dependence

The supplementary association analysis uses design-weighted logistic regression with all seven features entered together, reporting adjusted odds ratios (aOR) and 95% intervals for BMI category, age group, and leisure-time physical activity. Its form is logit{P(Y=1|X)} = β₀ + Σβ, and the aOR is the exponential of the corresponding coefficient. This parameter is the odds ratio for diagnosis history given the model, adjusting for the other features — not a ratio of disease probabilities; the difference matters especially because positives are not rare.

The association model and the regularized logistic regression used for prediction serve different tasks; their coefficients and confidence intervals should not be mixed. Interpretation of the key features always keeps the conditional-association wording, and the coefficients of the remaining adjustment terms are not all interpreted as independent causal effects. H3 requires checking direction, intervals, and permutation results separately; unknown categories carry no ordered-trend meaning.

The association model uses R 4.3.3 and survey 4.2.1, building the complete survey design first, then taking the eligible analysis domain, and fitting with `svyglm(..., family = quasibinomial())`. aOR intervals use design variance; a Python statsmodels fit independently checks the coefficients, with a maximum absolute difference of 9.81×10⁻⁸. The complete design and domain single-PSU handling, reference categories, design degrees of freedom, run warnings, and software environment are all saved in the computation materials.

Permutation importance is computed on the locked test set, each original feature shuffled 20 times, measuring the drop in AP. One-hot-encoded columns belonging to the same feature are treated as a group to avoid creating invalid category combinations. Permutation results are used only to interpret the completed model comparison and must not be used to retroactively change features, thresholds, or model selection, or the test set would no longer serve its final-evaluation purpose. Positive importance means the model depends on that feature under this evaluation distribution; it does not mean changing that feature would change disease status.

### 3.9 Subgroup and Robustness Analyses

Subgroup evaluation reports test sample size, positive count, positive rate, and model performance by demographic group, using the thresholds set in the overall training stage for all groups. Threshold performance differences across groups are used to observe false-positive and false-negative differences and do not constitute a completed fairness evaluation. Subgroups with fewer than 100 positives are described only, avoiding strong conclusions about unstable differences; even with larger samples, AP alone cannot judge whether one group receives better service.

Five sensitivity analyses each change one main condition. S1 drops records with any unknown feature, testing the handling of missing categories; S2 retrains with survey weights and evaluates weighted, testing the evaluation distribution; S3 merges self-reported prediabetes into the negative group, testing the outcome-exclusion rule; S4 uses state- or region-grouped cross-validation, testing generalization to regions not in training; S5 excludes the three territories, testing geographic coverage.

S4 groups by state or reporting region so that the same region never enters both a training and a validation fold. Its results explore regional differences; because the complete nested tuning process was not confirmed, they are not treated as selection-bias-free external validation. State groups still come from the same survey year and are not equivalent to cross-year or clinical validation; S5 changes geographic coverage and cannot restore Tennessee.

Robustness is judged by whether conclusions change markedly with definition, weights, and geography — including model ranking, gains relative to baselines, calibration, and the direction of key associations. Similar conclusions only show insensitivity to the tested choices; they do not exclude label error or unmeasured factors. All sensitivity analyses remain analyses of self-reported cross-sectional survey data and cannot be upgraded to clinical diagnosis or causal inference.

### 3.10 Research Ethics and Reporting Transparency

This report uses public survey data, with analyses mainly at aggregate and group levels. It does not attempt to re-identify individuals and does not output personal diagnostic advice. Race or ethnicity, sex, income, and similar variables are used only as measured in the survey; interpretation of results must consider measurement and social conditions, and group differences in models must not be used to infer essential group judgments.

Computation materials include the data hash, design audit, row-level splits and fold numbers, parameter-search results, encoder categories, out-of-fold and test probabilities, thresholds and confusion matrices, AP intervals, R association outputs, and the software environment. The study uses public survey data, and results are interpreted only under the given labels, features, and evaluation distribution. The byte identity of the inputs is locked by the extracted file's hash; this does not claim the complete CDC file or future updated versions have identical byte content.

## 4. Results

### 4.1 Sample Characteristics

The analysis sample is 441,934 people, of whom 65,809 report a diagnosis. The unweighted prevalence is 14.89%; design-weighted it is 12.98% (95% CI 12.76%–13.20%), lower than the unweighted figure, indicating that diagnosed people are slightly overrepresented in the sample. Compared with those without a diagnosis, diagnosed respondents are older (65 years and older: 44.5% vs. 20.0%), more often obese (46.7% vs. 26.5%), and less often physically active in leisure time (63.3% report activity vs. 79.5%). These percentages are design-weighted estimates.

### 4.2 RQ1: How Well Do the Models Identify the Label?

The test set contains 132,581 people, with 19,743 positives. The three models' AP values are 0.332887–0.337267, a gain of 0.058218–0.062598 over the age-plus-BMI-category baseline B2 (Table 6). With the training out-of-fold thresholds locked, recall is 0.572–0.623 and precision is 0.303–0.314; per-model thresholds and confusion matrices are in Table 5. Brier scores are 0.112–0.113, below the constant baseline of 0.127. The calibration plot provides grouped probability-consistency information; Brier scores or grouped plots alone cannot be used to claim full calibration or actual screening value.

**Table 4 — Overall test-set performance (unweighted, main analysis)**

| Model | AP (95% CI) | ROC-AUC | Recall | Precision | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Prevalence baseline (random) | 0.149 | 0.500 | — | — | — | 0.127 |
| Baseline 1: age only | 0.211 (0.209–0.213) | 0.664 | 0.908 | 0.199 | 0.327 | 0.121 |
| Baseline 2: age + BMI category | 0.275 (0.271–0.278) | 0.728 | 0.665 | 0.265 | 0.379 | 0.116 |
| Logistic regression | 0.333 (0.327–0.338) | 0.761 | 0.572 | 0.314 | 0.405 | 0.113 |
| Random forest | 0.336 (0.330–0.341) | 0.763 | 0.584 | 0.312 | 0.407 | 0.112 |
| Gradient boosting | 0.337 (0.332–0.343) | 0.764 | 0.623 | 0.303 | 0.408 | 0.112 |

Best parameters: logistic regression C = 0.01; random forest min_samples_leaf = 50, unlimited depth; gradient boosting learning rate 0.03, depth 5, 200 iterations.

**Table 5 — Classification thresholds and test confusion matrices**

| Model | Threshold | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- |
| LR | 0.20187605 | 11,293 | 24,682 | 8,450 | 88,156 |
| RF | 0.21264887 | 11,529 | 25,394 | 8,214 | 87,444 |
| GB | 0.19891804 | 12,300 | 28,305 | 7,443 | 84,533 |

The test-set precision–recall curves for the three models almost coincide and are all clearly above the baselines. The calibration curve, grouped by predicted probability into 10 groups, shows each group's mean predicted probability against its actual positive rate, with per-group deviations of at most 0.016.

### 4.3 RQ2: Differences Between Models and Across Groups

In the unweighted main analysis, random forest exceeds logistic regression by an absolute AP difference of 0.002673 (95% interval 0.000693–0.004673), gradient boosting exceeds logistic regression by 0.004380 (0.002800–0.005962), and gradient boosting exceeds random forest by 0.001707 (0.000277–0.003033). The largest difference is about 0.44 percentage points on the AP scale — not an accuracy gain. The nominal intervals exclude 0, suggesting small differences are detectable, but no multiple-comparison correction was applied and no substantive-importance threshold was pre-specified. In the weighted evaluation, the intervals for the differences all include 0 (Table 10), which does not support claiming any algorithm is stably better under a different evaluation distribution. Gains relative to B2 cannot separate the causal contributions of individual features.

**Table 6 — Paired differences in AP**

| Comparison | AP difference | 95% CI (paired bootstrap) | Interval includes 0? |
| --- | --- | --- | --- |
| RF − LR | +0.002673 | +0.000693 to +0.004673 | No |
| GB − LR | +0.004380 | +0.002800 to +0.005962 | No |
| GB − RF | +0.001707 | +0.000277 to +0.003033 | No |
| LR − B2 | +0.058218 | +0.054456 to +0.062021 | No |
| RF − B2 | +0.060891 | +0.057081 to +0.065031 | No |
| GB − B2 | +0.062598 | +0.058539 to +0.066858 | No |

Subgroup performance is in Table 7 (full results in Appendix B). Because AP rises with a subgroup's prevalence, ROC-AUC is the main basis for comparing groups: discrimination is lower at age 65 and older (about 0.69) and among Black respondents (about 0.74), and higher among Asian respondents (about 0.81). Also, with one shared threshold for everyone, recall varies sharply with subgroup prevalence: at ages 18–44 almost no diagnosed cases are found (recall 0.00–0.01), while recall is about 0.74–0.78 for household income under $15k and only about 0.17–0.22 for income of $200k or more.

**Table 7 — Performance in selected subgroups (test set; three values are logistic regression / random forest / gradient boosting)**

| Subgroup | Test positives | Subgroup prevalence | ROC-AUC (LR / RF / GB) | Recall (LR / RF / GB) |
| --- | --- | --- | --- | --- |
| Age: 18–44 | 1,514 | 3.8% | 0.704 / 0.713 / 0.721 | 0.000 / 0.000 / 0.009 |
| Age: 45–64 | 6,658 | 16.6% | 0.707 / 0.708 / 0.710 | 0.525 / 0.538 / 0.578 |
| Age: 65+ | 11,271 | 22.6% | 0.683 / 0.686 / 0.687 | 0.685 / 0.699 / 0.740 |
| Household income: <$15k | 1,340 | 22.8% | 0.743 / 0.747 / 0.749 | 0.784 / 0.740 / 0.782 |
| Household income: ≥$200k | 592 | 6.7% | 0.753 / 0.758 / 0.762 | 0.191 / 0.171 / 0.215 |
| Race/ethnicity: White (non-Hispanic) | 13,627 | 14.2% | 0.752 / 0.754 / 0.754 | 0.527 / 0.534 / 0.576 |
| Race/ethnicity: Black (non-Hispanic) | 2,162 | 21.4% | 0.738 / 0.743 / 0.745 | 0.758 / 0.781 / 0.799 |
| Race/ethnicity: Asian | 418 | 11.7% | 0.808 / 0.811 / 0.812 | 0.495 / 0.445 / 0.565 |
| Race/ethnicity: Hispanic | 1,963 | 14.3% | 0.794 / 0.801 / 0.802 | 0.635 / 0.680 / 0.706 |
| Sex: Male | 9,755 | 15.3% | 0.762 / 0.764 / 0.765 | 0.587 / 0.591 / 0.640 |
| Sex: Female | 9,988 | 14.5% | 0.759 / 0.762 / 0.764 | 0.557 / 0.577 / 0.606 |

### 4.4 RQ3: The Three Main Features

The design-weighted multivariable logistic regression shows: the odds of self-reported diagnosis for people with obesity are 3.51 times those of normal-weight people (95% CI 3.29–3.74); for age 65 and older, 9.95 times those of ages 18–44 (9.28–10.66); and for people with no leisure-time activity, 1.55 times those with activity (1.48–1.62). Unweighted results have the same direction and similar values. Permutation importance shows all three models depend most on age, then BMI category, then leisure-time physical activity (mean across the three models: 0.110, 0.068, 0.024); dependence on sex and education is smallest. The conditional-association results support the direction of H3 but do not indicate causality: cross-sectional data cannot distinguish "obesity first, then disease" from "behavior change after diagnosis," and leisure-time physical activity in particular may reflect reverse causation.

**Table 8 — Adjusted odds ratios and design intervals for the main features**

| Comparison | aOR | Lower | Upper |
| --- | --- | --- | --- |
| Obese vs. normal | 3.50630 | 3.28721 | 3.73999 |
| 65+ vs. 18–44 | 9.94648 | 9.28015 | 10.66066 |
| No activity vs. activity | 1.55036 | 1.48054 | 1.62346 |

In the figures, the BMI variable corresponds to _BMI5CAT and the activity variable to EXERANY2 (any vs. no leisure-time physical activity); neither represents continuous body weight or activity intensity.

### 4.5 Robustness Checks

The five sensitivity analyses show similar overall identification levels, but the algorithm ranking is not fixed (Table 9). In S2, logistic regression's AP is 0.352, above random forest's 0.350, with gradient boosting at 0.354. In S4, random forest and gradient boosting are both 0.338 at the reported precision, with state-grouped fold ranges of 0.319–0.356. S3's values are slightly lower but still exceed the corresponding prevalence reference; this cannot be used to quantify the full bias from excluding the prediabetes group. Robustness means only that the overall findings are similar under the tested choices; it does not mean every algorithm difference is verified by a design interval.

**Table 9 — Sensitivity analyses**

| ID | What changes | Sample size (positive rate) | AP (LR / RF / GB) | Conclusion changes? |
| --- | --- | --- | --- | --- |
| Main analysis | Unweighted; 70/30 split | 441,934 (14.89%) | 0.333 / 0.336 / 0.337 | — |
| S1 | Drop records with any "unknown" feature | 332,252 (14.96%) | 0.341 / 0.341 / 0.343 | No: gradient boosting still highest |
| S2 | Retrain weighted + weighted evaluation | Same as main analysis (weighted prevalence 13.05%) | 0.352 / 0.350 / 0.354 | No: gradient boosting still highest; random forest and logistic regression swap order; differences small |
| S3 | Prediabetes merged into the negative group | 453,241 (14.52%) | 0.321 / 0.324 / 0.325 | No: values slightly lower, still well above the prevalence reference |
| S4 | State-grouped 5-fold cross-validation | 441,934 (14.89%) | 0.334 / 0.338 / 0.338 | No: random forest and gradient boosting tied highest |
| S5 | Exclude Guam, Puerto Rico, U.S. Virgin Islands | 435,057 (14.81%) | 0.329 / 0.332 / 0.333 | No |

**Table 10 — Weighted AP and paired differences with design-linearized intervals**

| Model or comparison | Estimate | Lower | Upper |
| --- | --- | --- | --- |
| LR | 0.352299 | 0.336404 | 0.368194 |
| RF | 0.349516 | 0.333117 | 0.365915 |
| GB | 0.354014 | 0.336006 | 0.372023 |
| RF − LR | −0.002783 | −0.009502 | 0.003935 |
| GB − LR | 0.001715 | −0.005851 | 0.009281 |
| GB − RF | 0.004498 | −0.002418 | 0.011415 |

## 5. Discussion

### 5.1 Answers to the Research Questions

| Question | Answer |
| --- | --- |
| RQ1: Can the label be identified? | Yes, but only at a moderate level. AP is about 2.3 times the prevalence baseline and about 0.06 above the age-plus-BMI-category baseline; but precision is only about 30%, so many flagged cases are not diagnosed cases. |
| RQ2: Do the algorithms differ? | In the unweighted internal comparison, small differences are detectable, with a maximum absolute difference of 0.0044 (95% interval 0.0028–0.0060); under design weighting the difference intervals all include 0. There is no evidence yet of a substantive advantage or decision utility for the complex algorithms. |
| RQ3: Which features matter? | Age > BMI category > leisure-time physical activity. The directions match the ADA and CDC descriptions, but these are statistical associations and model dependence — not causality. |

### 5.2 Relation to Existing Research

The direction — obesity, age, and insufficient physical activity associated with diabetes — matches ADA and CDC descriptions (ADA, 2025), and this study provides 2024 BRFSS data support for these known associations. The small advantage of tree models over logistic regression fits a setting where "features are a few categories and information is approximately additive," but this study did not directly test interactions between features. Following Westreich and Greenland's (2013) warning, the aORs for race, education, and income are reported only as adjustment variables and are not interpreted item by item.

### 5.3 Limitations

1. **Cross-sectional design:** temporal order cannot be distinguished; results cannot be used as causal or future-risk predictions.
2. **Self-reported data:** undiagnosed diabetes is classified as negative; self-report bias is discussed in Pierannunzi et al. (2013).
3. **Limited features:** no family history, blood pressure, lipids, diet, smoking, insurance, and others, so the identification ceiling is limited.
4. **Excluding the prediabetes group:** makes the negative group "cleaner"; S3 shows the effect is limited.
5. **Coverage:** Tennessee is not included; the public dataset was modified by administrative order, so missingness patterns may not be fully consistent.
6. **Evaluation and evidence scope:** intervals are conditional on fixed models and the given test split and do not cover all variation from training, tuning, and threshold selection. The main-analysis AP, weighted evaluation, and association model have checkable run outputs; supplementary subgroup, permutation-importance, and S1, S3–S5 results have not completed full independent re-runs and are interpreted as exploratory evidence, without expanding population or causal conclusions. The study did not conduct cross-year external validation, clinical-outcome validation, or decision-utility research.

### 5.4 Next Steps

- Add other core questionnaire variables (insurance, smoking, general health status) and see how much identification improves.
- Use BRFSS from different years for temporal external validation.
- Choose thresholds by subgroup with cost-sensitive analysis, then evaluate whether the model can be used for screening prompts.

## 6. Conclusion and Dissemination Plan

Seven routine survey features from the 2024 BRFSS carry moderate ability to identify self-reported diagnosed diabetes in adults: the three models' AP values are about 0.33–0.34, with precision around 30%. In unweighted evaluation, algorithm differences are statistically detectable but the largest absolute difference is only 0.004380; under weighted evaluation the paired intervals all include 0, so there is no stable evidence of a substantive advantage for complex algorithms. Age, BMI category, and any leisure-time physical activity are the features the models depend on most; adjusted odds ratios express conditional associations only. Results must not be used for diagnosis, causal inference, or future-onset prediction. Future work should expand the features, conduct cross-year validation, and evaluate thresholds against false-positive/false-negative costs and resource capacity.

**Table 11 — Dissemination plan**

| Audience | Format | Focus |
| --- | --- | --- |
| Course and classmates | 12-minute class presentation | Problem → data and design → model comparison → interpretation boundaries → limitations; weighting handled on one slide |
| Academia | Submit to a health-informatics or public-health data-science conference or journal (e.g., AMIA Annual Symposium, Preventing Chronic Disease; exact requirements per call for papers) | Transparent, reproducible workflow; design weights and interval computation checks; algorithm differences, thresholds, and interpretation boundaries |
| Practitioners | One-page summary + code repository | What this kind of model can and cannot do; not for diagnosis |

## References

American Diabetes Association Professional Practice Committee. (2025). Standards of care in diabetes—2025. *Diabetes Care*, 48(Suppl. 1).

Bang, H., Edwards, A. M., Bomback, A. S., Ballantyne, C. M., Brillon, D., Callahan, M. A., Teutsch, S. M., Mushlin, A. I., & Kern, L. M. (2009). Development and validation of a patient self-assessment score for diabetes risk. *Annals of Internal Medicine*, 151(11), 775–783. https://doi.org/10.7326/0003-4819-151-11-200912010-00005

Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Centers for Disease Control and Prevention. (2024). 2024 Behavioral Risk Factor Surveillance System questionnaire. https://www.cdc.gov/brfss/questionnaires/pdf-ques/2024-BRFSS-Questionnaire-508.pdf

Centers for Disease Control and Prevention. (2025a). 2024 BRFSS survey data and documentation. https://www.cdc.gov/brfss/annual_data/annual_2024.html

Centers for Disease Control and Prevention. (2025b). Calculated variables in the 2024 data file of the Behavioral Risk Factor Surveillance System (Version 4). https://www.cdc.gov/brfss/annual_data/2024/pdf/2024-calculated-variables-version4-508.pdf

Centers for Disease Control and Prevention. (2025c). Complex sampling weights and preparing 2024 BRFSS module data for analysis. https://www.cdc.gov/brfss/annual_data/2024/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf

Centers for Disease Control and Prevention. Diabetes risk factors. https://www.cdc.gov/diabetes/risk-factors/index.html

Christodoulou, E., Ma, J., Collins, G. S., Steyerberg, E. W., Verbakel, J. Y., & Van Calster, B. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*, 110, 12–22. https://doi.org/10.1016/j.jclinepi.2019.02.004

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning*, 233–240.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks Track*.

Heeringa, S. G., West, B. T., & Berglund, P. A. (2017). *Applied survey data analysis* (2nd ed.). CRC Press.

Hernán, M. A. (2018). The C-word: Scientific euphemisms do not improve causal inference from observational data. *American Journal of Public Health*, 108(5), 616–619.

Lindström, J., & Tuomilehto, J. (2003). The diabetes risk score: A practical tool to predict type 2 diabetes risk. *Diabetes Care*, 26(3), 725–731. https://doi.org/10.2337/diacare.26.3.725

Lumley, T. (2010). *Complex surveys: A guide to analysis using R*. Wiley.

Noble, D., Mathur, R., Dent, T., Meads, C., & Greenhalgh, T. (2011). Risk models and scores for type 2 diabetes: Systematic review. *BMJ*, 343, d7163. https://doi.org/10.1136/bmj.d7163

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html

Pierannunzi, C., Hu, S. S., & Balluz, L. (2013). A systematic review of publications assessing reliability and validity of the Behavioral Risk Factor Surveillance System (BRFSS), 2004–2011. *BMC Medical Research Methodology*, 13, 49. https://doi.org/10.1186/1471-2288-13-49

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Shmueli, G. (2010). To explain or to predict? *Statistical Science*, 25(3), 289–310. https://doi.org/10.1214/10-STS330

Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine*, 17, 230. https://doi.org/10.1186/s12916-019-1466-7

Westreich, D., & Greenland, S. (2013). The Table 2 fallacy: Presenting and interpreting confounder and modifier coefficients. *American Journal of Epidemiology*, 177(4), 292–298. https://doi.org/10.1093/aje/kws412

R survey documentation. svydesign and Lonely PSUs. https://r-survey.r-forge.r-project.org/survey/html/svydesign.html

scikit-learn documentation. average_precision_score and Brier score loss. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

## Appendix A — Reproducibility Notes

The analysis input is `brfss2024_subset.csv.gz`, with 457,670 rows and 12 columns. Its SHA-256 is `f7d736fcad6f77e4090c6e84e5c72cbf3765d4e00795b37316c2c458421b46d3`.

The computing environment is Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, statsmodels 0.15.0; association and design-variance calculations use R 4.3.3 and survey 4.2.1.

Run order: `audit_data.py` and `validate_audit.py` check the sample and design; `models.py` runs preprocessing, tuning, out-of-fold thresholds, and test evaluation; `survey_check_numeric.R` fits the survey model; `aor_independent.py` checks the coefficients; `evaluate.py` computes the AP intervals; `ap_survey_check.R` and `compare_ap.py` check the design covariance.

The age-mapping file `age_mapping.csv` and three `encoder.json` files record the original age bands and the age3 categories; `split_manifest.csv.gz` records original row numbers, PSU keys, train–test labels, and fold numbers. Parameter searches, predicted probabilities, confusion matrices, and calibration groups are all provided as machine-readable files.

Weighted AP first combines records with identical predicted scores, then differentiates with respect to case weights; linearized contributions are placed in the complete design, with non-test and out-of-domain contributions set to zero. Derivatives are checked by finite differences and scale invariance, and the design covariance's maximum absolute difference from R survey::svytotal is 1.51×10⁻¹⁷.

The intervals target the evaluation uncertainty of fixed fitted models on the given test split. Additional variation from the training process, parameter and threshold selection, and holdout randomization is not included, so the intervals cannot be interpreted as all uncertainty after deployment.

## Appendix B — Complete Subgroup Results

Each cell's three values are logistic regression / random forest / gradient boosting. Subgroups with fewer than 100 positives (education: unknown) are descriptive only and not compared.

| Subgroup | Test n | Positives | AP | ROC-AUC | Recall |
| --- | --- | --- | --- | --- | --- |
| Sex: Male | 63,580 | 9,755 | 0.338 / 0.341 / 0.342 | 0.762 / 0.764 / 0.765 | 0.587 / 0.591 / 0.640 |
| Sex: Female | 69,001 | 9,988 | 0.327 / 0.330 / 0.333 | 0.759 / 0.762 / 0.764 | 0.557 / 0.577 / 0.606 |
| Race/ethnicity: Unknown | 2,576 | 396 | 0.348 / 0.336 / 0.346 | 0.755 / 0.757 / 0.755 | 0.593 / 0.646 / 0.669 |
| Race/ethnicity: White (non-Hispanic) | 95,969 | 13,627 | 0.312 / 0.315 / 0.315 | 0.752 / 0.754 / 0.754 | 0.527 / 0.534 / 0.576 |
| Race/ethnicity: Black (non-Hispanic) | 10,105 | 2,162 | 0.393 / 0.394 / 0.397 | 0.738 / 0.743 / 0.745 | 0.758 / 0.781 / 0.799 |
| Race/ethnicity: American Indian / Alaska Native | 1,889 | 417 | 0.400 / 0.394 / 0.402 | 0.733 / 0.734 / 0.734 | 0.748 / 0.707 / 0.746 |
| Race/ethnicity: Asian | 3,587 | 418 | 0.350 / 0.339 / 0.350 | 0.808 / 0.811 / 0.812 | 0.495 / 0.445 / 0.565 |
| Race/ethnicity: Native Hawaiian / Pacific Islander | 568 | 112 | 0.353 / 0.375 / 0.376 | 0.747 / 0.753 / 0.753 | 0.696 / 0.643 / 0.768 |
| Race/ethnicity: Other single race | 1,095 | 199 | 0.375 / 0.369 / 0.379 | 0.744 / 0.742 / 0.744 | 0.598 / 0.683 / 0.678 |
| Race/ethnicity: Multiple races | 3,033 | 449 | 0.375 / 0.371 / 0.382 | 0.798 / 0.798 / 0.801 | 0.601 / 0.633 / 0.666 |
| Race/ethnicity: Hispanic | 13,759 | 1,963 | 0.355 / 0.375 / 0.371 | 0.794 / 0.801 / 0.802 | 0.635 / 0.680 / 0.706 |
| Age: 18–44 | 40,316 | 1,514 | 0.083 / 0.088 / 0.089 | 0.704 / 0.713 / 0.721 | 0.000 / 0.000 / 0.009 |
| Age: 45–64 | 40,115 | 6,658 | 0.304 / 0.308 / 0.311 | 0.707 / 0.708 / 0.710 | 0.525 / 0.538 / 0.578 |
| Age: 65+ | 49,830 | 11,271 | 0.368 / 0.370 / 0.371 | 0.683 / 0.686 / 0.687 | 0.685 / 0.699 / 0.740 |
| Age: Unknown | 2,320 | 300 | 0.232 / 0.220 / 0.229 | 0.652 / 0.664 / 0.669 | 0.240 / 0.210 / 0.313 |
| Education: Unknown | 685 | 83 | 0.220 / 0.222 / 0.221 | 0.697 / 0.702 / 0.703 | 0.398 / 0.398 / 0.470 |
| Education: Less than high school | 7,658 | 1,727 | 0.410 / 0.419 / 0.420 | 0.746 / 0.755 / 0.754 | 0.786 / 0.814 / 0.836 |
| Education: High school graduate | 33,336 | 5,664 | 0.346 / 0.349 / 0.349 | 0.752 / 0.753 / 0.754 | 0.649 / 0.664 / 0.707 |
| Education: Some college / technical school | 34,953 | 5,907 | 0.344 / 0.347 / 0.347 | 0.744 / 0.746 / 0.747 | 0.616 / 0.620 / 0.674 |
| Education: College / technical school graduate | 55,949 | 6,362 | 0.278 / 0.279 / 0.282 | 0.760 / 0.763 / 0.764 | 0.407 / 0.419 / 0.446 |
| Household income: Unknown | 25,170 | 3,723 | 0.316 / 0.324 / 0.324 | 0.748 / 0.753 / 0.754 | 0.517 / 0.567 / 0.613 |
| Household income: <$15k | 5,890 | 1,340 | 0.415 / 0.420 / 0.422 | 0.743 / 0.747 / 0.749 | 0.784 / 0.740 / 0.782 |
| Household income: $15k–<25k | 9,528 | 2,195 | 0.402 / 0.404 / 0.405 | 0.728 / 0.729 / 0.731 | 0.807 / 0.774 / 0.826 |
| Household income: $25k–<35k | 12,047 | 2,380 | 0.375 / 0.376 / 0.377 | 0.745 / 0.746 / 0.748 | 0.707 / 0.721 / 0.745 |
| Household income: $35k–<50k | 14,297 | 2,482 | 0.335 / 0.339 / 0.340 | 0.737 / 0.741 / 0.742 | 0.671 / 0.685 / 0.704 |
| Household income: $50k–<100k | 32,715 | 4,608 | 0.303 / 0.303 / 0.304 | 0.749 / 0.749 / 0.750 | 0.502 / 0.530 / 0.595 |
| Household income: $100k–<200k | 24,121 | 2,423 | 0.231 / 0.235 / 0.236 | 0.751 / 0.754 / 0.754 | 0.319 / 0.315 / 0.315 |
| Household income: ≥$200k | 8,813 | 592 | 0.183 / 0.184 / 0.192 | 0.753 / 0.758 / 0.762 | 0.191 / 0.171 / 0.215 |

## Appendix C — AI Use Disclosure

Topic conception, writing organization, and code work used AI assistance, as follows. The study used ChatGPT and Codex to assist with literature and writing organization, code writing, computation, and result checking. Model training, metric computation, and R survey analysis all have real run outputs; the team is responsible for research decisions, citation checks, result interpretation, and confirming submission contents per course requirements. AI assistance does not replace the author's responsibility for data authenticity and scientific conclusions.

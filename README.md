# Spotify Genre Prediction

Dataset Used: [Spotify Track DataSet](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).

## Final Project Report
### Main Document

GitHub: https://github.com/rghosh1353/spotify_genre_proj/tree/main
#### Dataset Overview
Dataset Used: Spotify Tracks Dataset
Overview: This project uses a data set of 125 Spotify tracks with audio features such as popularity, danceability, duration, and genres. Thus, the dataset was suitable for the goal of our project, which was to classify songs into their respective genres.

Problem Addressed:
- How can audio and metadata features be used to predict the explicitness of a track?
- Which features are most influential in determining a track’s explicitness?

**Key Methodology**
The **Logistic Regression** algorithm was selected as the key methodology for its simplicity, efficiency, and interpretability, particularly suitable for binary classification tasks with high-dimensional data like the Spotify dataset. Logistic Regression assumes a linear relationship between features and the log-odds of the target variable, which enables straightforward implementation and analysis.

**Note on Our Goal**
We originally aimed to predict the track genre, but our models were unable to predict the genre even with proper training. Instead, we pivoted to use logistics regression, which requires a binary response variable, to predict whether or not a track was explicit depending on the other features.

#### Steps Involved:
**Exploratory Data Analysis**
- Data Loading: Loaded the Spotify dataset and inspected its structure for completeness and quality
- Duplicate Detection: Removed duplicate rows to ensure unique observations.
- Statistical Summary: Assessed mean, median, and standard deviation for numerical features to understand data distribution.

**Visualizations**
- Histograms to observe feature distributions
- Correlation matrices to identify multicollinearity
- Boxplots and Scatterplots for outlier detection
  
**Preprocessing:**
- Removed irrelevant columns such as track_id, artists, and track_name.
- Handled missing data by dropping incomplete rows, ensuring the dataset was clean and reliable.
- Encoded categorical variables (e.g., explicit, track_genre) into numeric representations.
  
**Model Building:**
- Implemented Logistic Regression using scikit-learn.
- Hyperparameters, including:
  - solver='liblinear'
  - class_weight='balanced'
- Applied cross-validation to ensure the model's robustness and generalizability.

**Feature Importance:**
- Interpreted coefficients to understand the contribution of each feature to genre prediction.

**Evaluation:**
- Evaluated the model using metrics like accuracy, precision, recall, and F1-score.
- Achieved an overall accuracy of ~74%, with clear trends in feature relevance and classification performance.
- Performed cross-validation for model performance.

#### Why Logistic Regression Worked:
- It effectively handled the dataset's high-dimensionality, identifying and leveraging key features without overfitting.
- The ensemble approach made it robust to noise, ensuring stable predictions across various cross-validation folds.
- Its interpretability allowed for actionable insights, making it easy to explain and visualize results.

#### Results and Conclusions
**Model Performance:** Logistic Regression

- Accuracy: ~74%.
- Strengths:
  - Best-performing traditional model for binary ‘explicit’ classification.
  - Robust to noise and irrelevant features.
  - Provided clear insights into feature importance.
- Weaknesses:
  - Logistic regression was inadequate for high-dimensional, multi-class problems.
  - Logistic regression is designed for binary classification, so it couldn’t directly use genre as a predictor since genre often has multiple categories (not binary).
  - Assumes linearity between features and log-odds of target variable, which may limit performance with complex, non-linear data
  - Since the original dataset is mostly non-explicit tracks, the prediction might only be good by coincidence.

**Findings:**
- Tracks with distinct feature patterns (e.g., high energy or danceability) were classified with higher accuracy.
- Popularity, while useful, was not a standalone indicator of explicitness and required complementary features for effective classification.

**Limitations:**
- Dataset size constrained the performance of more complex models, such as Neural Networks.
- Class imbalance led to poorer performance on explicit tracks.

**Conclusions:**
- Logistic Regression provided a strong foundation for understanding the dataset and making genre predictions.
- Future work could involve increasing the dataset size, balancing classes, and experimenting with advanced models like deep learning architectures.

#### Using the Code: Main.ipynb
**Preparation:**

- Ensure you have Python installed (preferably Python 3.8 or later)
- Ensure Python dependencies are installed, including pandas, scikit-learn, and matplotlib.
- Download the Spotify Tracks dataset and place it in the working directory.
  
**Execution:**
- Open the Jupyter Notebook main.ipynb.
- Run the notebook cells sequentially to:
- Preprocess the data.
  - Cleans the dataset
  - Splits the data into training and testing sets
  - Class Imbalance
- Model Training.
  - Implement a Logistic Regression model
  - Hyperparameters: solver='liblinear', class_weight='balanced’
- Evaluate model performance.
  - Calculates accuracy, confusion matrix, and other performance metrics.
- Visualize results, including feature importance and confusion matrices.
- Performed cross-validation metrics


## Appendix

1. Exploratory Data Analysis (EDA)
   
*Notebook: spotify_dataclean_EDA.ipynb*

**This key method was discussed in the main document**

**Visualizations:**
- Used correlation heatmaps to identify relationships between features such as danceability, energy, and popularity.
- Plotted feature distributions to detect outliers and understand the data's range.
- Scatterplots highlighted clusters in danceability and energy that corresponded to certain genres.
**Data Splitting:**
- Divided the dataset into 80% training and 20% testing subsets to evaluate model generalizability.

2. Data Preprocessing and Feature Engineering
   
*Notebook: spotify_dataclean_EDA.ipynb*
**This key method was discussed in the main document**
**Data Cleaning:**

- Removed irrelevant columns (track_id, artists, album_name, track_name) to focus on essential features.
- Handled missing data by dropping incomplete rows, ensuring the dataset was clean and reliable.

**Encoding:**
- Converted the explicit and track_genre columns into numeric formats using label encoding.
- 
**Feature Scaling:**
- Standardized all numerical features (e.g., popularity, danceability) using *StandardScaler* to normalize their ranges.
  
**Feature Importance:**
- Used Random Forest to rank features by importance, identifying danceability, energy, and popularity as the top contributors to genre prediction.

3. Regression Analysis
   
*Notebook: 148Project_LinearRegression.ipynb*

How was regression analysis applied:
- Linear Regression: Used to analyze relationships between predictors (e.g., danceability) and the target (popularity)
- Least Absolute Deviation (LAD) Regression: Applied alongside linear regression to minimize the absolute differences between predictions and actual values, making it robust to outliers.
- Metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) were calculated to assess performance.

What we learned about the data:
- Our model shows evidence of underfitting because the R-squared value is so low for both LS and LAD. It cannot predict well on the training dataset or the validation set.
- There was no relationship between danceability and popularity as the correlation was close to 0

Regularization:
- Ridge Regression was used. 
- The optimal alpha was determined through cross-validation, with results showing improved generalization performance on the validation set.
- However, regularization did not improve the performance of the model because the model was already underfitting to the current data
  
Insights:
- Danceability was not an impactful numeric feature for predicting popularity
- The track_genre column, being multi-class categorical, could not be directly used in regression analysis without encoding, highlighting the limitation of linear regression for such features.
- Showed that the relationship between danceability and popularity was a random scatter of points indicating there was no pattern to be determined between these two features.

4. Logistic Regression Analysis (described in main document)
   
*Notebook: Check_in_3_LogisticRegression.ipynb*

How was logistic regression applied:
- Logistic regression was used to predict the target variable, which likely had two possible outcomes (e.g., explicit or not explicit)
- The model’s performance was assessed using metrics like accuracy, recall, and ROC-AUC.
- Regularization (L2) was not necessary due to the model underfitting; therefore, regularization would not help.

Insights: 
- Logistic regression was inadequate for high-dimensional, multi-class problems.
- Logistic regression is designed for binary classification, so it couldn’t directly use genre as a predictor since genre often has multiple categories (not binary).
The analysis instead focused on numeric or binary variables for effective predictions.

6. Classification Methods
   
*Notebook: Check_in_4_Classification.ipynb*

**K-Nearest Neighbors (KNN):**
- Performed poorly due to the curse of dimensionality, with accuracy around ~20%.
- High computational costs for distance calculations in large datasets made it unsuitable for this task.
  
**Decision Trees:**
- Improved performance over KNN, achieving an accuracy of ~25%.
- Susceptible to overfitting, but results were improved with ensemble methods like Random Forest.

**Random Forest:**
- Outperformed other traditional models with an accuracy of ~30%.
- Leveraged ensemble learning to handle high-dimensional data and reduce overfitting.
- Provided critical insights into feature importance, guiding further analysis and interpretations.

6. Dimensionality Reduction + Clustering
   
*Notebook: Check_in_5_Unsupervised_Learning.ipynb*

**Clustering:**

- Applied clustering (e.g., k-means) to group tracks based on audio features.
- Insights: Certain genres were more tightly clustered, indicating distinct patterns in features like danceability. These clusters indicated a more distinctive patterned behavior around features like danceability, which aligns with our data findings as well. This analysis can help tailor recommendations and playlists by leveraging genre-specific audio feature patterns. 

7. Neural Networks
   
*Notebook: Check_in_6_NN.ipynb*

**Architecture:**
- Implemented a three-layer neural network with the following structure:
- Input Layer: Processed standardized features.
- Two Hidden Layers: Used ReLU activation with dropout to prevent overfitting.
- Output Layer: Applied softmax for multi-class classification.

**Results:**
- Training Accuracy: ~20%
- Validation Accuracy: ~26%
- Challenges:
  - Overfitting due to the limited dataset size and high feature dimensionality.
  - Struggled to generalize across genres, especially for underrepresented ones.

8. Hyperparameter Tuning
**Neural Network:**
- Experimented with:
  - Learning rates: 0.001 to 0.01
  - Batch sizes: 32, 64
  - Dropout rates: 0.2, 0.3
- We saw marginal improvements but did not fully resolve overfitting issues.

## Challenges and Limitations
- Class Imbalance:
  - Rare genres were underrepresented, causing misclassification and reduced accuracy.
- Dataset Size:
  - Insufficient samples limited the effectiveness of complex models like neural networks.
  - Limited samples restricted the ability to generalize well, especially for models that rely on larger datasets to capture nuanced patterns.
- Feature Complexity:
  - High-dimensional data posed challenges for simpler models, emphasizing the need for dimensionality reduction.
  - Feature complexity increased the likelihood of overfitting in models without proper regularization or preprocessing.
- Model Limitations
  - Some models, like binary logistic regression, were unable to handle multi-class classification directly, restricting their applicability for predicting categorical variables like genre.
  - The inability to directly use multi-class features without transformation (e.g., one-hot encoding) added complexity to the modeling process.


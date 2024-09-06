import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from scipy import stats

class AIEmployee:
    def __init__(self):
        self.data = None
        self.analysis_results = {}
        self.original_data = None
        self.label_encoders = {}
        self.target_column = None
        self.problem_type = None

    def ingest_data(self, file):
        try:
            file_type = file.name.split('.')[-1]
            if file_type == 'csv':
                self.data = pd.read_csv(file)
            elif file_type == 'json':
                self.data = pd.read_json(file)
            elif file_type == 'xlsx':
                self.data = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file type")
            self.original_data = self.data.copy()
            st.write(f"Data ingested. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            st.error(f"An error occurred during data ingestion: {e}")
            st.error(traceback.format_exc())

    def set_target_column(self, target_column):
        if target_column in self.data.columns:
            self.target_column = target_column
            if self.data[target_column].dtype == 'object' or self.data[target_column].nunique() < 10:
                self.problem_type = 'classification'
            else:
                self.problem_type = 'regression'
            st.write(f"Target column set to '{target_column}'. Problem type: {self.problem_type}")
        else:
            st.error(f"Column '{target_column}' not found in the dataset.")

    def preprocess_data(self):
      try:
        st.write("Starting data preprocessing...")
        # Handle missing values
        self.data = self.data.dropna()
        st.write(f"After removing NaN values, shape: {self.data.shape}")

        # Convert categorical columns to numeric
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
        st.write(f"Categorical columns encoded: {list(categorical_cols)}")

        # Scale numerical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        st.write(f"Numerical columns scaled: {list(numeric_cols)}")
        
        st.write("Data preprocessing completed successfully.")
      except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        st.error(traceback.format_exc())


    def analyze_data(self):
        try:
            st.write("Starting data analysis...")
            # Basic statistics
            self.analysis_results['basic_stats'] = self.original_data.describe(include='all')
            st.write("Basic statistics computed.")

            # Correlation analysis
            numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
            self.analysis_results['correlation'] = self.original_data[numeric_cols].corr()
            st.write("Correlation analysis completed.")

            # K-means clustering
            if len(numeric_cols) >= 3:
                kmeans = KMeans(n_clusters=min(3, self.data.shape[0]), random_state=42)
                self.analysis_results['kmeans'] = kmeans.fit_predict(self.data[numeric_cols])
                st.write("K-means clustering performed.")
            else:
                self.analysis_results['kmeans'] = None
                st.warning("Not enough numeric features for K-means clustering.")

            # PCA
            if len(numeric_cols) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(self.data[numeric_cols])
                self.analysis_results['pca'] = pca_result
                self.analysis_results['pca_explained_variance'] = pca.explained_variance_ratio_
                st.write("PCA performed.")

            # Outlier detection
            self.analysis_results['outliers'] = self.detect_outliers(numeric_cols)
            st.write("Outlier detection completed.")

            # Distribution analysis
            self.analysis_results['distributions'] = self.analyze_distributions(numeric_cols)
            st.write("Distribution analysis completed.")

            # Model training and evaluation
            if self.target_column is not None:
                self.train_and_evaluate_model()

            st.write("Data analysis completed successfully.")
            st.write(f"Analysis results keys: {list(self.analysis_results.keys())}")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.error(traceback.format_exc())

    def detect_outliers(self, columns):
        outliers = {}
        for col in columns:
            Q1 = self.original_data[col].quantile(0.25)
            Q3 = self.original_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = self.original_data[(self.original_data[col] < lower_bound) | (self.original_data[col] > upper_bound)]
        return outliers

    def analyze_distributions(self, columns):
        distributions = {}
        for col in columns:
            _, p_value = stats.normaltest(self.original_data[col])
            distributions[col] = {
                'skewness': self.original_data[col].skew(),
                'kurtosis': self.original_data[col].kurtosis(),
                'is_normal': p_value > 0.05
            }
        return distributions

    def train_and_evaluate_model(self):
        if self.target_column is None:
            st.warning("Target column is not set. Please set a target column before training a model.")
            return

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.problem_type == 'regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.analysis_results['model_performance'] = {
                'type': 'regression',
                'mse': mse,
                'r2': r2
            }
        else:  # classification
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.analysis_results['model_performance'] = {
                'type': 'classification',
                'accuracy': accuracy,
                'classification_report': report
            }

        st.write(f"Model training and evaluation completed for {self.problem_type} problem.")

    def display_basic_statistics(self):
        if 'basic_stats' in self.analysis_results:
            st.write("## Basic Statistics")
            st.write(self.analysis_results['basic_stats'])
        else:
            st.warning("No basic statistics available. Make sure to run the analysis first.")

    def display_correlation_heatmap(self):
        if 'correlation' in self.analysis_results:
            st.write("## Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(self.analysis_results['correlation'], annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No correlation analysis available. Make sure to run the analysis first.")

    def display_kmeans_clustering(self):
        if self.analysis_results.get('kmeans') is not None:
            st.write("## K-means Clustering Results")
            cluster_counts = pd.Series(self.analysis_results['kmeans']).value_counts()
            st.write(f"Cluster assignment count:\n{cluster_counts.to_string()}")
            
            if 'pca' in self.analysis_results:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(self.analysis_results['pca'][:, 0], self.analysis_results['pca'][:, 1], c=self.analysis_results['kmeans'], cmap='viridis')
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                ax.set_title('K-means Clustering Visualization (PCA)')
                plt.colorbar(scatter)
                st.pyplot(fig)
        else:
            st.warning("K-means clustering was not performed. Make sure to run the analysis first and that you have at least 3 numeric columns.")

    def display_pca_results(self):
        if 'pca' in self.analysis_results:
            st.write("## PCA Results")
            st.write(f"Explained variance ratio: {self.analysis_results['pca_explained_variance']}")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.bar(range(1, len(self.analysis_results['pca_explained_variance'])+1), self.analysis_results['pca_explained_variance'])
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('Scree Plot')
            st.pyplot(fig)
        else:
            st.warning("PCA was not performed. Make sure to run the analysis first and that you have at least 2 numeric columns.")

    def display_outliers(self):
        if 'outliers' in self.analysis_results:
            st.write("## Outlier Detection")
            for col, outliers in self.analysis_results['outliers'].items():
                st.write(f"### Outliers in {col}")
                st.write(outliers)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=self.original_data[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                st.pyplot(fig)
        else:
            st.warning("Outlier detection was not performed. Make sure to run the analysis first.")

    def display_distributions(self):
        if 'distributions' in self.analysis_results:
            st.write("## Distribution Analysis")
            for col, dist_info in self.analysis_results['distributions'].items():
                st.write(f"### Distribution of {col}")
                st.write(f"Skewness: {dist_info['skewness']:.2f}")
                st.write(f"Kurtosis: {dist_info['kurtosis']:.2f}")
                st.write(f"Normal distribution: {'Yes' if dist_info['is_normal'] else 'No'}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(self.original_data[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                st.pyplot(fig)
        else:
            st.warning("Distribution analysis was not performed. Make sure to run the analysis first.")

    def display_model_performance(self):
        if 'model_performance' in self.analysis_results:
            st.write("## Model Performance")
            performance = self.analysis_results['model_performance']
            if performance['type'] == 'regression':
                st.write(f"Mean Squared Error: {performance['mse']:.4f}")
                st.write(f"R-squared Score: {performance['r2']:.4f}")
            else:  # classification
                st.write(f"Accuracy: {performance['accuracy']:.4f}")
                st.write("Classification Report:")
                st.write(pd.DataFrame(performance['classification_report']).transpose())
        else:
            st.warning("Model performance results are not available. Make sure to run the analysis with a target column set.")

    def generate_summary(self):
        if self.original_data is not None:
            summary = f"The dataset contains {self.original_data.shape[0]} rows and {self.original_data.shape[1]} columns."
            st.write("## Summary")
            st.write(summary)
            
            st.write("### Column Types:")
            for col, dtype in self.original_data.dtypes.items():
                st.write(f"- {col}: {dtype}")

            if 'basic_stats' in self.analysis_results:
                st.write("### Key Insights:")
                numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    mean = self.analysis_results['basic_stats'].loc['mean', col]
                    std = self.analysis_results['basic_stats'].loc['std', col]
                    st.write(f"- {col}: Mean = {mean:.2f}, Std Dev = {std:.2f}")

            if 'correlation' in self.analysis_results:
                high_corr = self.analysis_results['correlation'].abs().unstack().sort_values(ascending=False).drop_duplicates()
                high_corr = high_corr[high_corr != 1.0][:5]  # Top 5 correlations
                st.write("### Top Correlations:")
                for (col1, col2), corr in high_corr.items():
                    st.write(f"- {col1} vs {col2}: {corr:.2f}")

            if 'model_performance' in self.analysis_results:
                st.write("### Model Performance Summary:")
                performance = self.analysis_results['model_performance']
                if performance['type'] == 'regression':
                    st.write(f"- Regression Model R-squared: {performance['r2']:.4f}")
                else:
                    st.write(f"- Classification Model Accuracy: {performance['accuracy']:.4f}")
        else:
            st.warning("No data available for summary. Make sure to upload a file first.")

def main():
    st.title("Enhanced AI Employee Data Analyst")

    ai_employee = AIEmployee()

    uploaded_file = st.file_uploader("Upload a dataset (csv, json, excel)", type=['csv', 'json', 'xlsx'])

    if uploaded_file is not None:
        data = ai_employee.ingest_data(uploaded_file)

        if data is not None:
            st.write("### Select Target Column")
            target_column = st.selectbox("Choose the target column for analysis:", options=['None'] + list(data.columns))
            
            if target_column != 'None':
                ai_employee.set_target_column(target_column)

            if st.button("Preprocess and Analyze Data"):
                with st.spinner("Processing data..."):
                    ai_employee.preprocess_data()
                    ai_employee.analyze_data()
                st.success("Data processing and analysis complete!")
                st.write(f"Final analysis results keys: {list(ai_employee.analysis_results.keys())}")

            st.sidebar.header("Display Options")

            if st.sidebar.checkbox("Show Summary"):
                ai_employee.generate_summary()

            if st.sidebar.checkbox("Show Basic Statistics"):
                ai_employee.display_basic_statistics()

            if st.sidebar.checkbox("Show Correlation Heatmap"):
                ai_employee.display_correlation_heatmap()

            if st.sidebar.checkbox("Show K-means Clustering Results"):
                ai_employee.display_kmeans_clustering()

            if st.sidebar.checkbox("Show PCA Results"):
                ai_employee.display_pca_results()

            if st.sidebar.checkbox("Show Outlier Detection"):
                ai_employee.display_outliers()

            if st.sidebar.checkbox("Show Distribution Analysis"):
                ai_employee.display_distributions()

            if st.sidebar.checkbox("Show Model Performance") and ai_employee.target_column is not None:
                ai_employee.display_model_performance()

if __name__ == "__main__":
    main()
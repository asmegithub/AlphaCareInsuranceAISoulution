import seaborn as sns
import matplotlib.pyplot as plt

def summarize_data(df):
    """Print descriptive statistics and data types."""
    print("Descriptive Statistics:\n", df.describe(include='all'))
    print("\nData Types:\n", df.dtypes)
def assess_data_quality(df):
    """Check for missing values and unique values in each column."""
    print("Missing Values:\n", df.isnull().sum())
    print("\nUnique Values per Column:\n", df.nunique())
    
def plot_univariate_distributions(df, numerical_features, categorical_features):
    """Plot histograms for numerical features and bar charts for categorical features."""
   

    # Plot histograms for numerical features
    for col in numerical_features:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Bar chart for categorical features
    for col in categorical_features:
        plt.figure(figsize=(8, 5))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.show()
        
        
def analyze_correlations(df, cols_to_correlate):
    """Generate and plot a correlation matrix for specified columns."""   

    correlation_matrix = df[cols_to_correlate].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    

def plot_bivariate_relationships(df, x_col, y_col, group_col):
    """Scatter plot to explore relationships between two variables grouped by another variable."""   

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, alpha=0.5)
    plt.title(f'{x_col} vs {y_col} by {group_col}')
    plt.show()
    
def compare_geographical_trends(df, group_col, compare_cols):
    # """Compare trends over geographical regions."""   

    grouped_data = df.groupby(group_col)[compare_cols].mean().reset_index()

    plt.figure(figsize=(12, 6))
    for col in compare_cols:
        sns.lineplot(data=grouped_data, x=group_col, y=col, marker='o', label=col)
    plt.title(f'Trends of {", ".join(compare_cols)} by {group_col}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
def detect_outliers(df, numerical_features):
    """Use box plots to detect outliers in numerical features."""


    for col in numerical_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f'Outliers in {col}')
        plt.show()
def create_visualizations(df):
    """Generate creative visualizations for EDA insights."""


    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['TotalPremium', 'TotalClaims', 'SumInsured']].corr(), annot=True, cmap='viridis')
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.show()

    # Scatter plot with regression line
    sns.lmplot(data=df, x='TotalPremium', y='TotalClaims', hue='Province', height=7, aspect=1.5)
    plt.title('TotalPremium vs TotalClaims with Regression Line by Province')
    plt.show()

    # Geographical trends (e.g., Premiums by Province)
    grouped_data = df.groupby('Province')['TotalPremium'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped_data, x='Province', y='TotalPremium', palette='Spectral')
    plt.title('Average TotalPremium by Province')
    plt.xticks(rotation=45)
    plt.show()

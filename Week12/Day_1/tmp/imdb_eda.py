import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import statsmodels.api as sm
import os
from scipy import stats

# Set Seaborn style
sns.set(style="whitegrid")

# Load dataset
def load_data(file_path):
    """
    Load the CSV data file into a pandas DataFrame.
    Enforce strict numeric conversion and remove any non-numeric artifacts.
    """
    df = pd.read_csv(file_path)
    
    # Print erroneous rows directly
    for num_col in ['Rating', 'Revenue (Millions)', 'Runtime (Minutes)', 'Year']:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
        # Logulate any row clearly displaying issues
        if df[num_col].isnull().any():
            print(f"Non-numeric entries in column '{num_col}':")
            print(df[df[num_col].isnull()][['Title', num_col]].head())

    # Clean dataset by dropping the faulty entries
    df_cleaned = df.dropna(subset=['Rating', 'Revenue (Millions)', 'Runtime (Minutes)', 'Year'])
    return df_cleaned

# 1. Data Overview & Quality Assessment

def data_overview_quality(df):
    """
    Provide an overview and quality assessment of the dataset.
    """
    # Basic info
    print("\nData Types and Info:")
    print(df.info())

    # Shape of the dataset
    print("\nDataset Shape:")
    print(df.shape)

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Visualize missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('missing_values_heatmap.png')
    
    # Statistical summary of numerical variables
    print("\nStatistical Summary:")
    print(df.describe())

# 2. Univariate Analysis

def univariate_analysis(df):
    """
    Perform the univariate analysis of the dataset.
    """
    # Movie Ratings
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    sns.histplot(df['Rating'], bins=20, kde=True)
    plt.title('Distribution of Movie Ratings')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(df['Rating'])
    plt.title('Box plot of Movie Ratings')
    
    plt.subplot(1, 3, 3)
    sns.violinplot(df['Rating'])
    plt.title('Violin plot of Movie Ratings')
    plt.tight_layout()
    plt.savefig('ratings_distribution.png')

    # Revenue Distribution with Outlier Detection
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Revenue (Millions)'], bins=20, kde=True)
    plt.title('Revenue Distribution')
    plt.savefig('revenue_distribution.png')

    # Genre frequency
    plt.figure(figsize=(14, 8))
    genre_data = df['Genre'].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    sns.barplot(x=genre_data.values, y=genre_data.index)
    plt.title('Genre Frequency')
    plt.savefig('genre_frequency.png')

    # Release Year Trends
    plt.figure(figsize=(10, 6))
    df['Year'].value_counts().sort_index().plot()
    plt.title('Number of Movies Released Each Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.savefig('release_year_trends.png')

    # Runtime Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Runtime (Minutes)'], bins=20, kde=True)
    plt.title('Runtime Distribution')
    plt.savefig('runtime_distribution.png')

# 3. Bivariate & Multivariate Analysis

def bivariate_multivariate_analysis(df):
    """
    Perform Bivariate and Multivariate analysis using plots.
    """
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.2)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')

    # Rating vs Revenue Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Rating', y='Revenue (Millions)', data=df)
    plt.title('Rating vs Revenue')
    plt.savefig('rating_revenue_scatter.png')

    # Genre vs Rating Box Plot
    plt.figure(figsize=(14, 6))
    melted_df = df[['Rating', 'Genre']].copy()
    melted_df = melted_df.dropna()
    melted_df['Genre'] = melted_df['Genre'].apply(lambda x: x.split(',')[0])
    sns.boxplot(x='Genre', y='Rating', data=melted_df)
    plt.title('Rating by Initial Genre')
    plt.xticks(rotation=45)
    plt.savefig('genre_vs_rating.png')

    # Director performance analysis
    top_directors = df[['Director', 'Rating']].groupby('Director').mean().sort_values(by='Rating', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_directors.index, y=top_directors['Rating'])
    plt.title('Top 10 Directors by Average Rating')
    plt.xticks(rotation=45)
    plt.savefig('top_directors.png')

    # Actor performance analysis
    actors_df = df['Actors'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
    actors_df = actors_df.to_frame('Actor').join(df['Rating'])
    top_actors = actors_df.groupby('Actor').mean().sort_values(by='Rating', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_actors.index, y=top_actors['Rating'])
    plt.title('Top 10 Actors by Average Rating')
    plt.xticks(rotation=45)
    plt.savefig('top_actors.png')

    # Year vs Rating trends over decades
    df['Decade'] = (df['Year'] // 10) * 10
    decade_rating = df[['Decade', 'Rating']].groupby('Decade').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(decade_rating.index, decade_rating['Rating'], marker='o')
    plt.title('Avg Rating Trends over Decades')
    plt.xticks(decade_rating.index, [f'{int(d)}s' for d in decade_rating.index])
    plt.savefig('decade_rating_trends.png')

# 4. Advanced Visualizations

def advanced_visualizations(df):
    """
    Create advanced visualizations.
    """
    # Top 20 highest-rated movies
    top20_rated_movies = df.nlargest(20, 'Rating')[['Title', 'Rating']].set_index('Title')
    plt.figure(figsize=(12, 8))
    top20_rated_movies.sort_values('Rating').plot(kind='barh', legend=False)
    plt.title('Top 20 Highest Rated Movies')
    plt.savefig('top_20_highest_rated.png')

    # Top 20 highest-grossing movies
    top20_grossing_movies = df.nlargest(20, 'Revenue (Millions)')[['Title', 'Revenue (Millions)']].set_index('Title')
    plt.figure(figsize=(12, 8))
    top20_grossing_movies.sort_values('Revenue (Millions)').plot(kind='barh', legend=False)
    plt.title('Top 20 Highest Grossing Movies')
    plt.savefig('top_20_highest_grossing.png')

    # Genre popularity over time
    genre_year_data = df['Genre'].str.get_dummies(sep=',').multiply(df['Year'], axis=0)
    genre_year_trends = genre_year_data.groupby(df['Year']).sum()
    genre_year_trends.plot(kind='area', stacked=True, figsize=(12, 6))
    plt.title('Genre Popularity Over Time')
    plt.savefig('genre_popularity_over_time.png')

    # Rating distribution by decade
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Decade', y='Rating', data=df)
    plt.xticks(rotation=45)
    plt.title('Rating Distribution by Decade')
    plt.savefig('rating_distribution_by_decade.png')

    # Revenue vs Rating colored by genre
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Rating', y='Revenue (Millions)', hue='Genre', data=df, palette='Dark2')
    plt.title('Revenue vs Rating by Genre')
    plt.savefig('revenue_vs_rating_by_genre.png')

# 5. Statistical Insights

def statistical_insights(df):
    """
    Derive statistical insights from the dataset.
    """
    # Outlier detection in Revenue
    revenue_outliers = df[df['Revenue (Millions)'] > df['Revenue (Millions)'].quantile(0.95)]
    print("Outliers in Revenue:")
    print(revenue_outliers[['Title', 'Revenue (Millions)']])

    # Key patterns in Ratings and Revenue
    reg_df = df[['Rating', 'Revenue (Millions)']].dropna()
    X = sm.add_constant(reg_df['Rating'])
    y = reg_df['Revenue (Millions)']
    trends = sm.OLS(y, X).fit()
    print("\nLinear regression model for Revenue based on Rating:")
    print(trends.summary())

# Main execution function
def main():
    file_path = 'https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv'
    df = load_data(file_path)
    data_overview_quality(df)
    univariate_analysis(df)
    bivariate_multivariate_analysis(df)
    advanced_visualizations(df)
    statistical_insights(df)

if __name__ == "__main__":
    main()
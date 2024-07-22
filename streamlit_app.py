import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
url = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df_cars = pd.read_csv(url)

# Streamlit app title
st.title("Car Dataset Analysis")

# Region filter buttons
region = st.sidebar.radio("Select Region", ("All", "US", "Europe", "Japan"))

# Filter data based on region
if region != "All":
    df_cars = df_cars[df_cars['continent'].str.contains(region)]

# Display the filtered dataset
st.write(f"## Dataset ({region})")
st.write(df_cars)

# Correlation heatmap
st.write("## Correlation Heatmap")
df_numeric = df_cars.select_dtypes(include=[float, int])
corr_matrix = df_numeric.corr()

fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, ax=ax)
st.pyplot(fig)

# Distribution plots
st.write("## Distribution Plots")

numeric_columns = df_numeric.columns

for col in numeric_columns:
    fig, ax = plt.subplots()
    sns.histplot(df_cars[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

# Comments section
st.write("## Comments")
st.write("""
The correlation heatmap shows the relationships between the numeric variables. High correlation values indicate strong relationships, 
either positive or negative. Distribution plots provide insights into the data distribution for each numeric feature.

- **High Correlation:** Variables like horsepower and weight show high positive correlation, indicating that heavier cars tend to have more horsepower.
- **Distribution Insights:** The distribution plots reveal the spread and central tendency of the data, with features like mpg showing a normal distribution, whereas weight and horsepower show skewed distributions.
""")

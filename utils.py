#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# utils.py
# Utility module for "The Air We Breathe" project.
# Contains shared constants, helper functions, EDA functions, and model
# evaluation / analysis functions used across both notebooks.
# =============================================================================

import warnings
import math

# Libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from IPython.display import display

# Statistical libraries
import scipy.stats as stats
from scipy.stats import chi2_contingency

# Machine Learning Interpretation
import shap as sh

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_pinball_loss,
)

# Configuration
warnings.filterwarnings("ignore")
sns.set_style('white')


# =============================================================================
# EU AQI CONSTANTS
# Thresholds based on the European Air Quality Index (updated February 2026).
# Reference: https://airindex.eea.europa.eu/AQI/index.html
# =============================================================================

EU_AQI_THRESHOLDS = {
    "pm2.5": [
        (0, 6, "Good"),
        (6, 16, "Fair"),
        (16, 51, "Moderate"),
        (51, 91, "Poor"),
        (91, 141, "Very Poor"),
        (141, 800, "Extremely Poor")
    ],
    "pm10": [
        (0, 16, "Good"),
        (16, 46, "Fair"),
        (46, 121, "Moderate"),
        (121, 196, "Poor"),
        (196, 271, "Very Poor"),
        (271, 1200, "Extremely Poor")
    ],
    "no2": [
        (0, 11, "Good"),
        (11, 26, "Fair"),
        (26, 61, "Moderate"),
        (61, 101, "Poor"),
        (101, 151, "Very Poor"),
        (151, 1000, "Extremely Poor")
    ],
    "o3": [
        (0, 61, "Good"),
        (61, 101, "Fair"),
        (100, 121, "Moderate"),
        (121, 161, "Poor"),
        (161, 181, "Very Poor"),
        (181, 800, "Extremely Poor")
    ],
    "so2": [
        (0, 21, "Good"),
        (21, 41, "Fair"),
        (41, 126, "Moderate"),
        (126, 191, "Poor"),
        (191, 276, "Very Poor"),
        (276, 1250, "Extremely Poor")
    ]
}

# Class orders from best to worst
EU_AQI_ORDER = ["Good", "Fair", "Moderate", "Poor", "Very Poor", "Extremely Poor"]


# =============================================================================
# WIND DIRECTION CONSTANTS
# Used to convert wind direction from degrees to cardinal directions.
# =============================================================================

wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
degrees_direction = [d * 22.5 for d in range(17)]


# =============================================================================
# CLUSTERING CONSTANTS
# Day type labels assigned to each K-Means cluster (12 clusters).
# =============================================================================

day_type_labels = {
    0:  "windy_humid_worsening",
    1:  "dry_stable_calm",
    2:  "overcast_persistent_light_rain",
    3:  "hot_dry_stable",
    4:  "mild_unstable_showers",
    5:  "variable_clearing",
    6:  "very_humid_overcast_calm",
    7:  "strongly_perturbed_very_windy",
    8:  "very_humid_frequently_rainy",
    9:  "continuous_stratiform_rain",
    10: "slightly_unstable_variable",
    11: "mild_stable_partly_cloudy"
}


# =============================================================================
# STYLE SETTINGS
# =============================================================================

def set_palette(n_colors=10):
    if n_colors <= 20:
        custom_palette = [
            "#6F5A8A",  # dusty purple
            "#3F5E7A",  # muted blue
            "#2E6C80",  # subdued teal blue
            "#6F8F3A",  # desaturated olive green
            "#D4B866",  # sandy yellow
            "#C97C3A",  # terracotta orange
            "#8C3F3F",  # brownish red
            "#2E4F63",  # muted deep blue
            "#C76A72",  # dusty salmon red
            "#9A5A83",  # muted magenta
            "#565084",  # soft dark purple
            "#8F5A86",  # desaturated purple
            "#C45A5F",  # soft red
            "#D1B04A",  # soft ochre
            "#6A9A86",  # sage green
            "#3F7DA6",  # muted light blue
            "#C87A55",  # soft warm orange
            "#B56183",  # muted fuchsia
            "#5E4E82",  # soft vintage purple
            "#D39A3A"   # soft ochre orange
            ]
        return custom_palette[:n_colors]
    else:
        return sns.color_palette("husl", n_colors)


hcmap = LinearSegmentedColormap.from_list(
    "custom_diverging",
    [
    "#355C7D",  # deep blue
    "#4F6F89",  # muted blue
    "#8AA0B0",  # blue-gray
    "#F2F2F2",  # neutral light gray
    "#B58A8A",  # muted red-gray
    "#9E575E",  # subdued red
    "#8C2F39"   # deep red
    ],
    N=256
)

hcmap_mono = LinearSegmentedColormap.from_list(
    "custom_diverging",
    [
    "#F2F2F2",  # neutral
    "#8AA0B0",  # light blue - grey
    "#4F6F89",  # muted blue
    "#355C7D"  # blue
    ],
    N=256
)


# =============================================================================
# EU AQI CLASSIFICATION FUNCTIONS
# =============================================================================

# Function to classify a single pollutant
def classify_level(value, thresholds):
    if pd.isna(value):
        return 'missing_vals'
    for min_val, max_val, level in thresholds:
        if min_val <= value < max_val:
            return level
    return "Extremely Poor"  # Above last threshold


# The EU AQI level is determined by the worst level between the pollutants.

# Function to compute final EU AQI for a row
def calculate_eu_aqi_category(row):
    worst_level = "Good"
    for pollutant in EU_AQI_THRESHOLDS:
        if pollutant in row:
            if pd.isna(row[pollutant]):
                return 'missing_vals'
            level = classify_level(row[pollutant], EU_AQI_THRESHOLDS[pollutant])
            # Compare magnitude
            if EU_AQI_ORDER.index(level) > EU_AQI_ORDER.index(worst_level):
                worst_level = level
    return worst_level


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def cardinal_direction(degrees):
    if math.isnan(degrees):
        return 'UNKNOWN'
    if degrees > 360:
        degrees -= 360
    closest_match = 0

    delta_degrees = abs(degrees_direction[0] - degrees)

    for i in range(1, len(degrees_direction)):
        difference = abs(degrees_direction[i] - degrees)
        if difference < delta_degrees:
            delta_degrees = difference
            closest_match = i
    return wind_directions[closest_match]


# There are different columns that represent a time of the day, it is useful to group them in four distinct moments
# of day: morning, afternoon, evening and night.
def day_moment(time):
    if pd.isna(time):
        return 'empty'
    elif time.hour >= 0 and time.hour < 6:
        return 'night'
    elif time.hour >= 6 and time.hour < 12:
        return 'morning'
    elif time.hour >= 12 and time.hour < 18:
        return 'afternoon'
    else:
        return 'evening'


# =============================================================================
# EDA UTILITY FUNCTIONS
# =============================================================================

# Definition of a function used to describe a numeric column with statistical informations for it, together with
# a measure of the skew of the distribution, an histogram and a boxplot
# This function is meant to be used for continuous variables, for which the informations displayed are more relevant
def describe_column(data, col, scale=True):
    print('Description of column', col)
    # First of all it is shown the number of null values, for that column, selecting just the rows of the dataframe
    # where it verified, and getting the number of rows resulting with the shape attribute
    print('The column', col, 'has', data[data[col].isna()].shape[0], 'missing values')
    # CHECK FOR ZERO VALUES
    print('The column', col, 'has', data[data[col] == 0].shape[0], 'zero values')
    # The describe() function transposed is used again to show the statistical informations of the column
    display(pd.DataFrame(data[col].describe()).T)
    # The skew() function gives a measure of the skewness of the feature's distribution
    print('Skew of the column:', data[col].skew())
    # The subplot is used to print two plots in a row, sharing the x-axis to give a better idea of the distribution's shape
    f1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 5))
    # The histplot() function from Seaborn is used to display the histogram; the kernel density estimator is shown
    # with kde = True to give a better idea of the distribution's shape

    if ((data[col].skew() > 4) and (scale)):
        sns.histplot(data, x=col, ax=ax1, kde=True, log_scale=True, color="#355C7D", edgecolor="white", alpha=0.8)
        ax1.set_title('Distribution of column ' + col, loc='left', fontsize=14, fontweight='bold', x=-0.1)
        ax1.set_ylabel("Frequency")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        sns.boxplot(data, x=col, ax=ax2, showmeans=True, log_scale=True, color="#F67280", fliersize=3, linewidth=1)
        ax2.grid(True, axis='x', linestyle="--", linewidth=0.5, alpha=0.6)
    else:
        sns.histplot(data=data, x=col, ax=ax1, kde=True, color="#355C7D", edgecolor="white", alpha=0.8)
        ax1.set_title('Distribution of column ' + col, loc='left', fontsize=14, fontweight='bold', x=-0.1)
        ax1.set_ylabel("Frequency")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        sns.boxplot(data=data, x=col, ax=ax2, showmeans=True, color="#F67280", fliersize=3, linewidth=1)
        ax2.grid(True, axis='x', linestyle="--", linewidth=0.5, alpha=0.6)
    # The boxplot() function is used to display a box-plot; the mean is also displayed with the parameter showmeans = True

    # Removing external borders
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    plt.show()


# Definition of a function used to describe a discrete numeric column with absolute and relative frequencies for
# each of its values together with a count plot to display the frequency for each value
# This function is meant to be used for discrete variables, for which the informations displayed are more relevant
def describe_discrete_column(data, c):
    print('Description of column', c)
    # First of all it is shown the number of null values, for that column, selecting just the rows of the dataframe
    # where it verified, and getting the number of rows resulting with the shape attribute
    print('The column', c, 'has', data[data[c].isna()].shape[0], 'missing values')
    # A dataframe is created with the absolute frequency for the feature in the only column
    # the reset_index function transforms the index (representing the values of the feature) in a separate column
    temp_df = pd.DataFrame(data[c].value_counts()).reset_index()
    # Renaming the columns
    temp_df.columns = [c + ' values', 'Absolute frequency']
    # A column with relative frequencies is added to the dataframe
    temp_df['Relative frequency (%)'] = round((temp_df['Absolute frequency'] / data.shape[0]) * 100, 2)
    temp_df.sort_values(by='Absolute frequency', ascending=False, inplace=True)
    display(temp_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    # The histplot() function is used to display a countplot: normally it is not its use
    # but it has been done in this way to be able to also show values for the feature which had no occurrencies in the data
    # giving a better idea of the distribution and eventually of outliers
    # The bins are for discrete values using the attribute discrete = True, and setting the shrink attribute
    # the columns are kept separed from one another (like in a normal count plot)
    sns.histplot(data=data, x=c, discrete=True, shrink=.8, color='#355C7D',
                 bins=np.arange(data[c].min(), data[c].max() + 1, 1))

    # The bar labels are used to add the absolute frequency on top of each bin
    ax.bar_label(ax.containers[0], color='black', fontsize=10)

    sns.despine()
    # The xticks are set to just see on the x axis the discrete values corresponding to each bin
    plt.xticks(np.arange(data[c].min(), data[c].max() + 1, 1))

    plt.suptitle('Number of occurrences for each value of column ' + c, fontsize=14, ha='left', fontweight='bold', x=0.1)

    plt.show()


# Definition of a function used to describe categorical columns
def describe_cat_column(data, c, resort=True):
    print('Description of column', c)
    # First of all it is shown the number of null values, for that column, selecting just the rows of the dataframe
    # where it verified, and getting the number of rows resulting with the shape attribute
    print('The column', c, 'has', data[data[c].isna()].shape[0], 'missing values')
    # A dataframe is created with the absolute frequency for the feature in the only column
    # the reset_index function transforms the index (representing the values of the feature) in a separate column
    temp_df = pd.DataFrame(data[c].value_counts()).reset_index()
    # Renaming the columns
    temp_df.columns = [c + ' values', 'Absolute frequency']
    # A column with relative frequencies is added to the dataframe
    temp_df['Relative frequency (%)'] = round((temp_df['Absolute frequency'] / data.shape[0]) * 100, 2)
    temp_df.sort_values(by='Absolute frequency', ascending=False, inplace=True)
    display(temp_df)

    fig, ax = plt.subplots(figsize=(12, 5))

    if resort:
        sns.countplot(data=data[data[c].isna() == False], order=data[data[c].isna() == False][c].value_counts().index,
                      x=c, color='#355C7D')
    else:
        sns.countplot(data=data[data[c].isna() == False], x=c, color='#355C7D')

    sns.despine()
    # The bar labels are used to add the absolute frequency on top of each bin
    ax.bar_label(ax.containers[0], color='black', fontsize=10)
    plt.suptitle(c + '- values distribution', fontsize=14, weight='bold', ha='left', x=0.05)

    plt.show()


# Definition of another function used to describe categorical columns, this one meant to be used for columns with more
# unique values, as it displays an horizontal barplot for an improved readability
def describe_cat_column_h(data, c, resort=True, custom_title=''):
    print('Description of column', c)
    # First of all it is shown the number of null values, for that column, selecting just the rows of the dataframe
    # where it verified, and getting the number of rows resulting with the shape attribute
    print('The column', c, 'has', data[data[c].isna()].shape[0], 'missing values')
    # A dataframe is created with the absolute frequency for the feature in the only column
    # the reset_index function transforms the index (representing the values of the feature) in a separate column
    temp_df = pd.DataFrame(data[c].value_counts()).reset_index()
    # Renaming the columns
    temp_df.columns = [c + ' values', 'Absolute frequency']
    # A column with relative frequencies is added to the dataframe
    temp_df['Relative frequency (%)'] = round((temp_df['Absolute frequency'] / data.shape[0]) * 100, 2)
    temp_df.sort_values(by='Absolute frequency', ascending=False, inplace=True)
    display(temp_df)

    if temp_df.shape[0] > 6:
        fig, ax = plt.subplots(figsize=(9, 10))
    else:
        fig, ax = plt.subplots()
    if resort:
        sns.countplot(data=data[data[c].isna() == False], order=data[data[c].isna() == False][c].value_counts().index,
                      y=c, color='#355C7D')
    else:
        sns.countplot(data=data[data[c].isna() == False], y=c, color='#355C7D')

    sns.despine()
    # The bar labels are used to add the absolute frequency on top of each bin
    ax.bar_label(ax.containers[0], color='black', fontsize=10)
    # The xticks are set to just see on the x axis the discrete values corresponding to each bin
    if custom_title == '':
        plt.suptitle('Number of occurrences for each value of column ' + c, fontsize=14, ha='left', fontweight='bold', x=-0.05)
    else:
        plt.suptitle(custom_title, ha='left', fontsize=14, fontweight='bold', x=-0.05)
    plt.subplots_adjust(top=0.93)
    plt.show()


# Definition of a function used to show the relationship between two numerical variables
# It takes as an input a dataframe and tuple of two column names
# It shows their correlation value
# Then it displays a scatterplot putting in evidence with the scatterplot how the points are distributed
# together with a regression line to better highlight the relationship
def num_corr(df, t):
    # Set dimensions in inches
    plt.figure(figsize=(6, 6))
    sns.regplot(data=df, x=t[0], y=t[1],
                scatter_kws={'s': 30, 'alpha': 0.7, 'color': "#003f5c"},
                line_kws={'color': '#f95d6a', 'linewidth': 2})

    plt.suptitle("Relationship between " + t[0] + " and " + t[1], fontsize=14, weight='bold', ha='left', x=0.0)
    plt.subplots_adjust(top=0.93)

    sns.despine()
    plt.show()

    print('\n Correlation test between', t[0], 'and', t[1], '\n')
    correlation = df[t[0]].corr(df[t[1]])
    print("The correlation between", t[0], "and", t[1], "is", round(correlation, 5))


# Definition of a function that displays the relationship between a categorical column and a numerical one
# It displays the result of the Kruskal-Wallis H-test, that measure if there is a relevant difference of the median
# of the numerical feature for at least two values of the categorical one
# Then it gives the statistical summary of the numerical feature computed for each value of the categorical
# Finally it displays a boxplot of the numerical feature of each group of the categorical
# The mean is shown as well, to highlight for each group how much the mean is different to the median
# (this gives an idea of how much that distribution is skewed)
def cat_num_rel(df, cat, num, lgscale=True):
    # Group the df by the categorical column cat
    groups = df[df[num].isna() == False].groupby(cat)[num]

    # Extract the groups into a list of arrays, one for each level of the categorical column
    df_groups = [group.values for name, group in groups]

    # Perform the Kruskal-Wallis H-test
    # This test is used as it does not have the requirements of normality that others have
    stat, p_value = stats.kruskal(*df_groups)

    if p_value >= 0.05:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', num, 'is', p_value,
              'so the median of', num, 'for the different groups does not show relevant deviations.')
    else:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', num, 'is', p_value,
              'so there is a significant deviation of the median of', num, 'in at least two groups.')
    # For each element of the categorical column it is given a statistical summary of the numerical column
    display(df.groupby(cat)[num].describe())

    y_vals_list = df[cat].unique().tolist()
    y_vals_list.sort()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(
        data=df,
        x=num,
        y=cat,
        color='#355C7D',
        order=y_vals_list,
        split=True,
        inner="quart",
        linewidth=1,
        linecolor='white',
        ax=ax,
        legend=False
    )

    xmin = df[num].min()
    xmax = df[num].max()
    pad = 0.02 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

    plt.suptitle("Distribution of " + num + " by " + cat, fontsize=14, weight='bold', ha='left', x=0.0)
    sns.despine()
    plt.tight_layout()
    plt.show()


# Definition of a function used to represent the relationship between two categorical features
# It performs the Chi squared test, to have an analytical measure of the strength of their relationship
# Then it displays a contingency table between the two, to see how many values fall in each combination of categories
# Finally it displays a stacked barplot to show visually, in each category of one feature, how many values there are
# of each value of the other feature
def cat_cat_rel(df, ccol1, ccol2, legend_title='', clr=[], rotate_tab=False):
    # The contingency table is computed with the function crosstab() that gives the total quantity of data points that
    # belong to each combination of categories
    contingency_table = pd.crosstab(df[ccol1], df[ccol2])

    # Performing the Chi-Squared test
    chi2, pval, dof, expected = chi2_contingency(contingency_table)

    # The null hypothesys is that the two columns are independent, we reject it with a p value lower than 0.05
    if pval >= 0.05:
        print('The result of the Chi Squared test does not leave us reject the null hypotesys that the column',
              ccol1, 'and the column', ccol2, 'are independent to each other')
    else:
        print('The result of the Chi Squared test makes us reject the null hypotesys, so the column',
              ccol1, 'and the column', ccol2, 'are dependent to each other')

    # count is necessary to customize the dimensions of the plot
    count = df[ccol2].nunique()
    # Defined to sort the values in the second contingency table from the lowest to the highest
    sort = df[ccol1].value_counts().index[-1]

    # A second contingency table, holding the percentuals instead that the counts, is defined
    # It is done with the parameter normalize = "index" in the function crosstab
    contingency_table_p = pd.crosstab(df[ccol2], df[ccol1], normalize="index").sort_values(by=sort, ascending=False)

    # To sort elements based on the sequence in clr list
    if clr:
        contingency_table = contingency_table.loc[clr]
        contingency_table_p = contingency_table_p[clr]

    # Display of the contingency table
    if rotate_tab:
        display(contingency_table.T)
    else:
        display(contingency_table)

    # The stacked barplot is defined using the parameter stacked = True
    count = df[ccol2].nunique()

    if len(clr) == 0:
        contingency_table_p.plot(kind="barh", stacked=True, figsize=(5, (count + 1)),
                                 color=sns.set_palette(set_palette(n_colors=len(contingency_table_p.columns))))
    else:
        color_palette = sns.set_palette(set_palette(n_colors=len(clr)))
        contingency_table_p.plot(kind="barh", stacked=True, figsize=(5, (count + 1)),
                                 color=color_palette)
    # The position of the legend is set in a way that does not gets in the way of the plot

    if legend_title == '':
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title=legend_title)
    sns.despine()
    plt.suptitle("Distribution of " + ccol2 + " by " + ccol1, fontsize=14, weight='bold', ha='left', x=0.0)
    plt.subplots_adjust(top=0.93)

    # Display of the plot
    plt.show()


def categories_by_groups(df, groups, categories, ordered_categories):

    df[categories] = pd.Categorical(df[categories], categories=ordered_categories, ordered=True)

    # Obtain the names of the unique groups
    unique_groups = df[groups].unique()

    # Itera su ogni gruppo per generare la tabella e il grafico
    for group_name in unique_groups:
        print('*' * 45, group_name, '*' * 45)

        # Filter the DataFrame for the current group
        group_df = df[df[groups] == group_name]

        # Compute absolute frequency
        absolute_frequency = group_df[categories].value_counts().reindex(ordered_categories, fill_value=0)

        # Compute relative frequency (%)
        total_count = absolute_frequency.sum()
        relative_frequency = (absolute_frequency / total_count * 100).round(2)

        # Create a DataFrame for the table
        table_data = pd.DataFrame({
            categories: absolute_frequency.index,
            'Absolute frequency': absolute_frequency.values,
            'Relative frequency (%)': relative_frequency.values
        })

        # Print the table
        display(table_data)

        # Generate a barplot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=absolute_frequency.index, y=absolute_frequency.values, color='#355C7D')
        sns.barplot(x=absolute_frequency.index, y=absolute_frequency.values, color='#355C7D')

        plt.suptitle(categories + ' in case of ' + group_name, fontsize=14, weight='bold', ha='left', x=0.0)
        plt.subplots_adjust(top=0.93)

        plt.xlabel(categories)
        plt.ylabel('Absolute Frequency')
        plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability

        for container in ax.containers:  # ax.containers contains plot elements (the bars)
            ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10, padding=3)

        plt.tight_layout()
        plt.show()


def numx_numy_cat_rel(df, numx, numy, cat, legend_title='', clr=[]):
    # Drawing the plot first, showing a FacetGrid between the two numerical features with distinction based on the categorical

    g = sns.FacetGrid(df, col=cat, col_wrap=3, height=3, aspect=1)
    g.map_dataframe(sns.regplot, x=numx, y=numy,
                    scatter_kws={'s': 20, 'alpha': 0.6, 'color': "#355C7D"},
                    line_kws={'color': '#f95d6a', 'linewidth': 1.2})
    g.set_titles(col_template="{col_name}", size=10, fontweight='bold')
    g.set_axis_labels(numx, numy)

    sns.despine()

    plt.suptitle(numx + " vs. " + numy + " distributed by " + cat, fontsize=14, weight='bold', ha='left', x=0.0)
    plt.subplots_adjust(top=0.92)

    plt.show()

    # Correlation test between the two numerical features

    print('\n Correlation test between', numx, 'and', numy, '\n')
    correlation = df[numx].corr(df[numy])
    print("The correlation between", numx, "and", numy, "is", round(correlation, 5))

    # First ANOVA test
    print("\n\n ANOVA test between the values of ", numx, "based on the groups of", cat, "\n")

    groups = [df[df[cat] == category][numx] for category in df[cat].unique()]
    stat, p_value = stats.kruskal(*groups, nan_policy='omit')

    if p_value >= 0.05:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', numx, 'is', round(p_value, 5),
              'so the median of', numx, 'for the different groups does not show relevant deviations.')
    else:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', numx, 'is', round(p_value, 5),
              'so there is a significant deviation of the median of', numx, 'in at least two groups.')

    # Second ANOVA test
    print("\n\n ANOVA test between the values of ", numy, "based on the groups of", cat, "\n")

    groups2 = [df[df[cat] == category][numy] for category in df[cat].unique()]
    stat2, p_value2 = stats.kruskal(*groups2, nan_policy='omit')

    if p_value2 >= 0.05:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', numy, 'is', round(p_value2, 5),
              'so the median of', numy, 'for the different groups does not show relevant deviations.')
    else:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', numy, 'is', round(p_value2, 5),
              'so there is a significant deviation of the median of', numy, 'in at least two groups.')


def numx_caty_cat_rel(df, numx, caty, cat, legend_title='', clr=[]):
    # To represent the relationship between two categorical features and a numerical one a beeswarm plot is used

    plt.figure(figsize=(8, 8))
    if len(clr) == 0:
        colormap = sns.set_palette(set_palette(n_colors=df[cat].nunique()))
        pl = sns.swarmplot(data=df, x=numx, y=caty, hue=cat, palette=colormap)
    else:
        color_palette = sns.color_palette(set_palette(n_colors=df[cat].nunique()), n_colors=len(clr))
        palette_dict = {cat: color for cat, color in zip(clr, color_palette)}
        pl = sns.swarmplot(data=df, x=numx, y=caty, hue=cat, palette=palette_dict, hue_order=clr)

    sns.despine()

    plt.suptitle(numx + ' by ' + caty + ' and ' + cat, fontsize=14, weight='bold', ha='left', x=0.0)
    plt.subplots_adjust(top=0.93)

    if legend_title == '':
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title=cat)
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title=legend_title)
    plt.show()

    # First ANOVA test
    print("\n\n ANOVA test between the values of ", numx, "based on the groups of", cat, "\n")

    groups = [df[df[cat] == category][numx] for category in df[cat].unique()]
    stat, p_value = stats.kruskal(*groups, nan_policy='omit')

    if p_value >= 0.05:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', numx, 'is', round(p_value, 5),
              'so the median of', numx, 'for the different groups does not show relevant deviations.')
    else:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              cat, 'and the column', numx, 'is', round(p_value, 5),
              'so there is a significant deviation of the median of', numx, 'in at least two groups.')

    # Second ANOVA test
    print("\n\n ANOVA test between the values of ", numx, "based on the groups of", caty, "\n")

    groups2 = [df[df[caty] == category][numx] for category in df[caty].unique()]
    stat2, p_value2 = stats.kruskal(*groups2, nan_policy='omit')

    if p_value2 >= 0.05:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              caty, 'and the column', numx, 'is', round(p_value2, 5),
              'so the median of', numx, 'for the different groups does not show relevant deviations.')
    else:
        print('The Kruskal-Wallis H-test for the different groups of the column',
              caty, 'and the column', numx, 'is', round(p_value2, 5),
              'so there is a significant deviation of the median of', numx, 'in at least two groups.')

    # Chi-Squared Test
    print("\n\n Chi-Squared test between the elements of ", cat, "and those of", caty, "\n")
    contingency_table = pd.crosstab(df[caty], df[cat])

    # Performing the Chi-Squared test
    chi2, pval, dof, expected = chi2_contingency(contingency_table)

    # The null hypothesys is that the two columns are independent, we reject it with a p value lower than 0.05
    if pval >= 0.05:
        print('The result of the Chi Squared test, with a Pvalue of', round(pval, 5),
              ' does not leave us reject the null hypotesys that the column',
              cat, 'and the column', caty, 'are independent to each other')
    else:
        print('The result of the Chi Squared test, with a Pvalue of', round(pval, 5),
              'makes us reject the null hypotesys, so the column',
              cat, 'and the column', caty, 'are dependent to each other')


# =============================================================================
# MODEL EVALUATION FUNCTIONS
# Since the type of regression that will be done is a quantile regression the target metric is the pinball loss,
# however for the median model it is equal to half the MAE, which is a much easier metric to interpret for people,
# so it will be used to represent the quality of median models together with pinball score.
# =============================================================================

# Function to compute MAE to check performance of a regression model
def model_performance_regression(model, xt, yt):
    pred = model.predict(xt)                  # Predict using the independent variables
    mae = mean_absolute_error(yt, pred)       # To compute MAE

    # Creating a dataframe of metrics
    perf = pd.DataFrame(
        {
            "MAE": mae
        },
        index=[0],
    )

    return perf


# Function to compute MAE to check performance of a regression model with the digits rounded to two decimals
# This function will be used to compare the results of the final model choosed with other candidates
def model_performance_regression_rounded(model, model_name, xt, yt):
    pred = model.predict(xt)                  # Predict using the independent variables
    mae = mean_absolute_error(yt, pred)       # To compute MAE

    # Creating a dataframe of metrics
    perf = pd.DataFrame(
        {
            "MAE": round(mae, 2)
        },
        index=[model_name],  # Using the model name as the index of the row
    )

    return perf


# Function to compute MAE on both train set and test set
# The name variable is used to make the output easier to read
# Used for making a more compact display instead of calling model_performance_regression two times
def train_test_performances(mod, xtr, ytr, xte, yte, name):
    # Metrics on the training set
    train_r = model_performance_regression(mod, xtr, ytr)
    # Metrics on the test set
    test_r = model_performance_regression(mod, xte, yte)
    # COncatenating the two dataframes
    train_test_r = pd.concat([train_r, test_r])

    # Setting the indexes with the name passed to the function plus train and test respectively
    # This make easier to read for what model the metrics are calculated and on what part of the dataset

    train_test_r.reset_index(inplace=True)
    train_test_r.drop('index', axis=1, inplace=True)
    ind = [name + ' train', name + ' test']
    train_test_r.index = ind
    return train_test_r


# =============================================================================
# BASELINE MODEL FUNCTIONS
# It is important to be able to compare the scores obtained with complex models to something simpler, because if
# the more complex models performs only a little better than a very simple one (or equally or even worse) even with
# a low pinball loss (or any other metric) it is not a good idea to use it.
#
# So in this case a given quantile is computed grouping the target training values by month, and the results are
# used to predict the test data. If it is not possible to outperform those results with a machine learning model
# it is better to just use them as an estimation.
# =============================================================================

def baseline(df, target, train_data, test_data, quantile):
    data_months = df[['month of year', target]].copy()
    month_pred = pd.DataFrame(df.loc[train_data.index].groupby('month of year')[target].quantile(quantile)).reset_index()
    data_months = data_months[data_months[target].isna() == False]
    data_months.reset_index(inplace=True)
    estimation = 'quantile ' + str(quantile) + ' ' + target
    month_pred.columns = ['month of year', estimation]
    data_months = data_months.merge(month_pred)
    data_months.index = data_months['date']
    data_months.drop('date', axis=1, inplace=True)
    baseline_train = mean_pinball_loss(data_months.loc[train_data.index][target],
                                      data_months.loc[train_data.index][estimation],
                                      alpha=quantile)
    baseline_test = mean_pinball_loss(data_months.loc[test_data.index][target],
                                     data_months.loc[test_data.index][estimation],
                                     alpha=quantile)
    baseline_tot = mean_pinball_loss(data_months[target], data_months[estimation], alpha=quantile)

    return (baseline_train, baseline_test, baseline_tot)


def baseline_mean(df, target, train_data, test_data):
    data_months = df[['month of year', target]].copy()
    month_pred = pd.DataFrame(df.loc[train_data.index].groupby('month of year')[target].mean()).reset_index()
    data_months = data_months[data_months[target].isna() == False]
    data_months.reset_index(inplace=True)
    estimation = 'mean'
    month_pred.columns = ['month of year', estimation]
    data_months = data_months.merge(month_pred)
    data_months.index = data_months['date']
    data_months.drop('date', axis=1, inplace=True)
    baseline_train = np.sqrt(mean_squared_error(data_months.loc[train_data.index][target],
                                                data_months.loc[train_data.index][estimation]))
    baseline_test = np.sqrt(mean_squared_error(data_months.loc[test_data.index][target],
                                               data_months.loc[test_data.index][estimation]))
    baseline_tot = np.sqrt(mean_squared_error(data_months[target], data_months[estimation]))

    return (baseline_train, baseline_test, baseline_tot)


# =============================================================================
# MODEL ANALYSIS FUNCTIONS (SHAP)
# One of the key interest in the project is to see HOW weather conditions influence pollutant levels, as this
# gives to people ready-to-use informations beside the models predictions.
# With a machine learning model it is possible to see what importance each feature has in the estimation and also
# how their combination of values causes (maybe) different results, capturing complex relations that would be
# otherwise transparent in a simpler analysis. To do so it will be used the Shap library and custom functions are
# defined to represent how different values of categorical features, and also the interactions between two features
# influence the models predictions.
#
# This estimation will be performed on the median models, as the target is to take them as first approximation
# of the real values.
# =============================================================================

def plot_shap_boxplot_colored(X, shap_vals, feature_name, cmap='coolwarm'):
    # Prepare data
    shap_feature_values = shap_vals[:, feature_name].values
    df_plot = pd.DataFrame({
        feature_name: X[feature_name],
        'SHAP': shap_feature_values
    })

    # Compute mean SHAP value per category
    grouped_means = df_plot.groupby(feature_name)['SHAP'].mean()
    categories = grouped_means.index.tolist()
    mean_vals = grouped_means.values

    # Normalize means for color mapping
    norm = mcolors.TwoSlopeNorm(vmin=min(mean_vals), vcenter=0, vmax=max(mean_vals))
    colormap = cm.get_cmap(cmap)

    # Map categories to colors
    colors = [colormap(norm(val)) for val in mean_vals]

    # Plot manually to color each box
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cat in enumerate(categories):
        vals = df_plot[df_plot[feature_name] == cat]['SHAP']
        bp = ax.boxplot(
            vals,
            positions=[i],
            vert=False,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[i], linewidth=1.5),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor='black')
        )

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('SHAP value')
    ax.set_title(
        f'SHAP values by {feature_name}',
        loc="left",
        fontsize=14,
        fontweight="bold",
        x=-0.15
    )
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Plot a SHAP interaction heatmap for any two features
# (numeric or categorical).
# ------------------------------------------------------------
# Parameters:
# - interaction_vals: array of shape (n_samples, n_features, n_features)
# - X: dataframe used for explanation (same columns/order as the model)
# - f1, f2: feature names to visualize
# - bins: number of bins for numeric features
# - binning: "cut" (equal-width) or "qcut" (quantile-based)
# - missing_label: label used for missing categorical values
# - agg: aggregation method ("mean" or "median")
# - abs_values: if True, use absolute interaction values
# - center: colormap center (0 is meaningful for signed interactions)
# - figsize: matplotlib figure size
# - cmap: colormap
# - annot: whether to show numeric annotations in the heatmap
# - fmt: numeric format for annotations
# ------------------------------------------------------------

def plot_shap_interaction_heatmap(
    interaction_vals,
    X: pd.DataFrame,
    f1: str,
    f2: str,
    *,
    bins: int = 30,
    binning: str = "cut",          # "cut" or "qcut"
    missing_label: str = "__MISSING__",
    agg: str = "mean",             # "mean" or "median"
    abs_values: bool = False,      # if True, use |interaction|
    center: float | None = 0.0,    # colormap center (0 makes sense for signed values)
    figsize=(10, 8),
    cmap=hcmap,
    annot: bool = False,
    fmt: str = ".2f",
):

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    if binning not in {"cut", "qcut"}:
        raise ValueError("binning must be 'cut' or 'qcut'")

    # Feature indices
    i = X.columns.get_loc(f1)
    j = X.columns.get_loc(f2)

    inter = interaction_vals[:, i, j]
    if abs_values:
        inter = np.abs(inter)

    df = pd.DataFrame({f1: X[f1].values, f2: X[f2].values, "inter": inter})

    # Helper: check if series is numeric
    def is_numeric(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    # Helper: numeric binning
    def bin_numeric(series: pd.Series, bins: int, how: str) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if how == "qcut":
            # qcut may fail if too many duplicates: use duplicates="drop"
            return pd.qcut(s, q=bins, duplicates="drop")
        else:
            return pd.cut(s, bins=bins)

    # Helper: categorical handling without grouping
    def process_categorical(series: pd.Series) -> pd.Series:
        # Replace only missing values with missing_label
        return series.astype("object").where(series.notna(), missing_label)

    s1 = df[f1]
    s2 = df[f2]

    f1_is_num = is_numeric(s1)
    f2_is_num = is_numeric(s2)

    # Build "b1" and "b2"
    if f1_is_num:
        df["b1"] = bin_numeric(s1, bins=bins, how=binning)
    else:
        df["b1"] = process_categorical(s1)

    if f2_is_num:
        df["b2"] = bin_numeric(s2, bins=bins, how=binning)
    else:
        df["b2"] = process_categorical(s2)

    # Drop rows where binning/labeling was not possible
    # (e.g. numeric column entirely NaN)
    df = df.dropna(subset=["b1", "b2", "inter"])

    # Pivot table
    pivot = df.pivot_table(
        values="inter",
        index="b1",
        columns="b2",
        aggfunc=("median" if agg == "median" else "mean")
    )

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=center if (center is not None and not abs_values) else None,
        annot=annot,
        fmt=fmt
    )

    # Title and axes
    t = f"SHAP interaction heatmap: {f1} × {f2} ({agg})"
    if abs_values:
        t += " [abs]"
    plt.title(t, loc="left", fontsize=14, fontweight="bold", x=-0.1)
    plt.xlabel(f2)
    plt.ylabel(f1)
    plt.tight_layout()
    plt.show()


# =============================================================================
# QUANTILE CLASSIFICATION FUNCTIONS
# =============================================================================

# The function below is used to classify the real values of the target variable using the estimated quantiles.
# In this way it is possible, instead of simply considering an estimation right or wrong, use it to find days
# where pollutants where more or less elevated than what should be expected.
def classify_quartiles(low, q1, q2, q3, high, lt):
    if lt < low:
        return 'low outlier'
    elif lt >= low and lt < q1:
        return 'low'
    elif lt >= q1 and lt < q2:
        return 'low to medium'
    elif lt >= q2 and lt < q3:
        return 'medium to high'
    elif lt >= q3 and lt < high:
        return 'high'
    elif lt >= high:
        return 'high outlier'
    else:
        return ''


# The other function below is used to decide if the real values of the target feature fall inside the estimated IQR
# or otherwise what is the closest quartile. Afterwards, the value returned will be used to classify it with the
# danger levels proposed by the EU AQI, and comparing it to the real one.
def choose_quartile_ref(real, q1, q2, q3):
    if real >= q1 and real <= q3:
        return real
    elif real < q1:
        return q1
    elif real > q3:
        return q3
    else:
        return -1

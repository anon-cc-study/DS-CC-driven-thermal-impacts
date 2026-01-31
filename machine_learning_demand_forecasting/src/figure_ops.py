import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import numpy as np

def generate_metrics_table(loaded_ml_results, metrics):
    """
    Generates a table displaying specified metrics for each key in loaded_ml_results.
    
    Parameters:
    - loaded_ml_results (dict): Dictionary containing model results with metric values.
    - metrics (list): List of metric names to include in the table.
    
    Returns:
    - DataFrame: A Pandas DataFrame with keys as rows and metrics as columns.
    """
    data = []
    index_labels = []
    
    for key, results in loaded_ml_results.items():
        row = [results.get(metric, None) for metric in metrics]
        data.append(row)
        # index_labels.append(" ".join(map(str, key)))  # Create readable key labels
        index_labels.append(f"{key[1]} {key[2]} {key[4]}")  # Create labels for each scenario

    
    df = pd.DataFrame(data, columns=metrics, index=index_labels)
    return df

def plot_metric_comparisons(loaded_ml_results, font_size, metric_pairs):
    """
    Creates subplots of scatter plots comparing specified metric pairs with labeled data points.
    
    Parameters:
    - loaded_ml_results (dict): Dictionary containing model results with metric values.
    - font_size (int): Font size for subplot titles.
    - metric_pairs (dict): Dictionary where keys are subplot titles and values are tuples of two metric names to compare.
    """
    num_plots = len(metric_pairs)
    fig, axes = plt.subplots(
        nrows=int(np.ceil(num_plots / 2)), ncols=2, figsize=(12, 5 * np.ceil(num_plots / 2))
    )
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for idx, (title, (metric_x, metric_y)) in enumerate(metric_pairs.items()):
        ax = axes[idx]
        
        x_values = []
        y_values = []
        labels = []
        
        for key, results in loaded_ml_results.items():
            if metric_x in results and metric_y in results:
                x_values.append(results[metric_x])
                y_values.append(results[metric_y])
                # labels.append(" ".join(map(str, key)))  # Create labels by concatenating key elements
                labels.append(f"{key[1]} {key[2]} {key[4]}")  # Create labels for each scenario
      
        ax.scatter(x_values, y_values, alpha=0.7)
        
        # Add labels to each point
        for i, label in enumerate(labels):
            ax.annotate(label, (x_values[i], y_values[i]), fontsize=8, alpha=0.75)
        
        ax.set_xlabel(metric_x, fontsize=font_size)
        ax.set_ylabel(metric_y, fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
    
    # Hide any unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()


def concatenate_selected(tuple_values, key_elements_for_title):
    """
    Concatenates selected elements from a tuple based on a boolean selection array.
    
    Parameters:
    - tuple_values (tuple): The input tuple containing strings.
    - selection_array (list of bool): A list indicating which elements to include (True = include, False = exclude).
    
    Returns:
    - str: A concatenated string of selected elements.
    """
    if len(tuple_values) != len(key_elements_for_title):
        raise ValueError("Tuple and selection array must have the same length")
    
    selected_values = [val for val, use in zip(tuple_values, key_elements_for_title) if use]
    return " ".join(selected_values)  # Using underscore as a separator

def plot_predictions_vs_actuals(loaded_ml_results, key_elements_for_title, font_size, config):
    num_plots = len(loaded_ml_results)
    fig, axes = plt.subplots(
        nrows=int(np.ceil(num_plots / 2)), ncols=2, figsize=(12, 5 * np.ceil(num_plots / 2))
    )
    axes = axes.flatten() if num_plots > 1 else [axes]

    for idx, (key, results) in enumerate(loaded_ml_results.items()):
        y_test = results["y_test"]
        y_test_pred = results["y_test_pred"]
        r2 = results["r2"]
        nrmse = results["nrmse"]
        mape = results["mape"]
        peak_load_error = results["peak_load_error"]
        
        ax = axes[idx]
        ax.scatter(y_test, y_test_pred, alpha=0.5, label="Predicted vs Actual")
        
        # Plot y=x reference line
        min_val = min(min(y_test), min(y_test_pred))
        max_val = max(max(y_test), max(y_test_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
        
        # Create title using the key
        key_for_title = concatenate_selected(key, key_elements_for_title)
        
        ax.set_xlabel("Actual Load", fontsize=font_size)
        ax.set_ylabel("Predicted Load", fontsize=font_size)
        ax.set_title(f"{key_for_title}", fontsize=font_size)
        
        # Add legend for scatter and y=x line
        ax.legend()
        
        # Display metrics as text inside the plot
        metrics_text = (
            f"R²: {r2:.2f}\n"
            f"NRMSE: {nrmse:.2f}%\n"
            f"MAPE: {mape:.2f}%\n"
            f"Peak Load Error: {peak_load_error:.2f}%"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=font_size,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Hide any unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])      
    
    fig.suptitle('Output: {Y} \n Model: {M} \n months: {S}-{E}'.format(Y=config['Y_column'], M=config['prediction_model'], S=config['start_month'], E=config['end_month']), fontsize=16)   
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to leave space for suptitle
    plt.show()

def plot_heating_and_cooling_share(dict_df, cooling_sum_col, heating_col, total_col, metric='max'):
    """
    Plots bar charts for heating and cooling shares out of total site electricity.
    
    Parameters:
        dict_df (dict): Dictionary containing dataframes for different keys.
        metric (str): 'max' for maximum values, 'avg' for average values.
    """
    keys = []
    heating_percentages = []
    cooling_percentages = []
    
    for key, df in dict_df.items():
        if all(col in df.columns for col in [cooling_sum_col, heating_col, total_col]):
            if metric == 'max':
                max_heating_idx = df[heating_col].idxmax()
                max_heating = df.loc[max_heating_idx, heating_col]
                max_total_heating = df.loc[max_heating_idx, total_col]
                heating_percentage = (max_heating / max_total_heating) * 100
                
                max_cooling_idx = df[cooling_sum_col].idxmax()
                max_cooling = df.loc[max_cooling_idx, cooling_sum_col]
                max_total_cooling = df.loc[max_cooling_idx, total_col]
                cooling_percentage = (max_cooling / max_total_cooling) * 100
            
            elif metric == 'avg':
                avg_heating = df[heating_col].mean()
                avg_total_heating = df[total_col].mean()
                heating_percentage = (avg_heating / avg_total_heating) * 100
                
                avg_cooling = df[cooling_sum_col].mean()
                avg_total_cooling = df[total_col].mean()
                cooling_percentage = (avg_cooling / avg_total_cooling) * 100
            
            else:
                raise ValueError("Invalid metric. Use 'max' or 'avg'.")
            
            keys.append(str(key))
            heating_percentages.append(heating_percentage)
            cooling_percentages.append(cooling_percentage)
    
    # Plot Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.4
    x = range(len(keys))
    
    ax.bar(x, heating_percentages, width=bar_width, label='Heating %', color='#CE2D4F')
    ax.bar([i + bar_width for i in x], cooling_percentages, width=bar_width, label='Cooling %', color='#4056F4')
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax.set_title(f'{metric.title()} Heating and Cooling Share out of Total Site Electricity')
    ax.legend()
    
    plt.show()


def plot_cooling_components_shares(dict_df, cooling_components, total_col, metric='max'):
    """
    Plots stacked bar charts for heating and cooling shares out of total site electricity.
    
    Parameters:
        dict_df (dict): Dictionary containing dataframes for different keys.
        metric (str): 'max' for maximum values, 'avg' for average values.
    """   
    keys = []
    cooling_sub_percentages = {col: [] for col in cooling_components}
    
    for key, df in dict_df.items():
        if all(col in df.columns for col in [total_col] + cooling_components):
            if metric == 'max':
                max_cooling_idx = df[cooling_components].sum(axis=1).idxmax()
                cooling_sub_values = {col: df.loc[max_cooling_idx, col] for col in cooling_components}
                total_value = df.loc[max_cooling_idx, total_col]
            elif metric == 'avg':
                cooling_sub_values = {col: df[col].mean() for col in cooling_components}
                total_value = df[total_col].mean()
            else:
                raise ValueError("Invalid metric. Use 'max' or 'avg'.")
            
            for col in cooling_components:
                cooling_sub_percentages[col].append((cooling_sub_values[col] / total_value) * 100)
            
            keys.append(str(key))
    
    # Plot Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.5
    x = range(len(keys))
    
    # Stack Cooling Subcategories
    bottom = [0] * len(keys)
    for col, color in zip(cooling_components, ['#4056F4', '#6A89CC', '#85C1E9']):
        ax.bar(x, cooling_sub_percentages[col], width=bar_width, bottom=bottom, label=f'{col} ({metric.title()})', color=color)
        bottom = [b + p for b, p in zip(bottom, cooling_sub_percentages[col])]
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax.set_title(f'{metric.title()} Cooling Components Share out of {metric.title()} Total Site Electricity (Stacked)')
    ax.legend()
    
    plt.show()

def plot_correlation_boxplots(input_data_dict, city_regions_dict, target_column, columns_to_compare, cor_threshold, show_plot):
    correlation_data = []
    
    for (year, city, region, _, building_type), df in input_data_dict.items():
        if city in city_regions_dict and region in city_regions_dict[city]:
            # Filter dataframe to relevant columns
            selected_columns = [target_column] + columns_to_compare
            df_selected = df[selected_columns].dropna()
            
            if df_selected.empty:
                continue
            
            city_region = f"{city}-{region}"
            
            # Compute Pearson and Spearman correlations
            pearson_corr = df_selected.corr(method='pearson')[target_column].drop(target_column)
            spearman_corr = df_selected.corr(method='spearman')[target_column].drop(target_column)
            
            # Store correlation results
            for label in columns_to_compare:
                if label in pearson_corr and label in spearman_corr:
                    correlation_data.append({
                        'City-Region': city_region,
                        'Building Type': building_type,
                        'Correlation Type': 'Pearson',
                        'Columns to compare': label,
                        'Correlation Value': pearson_corr[label]
                    })
                    correlation_data.append({
                        'City-Region': city_region,
                        'Building Type': building_type,
                        'Correlation Type': 'Spearman',
                        'Columns to compare': label,
                        'Correlation Value': spearman_corr[label]
                    })
    
    # Convert collected data to DataFrame
    correlation_df = pd.DataFrame(correlation_data)

    # Plot box plots for each combination of correlation type and building type
    if show_plot == True:
        plt.figure(figsize=(16, 10))
        for i, (corr_type, bldg_type) in enumerate([
            ('Pearson', 'res'), ('Pearson', 'com'),
            ('Spearman', 'res'), ('Spearman', 'com')
        ]):
            plt.subplot(2, 2, i+1)
            subset_df = correlation_df[(correlation_df['Correlation Type'] == corr_type) & 
                                       (correlation_df['Building Type'] == bldg_type)]
            sns.boxplot(x='Columns to compare', y='Correlation Value', hue='City-Region', data=subset_df)
            plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
            plt.axhline(cor_threshold, color='red', linestyle='dashed', linewidth=1.2)
            plt.axhline(-cor_threshold, color='red', linestyle='dashed', linewidth=1.2)
            plt.title(f'{corr_type} Correlation - {bldg_type.upper()}')
            plt.xticks(rotation=90)
            # plt.legend(loc='best', fontsize='small')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')


        plt.tight_layout()
        plt.show()
    
    return correlation_df

#  Box Plot function per region with flexible categorical column filtering
def regions_plot_box(dict_data, city_regions_to_run, column_name_to_plot, column_values_to_compare, column_name_to_compare, smart_ds_years, aggregation_level, building_type, title, sns_custom_palette, separate_plots_per_city=True):
    df_list = []
    for city, regions in city_regions_to_run.items():
        for region in regions:
            for year in smart_ds_years:
                key = (str(year), city, region, aggregation_level, building_type)  # Ensure year is a string
                # Check if the key exists before accessing
                if key in dict_data:
                    temp_df = dict_data[key].copy()

                    # Ensure column_name_to_compare exists in DataFrame
                    if column_name_to_compare not in temp_df.columns:
                        print(f"Skipping {key}: Column '{column_name_to_compare}' not found in data.")
                        continue

                    # Filter only the relevant values in column_name_to_compare
                    temp_df = temp_df[temp_df[column_name_to_compare].isin(column_values_to_compare)]

                    if temp_df.empty:
                        continue  # Skip empty DataFrames

                    temp_df["Region"] = f"{region}\n({city})"  # Format region with city name
                    temp_df["City"] = city      # Add city column for combined plots
                    df_list.append(temp_df)

    if not df_list:
        print("No data available for plotting.")
        return

    # Concatenate all collected DataFrames
    df = pd.concat(df_list, ignore_index=True)

    # Convert the selected column to a categorical variable for consistent ordering
    df[column_name_to_compare] = pd.Categorical(df[column_name_to_compare], categories=column_values_to_compare)

    if separate_plots_per_city:
        # Generate a separate plot for each city
        for city in city_regions_to_run.keys():
            city_df = df[df["City"] == city]  # Filter data for the specific city

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=city_df, x="Region", y=column_name_to_plot, hue=column_name_to_compare, palette=sns_custom_palette)
            plt.title(f"{title} - {city}")
            plt.xlabel("Region (City)")
            plt.ylabel(column_name_to_plot)
            plt.legend(title=column_name_to_compare)
            plt.xticks(rotation=15)  # Rotate x-axis labels for readability
            plt.show()
    
    else:
        # Combine all cities into one plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Region", y=column_name_to_plot, hue=column_name_to_compare, palette=sns_custom_palette)
        plt.title(f"{title}")
        plt.xlabel("Region (City)")
        plt.ylabel(column_name_to_plot)
        plt.legend(title=column_name_to_compare)
        plt.xticks(rotation=15)  # Rotate x-axis labels for readability
        plt.show()

# Scatter Plot function per region with flexible categorical column filtering
def regions_plot_scatter(dict_data, city_regions_to_run, x_column_name_to_plot, y_column_name_to_plot, column_values_to_compare, column_name_to_compare, smart_ds_years, aggregation_level, building_type, title, sns_custom_palette, separate_plots_per_city, separate_plots_per_region, num_columns=3):
    df_list = []
    for city, regions in city_regions_to_run.items():
        for region in regions:
            for year in smart_ds_years:
                key = (str(year), city, region, aggregation_level, building_type)  # Ensure year is a string
                if key in dict_data:
                    temp_df = dict_data[key].copy()

                    if column_name_to_compare not in temp_df.columns:
                        print(f"Skipping {key}: Column '{column_name_to_compare}' not found in data.")
                        continue

                    temp_df = temp_df[temp_df[column_name_to_compare].isin(column_values_to_compare)]
                    if temp_df.empty:
                        continue  # Skip empty DataFrames

                    temp_df["Region"] = f"{region}\n({city})"
                    temp_df["City"] = city
                    df_list.append(temp_df)

    if not df_list:
        print("No data available for plotting.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df[column_name_to_compare] = pd.Categorical(df[column_name_to_compare], categories=column_values_to_compare)

    markers = {region: marker for region, marker in zip(df["Region"].unique(), ["o", "s", "D", "^", "v", "<", ">", "p", "*"])}

    if separate_plots_per_city:
        for city in city_regions_to_run.keys():
            city_df = df[df["City"] == city]
            plt.figure(figsize=(10, 6))

            for region in city_df["Region"].unique():
                region_df = city_df[city_df["Region"] == region]
                sns.scatterplot(data=region_df, x=x_column_name_to_plot, y=y_column_name_to_plot, hue=column_name_to_compare, palette=sns_custom_palette, marker=markers[region])

            plt.title(f"{title} - {city}")
            plt.xlabel(x_column_name_to_plot)
            plt.ylabel(y_column_name_to_plot)
            plt.legend(title=column_name_to_compare)
            plt.xticks(rotation=15)
            plt.show()

    elif separate_plots_per_region:
        unique_regions = df["Region"].unique()
        num_rows = int(np.ceil(len(unique_regions) / num_columns))  # Calculate required rows
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows), constrained_layout=True)
        axes = np.array(axes).reshape(-1)  # Flatten in case of a single row

        for i, region in enumerate(unique_regions):
            region_df = df[df["Region"] == region]
            ax = axes[i]
            sns.scatterplot(data=region_df, x=x_column_name_to_plot, y=y_column_name_to_plot, hue=column_name_to_compare, palette=sns_custom_palette, marker=markers[region], ax=ax)
            ax.set_title(f"{title} - {region}")
            ax.set_xlabel(x_column_name_to_plot)
            ax.set_ylabel(y_column_name_to_plot)
            ax.tick_params(axis='x', rotation=15)
            ax.legend(title=column_name_to_compare)

        # Hide empty subplots if there are more subplots than needed
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.show()

    else:
        plt.figure(figsize=(12, 6))
        for region in df["Region"].unique():
            region_df = df[df["Region"] == region]
            sns.scatterplot(data=region_df, x=x_column_name_to_plot, y=y_column_name_to_plot, hue=column_name_to_compare, palette=sns_custom_palette, marker=markers[region])
        
        plt.title(f"{title}")
        plt.xlabel(x_column_name_to_plot)
        plt.ylabel(y_column_name_to_plot)
        plt.legend(title=column_name_to_compare)
        plt.xticks(rotation=15)
        plt.show()


        
        
# Box Plot function for Pearson Correlation Coefficient per region
def regions_plot_pearson_box(dict_data, city_regions_to_run, x_column_name_to_plot, y_column_name_to_plot, column_values_to_compare, column_name_to_compare, smart_ds_years, aggregation_level, building_type, title, sns_custom_palette, separate_plots_per_city=True, separate_plots_per_region=False):
    df_list = []
    correlation_data = []
    
    for city, regions in city_regions_to_run.items():
        for region in regions:
            for year in smart_ds_years:
                key = (str(year), city, region, aggregation_level, building_type)  # Ensure year is a string
                # Check if the key exists before accessing
                if key in dict_data:
                    temp_df = dict_data[key].copy()

                    # Ensure column_name_to_compare exists in DataFrame
                    if column_name_to_compare not in temp_df.columns:
                        print(f"Skipping {key}: Column '{column_name_to_compare}' not found in data.")
                        continue

                    # Filter only the relevant values in column_name_to_compare
                    temp_df = temp_df[temp_df[column_name_to_compare].isin(column_values_to_compare)]

                    if temp_df.empty:
                        continue  # Skip empty DataFrames

                    temp_df["Region"] = f"{region}\n({city})"  # Format region with city name
                    temp_df["City"] = city      # Add city column for combined plots
                    
                    # Compute Pearson Correlation Coefficient (R)
                    pearson_r, _ = stats.pearsonr(temp_df[x_column_name_to_plot], temp_df[y_column_name_to_plot])
                    temp_df["Pearson_R"] = pearson_r
                    
                    correlation_data.append({
                        "City": city,
                        "Region": f"{region}\n({city})",
                        "Pearson_R": pearson_r
                    })
    
    if not correlation_data:
        print("No data available for plotting.")
        return
    
    # Convert correlation data to DataFrame
    correlation_df = pd.DataFrame(correlation_data)
    
    if separate_plots_per_city:
        # Generate a separate plot for each city
        for city in city_regions_to_run.keys():
            city_df = correlation_df[correlation_df["City"] == city]  # Filter data for the specific city
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=city_df, x="Region", y="Pearson_R", palette=sns_custom_palette)
            plt.axhline(y=0, color='black', linestyle='--')  # Reference line at R=0
            plt.title(f"{title} - {city}")
            plt.xlabel("Region (City)")
            plt.ylabel("Pearson Correlation Coefficient (R)")
            plt.xticks(rotation=15)  # Rotate x-axis labels for readability
            plt.show()
    
    elif separate_plots_per_region:
        # Generate a separate plot for each region
        for region in correlation_df["Region"].unique():
            region_df = correlation_df[correlation_df["Region"] == region]
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=region_df, x="Region", y="Pearson_R", palette=sns_custom_palette)
            plt.axhline(y=0, color='black', linestyle='--')  # Reference line at R=0
            plt.title(f"{title} - {region}")
            plt.xlabel("Region")
            plt.ylabel("Pearson Correlation Coefficient (R)")
            plt.xticks(rotation=15)  # Rotate x-axis labels for readability
            plt.show()
    
    else:
        # Combine all cities into one plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=correlation_df, x="Region", y="Pearson_R", palette=sns_custom_palette)
        plt.axhline(y=0, color='black', linestyle='--')  # Reference line at R=0
        plt.title(f"{title}")
        plt.xlabel("Region (City)")
        plt.ylabel("Pearson Correlation Coefficient (R)")
        plt.xticks(rotation=15)  # Rotate x-axis labels for readability
        plt.show()
    return correlation_df


def plot_correlation_bars(df, target_column, columns_to_compare, scenario_name):
    # Filter dataframe to relevant columns
    selected_columns = [target_column] + columns_to_compare
    df_selected = df[selected_columns].dropna()
    
    # Compute Pearson correlations
    pearson_corr = df_selected.corr(method='pearson')[target_column].drop(target_column)
    
    # Compute Spearman correlations
    spearman_corr = df_selected.corr(method='spearman')[target_column].drop(target_column)
    
    # Plot Pearson correlation bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=pearson_corr.index, y=pearson_corr.values)
    plt.xticks(rotation=90)
    plt.ylabel('Pearson Correlation')
    plt.title(f'Pearson Correlation with {target_column} \n in {scenario_name}')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(0.5, color='red', linestyle='dashed', linewidth=1.2)
    plt.axhline(-0.5, color='red', linestyle='dashed', linewidth=1.2)
    plt.show()
    
    # Plot Spearman correlation bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=spearman_corr.index, y=spearman_corr.values)
    plt.xticks(rotation=90)
    plt.ylabel('Spearman Correlation')
    plt.title(f'Spearman Correlation with {target_column} \n in {scenario_name}')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(0.5, color='red', linestyle='dashed', linewidth=1.2)
    plt.axhline(-0.5, color='red', linestyle='dashed', linewidth=1.2)
    plt.show()
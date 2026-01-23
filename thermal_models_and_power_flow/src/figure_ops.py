import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

red_palette = ["#F2A541", "#CE1A1A", "#670606"]  # Red, Orange, Yellow
sns.set_palette(red_palette)

#  Box Plot function per region with flexible categorical column filtering
def regions_plot_box(transformers_data, city_regions_to_run, column_filter_array, column_name, smart_ds_years, title, separate_plots_per_city=True):
    df_list = []
    
    for city, regions in city_regions_to_run.items():
        for region in regions:
            for year in smart_ds_years:
                key = (str(year), city, region)  # Ensure year is a string
                
                # Check if the key exists before accessing
                if key in transformers_data:
                    temp_df = transformers_data[key].copy()

                    # Ensure column_name exists in DataFrame
                    if column_name not in temp_df.columns:
                        print(f"Skipping {key}: Column '{column_name}' not found in data.")
                        continue
                    
                    # Filter only the relevant values in column_name
                    temp_df = temp_df[temp_df[column_name].isin(column_filter_array)]

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
    df[column_name] = pd.Categorical(df[column_name], categories=column_filter_array)

    if separate_plots_per_city:
        # Generate a separate plot for each city
        for city in city_regions_to_run.keys():
            city_df = df[df["City"] == city]  # Filter data for the specific city

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=city_df, x="Region", y="Loading [%]", hue=column_name, palette=red_palette)
            plt.title(f"{title} - {city}")
            plt.xlabel("Region (City)")
            plt.ylabel("Loading [%]")
            plt.legend(title=column_name)
            plt.xticks(rotation=15)  # Rotate x-axis labels for readability
            plt.show()
    
    else:
        # Combine all cities into one plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Region", y="Loading [%]", hue=column_name, palette=red_palette)
        plt.title(f"{title}")
        plt.xlabel("Region (City)")
        plt.ylabel("Loading [%]")
        plt.legend(title=column_name)
        plt.xticks(rotation=15)  # Rotate x-axis labels for readability
        plt.show()

# Colored histogram of transformer loading [%]
def loading_histogram(df_to_plot, title, ylabel, xlabel, font_size=14, n_bins=10):
    # Define the number of bins for the histogram
    max_load = df_to_plot['Loading [%]'].max()
    num_bins = int(max_load / n_bins)

    # plot the histogram
    fig, ax = plt.subplots()

    # Creating the histogram
    n, bins, patches = ax.hist(df_to_plot['Loading [%]'], bins=num_bins, edgecolor='black')

    # Step 3: Change the color of bins based on their values
    for i in range(len(bins) - 1):
        if 70 <= bins[i] < 80:
            patches[i].set_facecolor('#F3CA40')
        elif 80 <= bins[i] < 90:
            patches[i].set_facecolor('#F2A541')
        elif 90 <= bins[i] < 100:
            patches[i].set_facecolor('#CE1A1A')
        elif bins[i] >= 100:
            patches[i].set_facecolor('#670606')

    # Count the number of lines with loading above 100%
    transformers_above_100 = sum(df_to_plot['Loading [%]'] > 100)
    transformers_above_100_per = 100*(transformers_above_100/len(df_to_plot))
    
    # Adding labels and title
    ax.set_title(f'{title} \nEquipment with loading over 100%: {transformers_above_100} ({transformers_above_100_per:.2f}%)', fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    # ax.set_title(f'Ambient Temp: {ambient_temp} C', fontsize=font_size)  # Need to pass metadata (e.g., with json file) for this title
    plt.grid(True)

    # Show the plot
    plt.show()
    
def loading_histogram_subplot(df_to_plot_1, title_1, df_to_plot_2, title_2, ylabel, xlabel, font_size=14, n_bins=10):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for i, (df, title, axis) in enumerate(zip(
        [df_to_plot_1, df_to_plot_2],
        [title_1, title_2],
        ax
    )):
        max_load = df['Loading [%]'].max()
        num_bins = int(max_load / n_bins)

        n, bins, patches = axis.hist(df['Loading [%]'], bins=num_bins, edgecolor='black')

        # Color the bars
        for j in range(len(bins) - 1):
            if 70 <= bins[j] < 80:
                patches[j].set_facecolor('#F3CA40')
            elif 80 <= bins[j] < 90:
                patches[j].set_facecolor('#F2A541')
            elif 90 <= bins[j] < 100:
                patches[j].set_facecolor('#CE1A1A')
            elif bins[j] >= 100:
                patches[j].set_facecolor('#670606')

        overloaded = sum(df['Loading [%]'] > 100)
        overloaded_per = 100*(overloaded/len(df))
        axis.set_title(f'{title} \nTransformers with loading over 100%: {overloaded} ({overloaded_per:.2f}%)', fontsize=font_size)
        axis.set_xlabel(xlabel, fontsize=font_size)
        axis.set_ylabel(ylabel, fontsize=font_size)
        axis.grid(True)

    plt.tight_layout()
    plt.show()
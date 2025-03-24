import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

sns.set_context("poster", font_scale=1)
sns.set_style("ticks")


def extract_concentrations(ink_type):
    ink_type = ink_type.replace("-", "_")
    parts = ink_type.split("_")
    alg = float(parts[0].replace("ALG", ""))
    if len(parts) > 1 and "HA" in parts[1]:
        ha = float(parts[1].replace("HA", ""))
    else:
        ha = 0.0
    return pd.Series({"ALG %(w/v)": alg, "HA %(w/v)": ha})


def load_and_preprocess_data(base_path="csv_data_files/"):
    # Read data files
    rheology_viscosity = pd.read_csv(
        f"{base_path}ALG_HA_cross_power_law_svr_predictions.csv"
    )
    rheology_modulus = pd.read_csv(f"{base_path}ALG_HA_GG_at_1hz_svr_predictions.csv")
    ink_data_index = pd.read_csv(f"{base_path}ALG_HA_index.csv")
    printability_results = pd.read_csv(f"{base_path}lattice_analysis_results.csv")
    pc12_data = pd.read_csv(f"{base_path}PC-12_avg_data.csv")

    # Process rheology viscosity data
    rheology_viscosity[["ALG %(w/v)", "HA %(w/v)"]] = rheology_viscosity[
        "ink_type"
    ].apply(extract_concentrations)
    rheology_viscosity = rheology_viscosity.rename(
        columns={
            "eta_0": r"$\eta_0$",
            "m": r"$m$",
            "n": r"$n$",
        }
    )

    # Process rheology modulus data
    rheology_modulus[["ALG %(w/v)", "HA %(w/v)"]] = rheology_modulus["ink_type"].apply(
        extract_concentrations
    )
    rheology_modulus = rheology_modulus.rename(
        columns={
            "G'": r"$G'$",
            "G''": r"$G''$",
            "tan_delta": r"$\tan(\delta)$",
            "eta_star": r"$|\eta^*|$",
        }
    )

    # Process printability data
    printability_results["image_id"] = (
        printability_results["Image"].str.extract(r"IMG_(\d+)").astype(int)
    )
    printability_data = pd.merge(
        printability_results[
            ["image_id", "Average_HD", "Average_Pr", "Construct_Area_mm2"]
        ],
        ink_data_index[["image_id", "ink_type"]],
        on="image_id",
    )
    printability_data[["ALG %(w/v)", "HA %(w/v)"]] = printability_data[
        "ink_type"
    ].apply(extract_concentrations)
    printability_data = printability_data.rename(
        columns={
            "Average_HD": "HD",
            "Average_Pr": "Pr",
            "Construct_Area_mm2": "Area",
        }
    )

    # Process PC12 data
    pc12_filtered = pc12_data[pc12_data["flow_rate_uL_per_s"] == 2.0]
    pc12_filtered = pc12_filtered[pc12_filtered["alg_concentration"] != 4.25]
    pc12_filtered = pc12_filtered.rename(
        columns={
            "alg_concentration": "ALG %(w/v)",
            "ha_concentration": "HA %(w/v)",
            "relative_cell_viability": "PC12 Viability",
        }
    )

    return rheology_viscosity, rheology_modulus, printability_data, pc12_filtered


def merge_datasets(rheology_viscosity, rheology_modulus, printability_data, pc12_data):
    # Merge rheology data first
    merged_df = pd.merge(
        rheology_viscosity.drop(columns=["ink_type"]),
        rheology_modulus.drop(columns=["ink_type"]),
        on=["ALG %(w/v)", "HA %(w/v)"],
        how="outer",
    )

    # Merge with printability data
    merged_df = pd.merge(
        merged_df,
        printability_data.drop(columns=["image_id", "ink_type"]),
        on=["ALG %(w/v)", "HA %(w/v)"],
        how="outer",
    )

    # Handle duplicate measurements in PC12 data by taking the mean
    pc12_agg = (
        pc12_data.groupby(["ALG %(w/v)", "HA %(w/v)"])["PC12 Viability"]
        .agg(["mean", "std"])
        .reset_index()
    )
    pc12_agg = pc12_agg.rename(columns={"mean": "PC12 Viability"})

    # Merge with aggregated PC12 data
    merged_df = pd.merge(
        merged_df,
        pc12_agg[["ALG %(w/v)", "HA %(w/v)", "PC12 Viability"]],
        on=["ALG %(w/v)", "HA %(w/v)"],
        how="outer",
    )

    return merged_df


def impute_and_correlate(df, output_dir="corr_matrix"):
    # Prepare data for imputation
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    X = df[numerical_columns].copy()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Perform KNN imputation with increased neighbors for robustness
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_imputed = imputer.fit_transform(X_scaled)

    # Transform back to original scale
    X_imputed = scaler.inverse_transform(X_imputed).round(5)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns).round(5)

    # Calculate correlation matrix
    correlation_matrix = X_imputed.corr()
    correlation_matrix = correlation_matrix.map(lambda x: 0 if -0.01 < x < 0.01 else x)

    # Reorder columns
    column_order = ["ALG %(w/v)", "HA %(w/v)"] + [
        col
        for col in correlation_matrix.columns
        if col not in ["ALG %(w/v)", "HA %(w/v)"]
    ]
    correlation_matrix = correlation_matrix.reindex(
        index=column_order, columns=column_order
    )

    return X_imputed, correlation_matrix


def plot_correlation_matrix(correlation_matrix, output_dir="corr_matrix"):
    plt.figure(figsize=(13, 13))

    hm = sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        fmt=".2f",
        cbar=True,
        cbar_kws={
            "label": "Correlation Coefficient",
            "shrink": 0.7375,
            "aspect": 30,
            "pad": 0.03,
        },
        linewidths=2,
        linecolor="black",
    )

    # Remove the color bar
    hm.collections[0].colorbar.remove()

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    plt.tight_layout()

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.savefig(
        f"{output_dir}/correlation_matrix_imputed.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.5,
    )
    plt.close()


def main():
    # Load and preprocess all datasets
    rheology_viscosity, rheology_modulus, printability_data, pc12_data = (
        load_and_preprocess_data()
    )

    # Merge all datasets
    master_df = merge_datasets(
        rheology_viscosity, rheology_modulus, printability_data, pc12_data
    )

    # Save unimputed data
    master_df.to_csv("corr_matrix/master_df_unimputed.csv", index=False)

    # Print summary of missing values before imputation
    print("\nMissing values before imputation:")
    missing_before = master_df.isnull().sum()
    print(missing_before[missing_before > 0])

    # Perform imputation and correlation analysis
    X_imputed, correlation_matrix = impute_and_correlate(master_df)

    # Create and save visualization
    plot_correlation_matrix(correlation_matrix)

    # Save processed data
    X_imputed.to_csv("corr_matrix/master_df_imputed.csv", index=False)
    correlation_matrix.to_csv("corr_matrix/correlation_matrix_imputed.csv")

    print("\nNo missing values remain after imputation")

    # Print summary statistics for verification
    print("\nSummary statistics after imputation:")
    print(X_imputed.describe())


if __name__ == "__main__":
    main()

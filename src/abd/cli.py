# Main
import ee
import pandas as pd
import numpy as np
import time
import warnings
import os
import traceback
import gc
from pathlib import Path
import click

# Sub
from datetime import datetime as dt
from datetime import timedelta

from ee import EEException

# Extras src
from abd.abd import Abd

# ignore warnings
warnings.filterwarnings("ignore")

this_directory = Path(__file__).parent
__version__ = ""
exec((this_directory / "version.py").read_text(encoding="utf-8"))


@click.command()
@click.option(
    "--lat_lon",
    default="-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091",
    help="Two diagnal points (Latitude 1, Longitude 1, Latitude 2, Longitude 2) "
    "of the study area",
)
@click.option(
    "--dates",
    default="2019-06-03,2019-07-01,2019-08-19,2019-09-24",
    help="Comma-separated date to be applied the algorithm",
)
@click.option("--name", default="erie", help="Place where to save generated files")
@click.option(
    "--days_threshold",
    default=180,
    help="Days threshold used to build the timeseries and training set: 90, 180 ou 365",
)
@click.option(
    "--model",
    default=None,
    help="Select the desired model: ocsvm, rf, if or None for all",
)
@click.option(
    "--sensor",
    default="modis",
    help="Define the selected sensor where images will be downloaded from: "
    "landsat, sentinel, modis",
)
@click.option("--save_collection", help="Save collection images (tiff and png)")
@click.option("--save_train", help="Enable saving the training dataset (csv)")
@click.option("--force_cache", help="Force cache resetting to prevent image errors")
@click.option(
    "--attributes",
    default="ndvi,fai",
    help="Define the attributes used in the modelling process",
)
@click.option(
    "--outliers_zscore",
    default=3.0,
    help="Define the Z-Score used in the median outlier removal threshold",
)
@click.option(
    "--attribute_doy",
    help="Define if the doy attribute will be used in the modelling process",
)
@click.option(
    "--roi",
    default=None,
    help="Select the GEE roi version used in the validation process. "
    "E.g. users/pedroananias/roi",
)
@click.option(
    "--cloud_threshold",
    default=None,
    help="Allow CLOUD threshold customization by user",
)
@click.option(
    "--output_folder",
    default=None,
    help="Specify desired results output folder",
)
def detection(
    lat_lon: str,
    dates: str,
    name: str,
    days_threshold: int,
    model: str,
    sensor: str,
    save_collection: bool,
    save_train: bool,
    force_cache: bool,
    attributes: str,
    outliers_zscore: float,
    attribute_doy: bool,
    roi: str,
    cloud_threshold: float,
    output_folder: str
):
    try:

        # Start script time counter
        start_time = time.time()

        # Google Earth Engine API initialization
        try:
            ee.Initialize()
        except (Exception, EEException) as e:
            print(f"Google Earth Engine authentication/initialization error: {e}. "
                  f"Please, manually log in GEE paltform with `earthengine authenticate`. "
                  f"** See README.md file for the complete instructions **")

        # ### Working directory

        # Data path
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            folderRoot = output_folder + "/results"
            if not os.path.exists(folderRoot):
                os.mkdir(folderRoot)
        else:
            folderRoot = os.path.dirname(os.path.realpath(__file__)) + "/../../results"
            if not os.path.exists(folderRoot):
                os.mkdir(folderRoot)

        # Images path
        folderCache = os.path.dirname(os.path.realpath(__file__)) + "/.cache"
        if not os.path.exists(folderCache):
            os.mkdir(folderCache)

        # ### ABD execution

        # check if date range was selected and build time series based on it
        yearly = False
        if "*" in dates:
            yearly = True
            path_df_yearly_occurrence = folderRoot + "/df_yearly_occurrence.csv"
            start = dt.strptime(dates.split("*")[0], "%Y-%m-%d")
            end = dt.strptime(dates.split("*")[1], "%Y-%m-%d")
            dates = [
                (start + timedelta(days=x)).strftime("%Y-%m-%d")
                for x in range(0, (end - start).days + 1)
            ]
            dates = ",".join(dates)

        # go through each parameter
        for date in dates.split(","):

            # folder to save results from algorithm at
            folder = (
                folderRoot
                + "/"
                + dt.now().strftime("%Y%m%d_%H%M%S")
                + "[v="
                + str(__version__)
                + "-"
                + str(name)
                + ",d="
                + str(date)
                + ",t="
                + str(days_threshold)
                + ",m="
                + str(model)
                + ",s="
                + str(sensor)
                + ",attr="
                + str(attributes)
                + "]"
            )
            if not os.path.exists(folder):
                os.mkdir(folder)

            # create ABD algorithm
            algorithm = Abd(
                lat_lon=lat_lon,
                date=date,
                days_threshold=days_threshold,
                model=model,
                sensor=sensor,
                scaler="robust",
                cache_path=folderCache,
                force_cache=force_cache,
                remove_outliers=True,
                reduce_dimensionality=False,
                morph_op=None,
                morph_op_iters=1,
                convolve=False,
                convolve_radius=1,
                outliers_zscore=outliers_zscore,
                attributes=attributes.split(","),
                attribute_doy=attribute_doy,
                cloud_threshold=cloud_threshold,
            )

            # preprocessing
            algorithm.process_timeseries_data()

            # creating train and grid datasets
            algorithm.process_training_data(df=algorithm.df_timeseries)

            # start training process
            algorithm.train()

            # apply detection algorithm
            algorithm.detect()

            # validate using ROI
            if roi:
                algorithm.validate_using_roi(
                    path=roi,
                    rois=[
                        date + "_" + sensor + "_regular",
                        date + "_" + sensor + "_anomaly",
                    ],
                    labels=[0, 1],
                )

            # save geojson
            algorithm.save_geojsons(folder=folder + "/geojson")

            # save results
            algorithm.save_dataset(
                df=algorithm.df_results, path=folder + "/results.csv"
            )

            # save collection images
            if (
                isinstance(algorithm.df_image, pd.DataFrame)
                and not algorithm.df_image.empty
            ):
                algorithm.save_detection_plot(path=folder + "/detection.png")
                algorithm.save_image_png(
                    image=algorithm.image_clip,
                    date=dt.strptime(date, "%Y-%m-%d"),
                    path=folder + "/" + date + ".png",
                )
                algorithm.save_image_tiff(
                    image=algorithm.image_clip,
                    date=dt.strptime(date, "%Y-%m-%d"),
                    path=folder + "/" + date + ".zip",
                    folderName=name,
                )

            # save collection images (tiff and png)
            if save_collection:
                algorithm.save_timeseries_plot(
                    df=algorithm.df_timeseries, path=folder + "/timeseries.png"
                )
                algorithm.save_collection_tiff(folder=folder + "/tiff", folderName=name)
                algorithm.save_collection_png(folder=folder + "/png")

            # save preprocessing results
            if save_train:
                algorithm.save_dataset(
                    df=algorithm.df_timeseries, path=folder + "/timeseries.csv"
                )
                algorithm.save_dataset(
                    df=algorithm.df_train, path=folder + "/df_train.csv"
                )
                algorithm.save_dataset(
                    df=algorithm.df_gridsearch, path=folder + "/df_gridsearch.csv"
                )
                algorithm.save_dataset(
                    df=algorithm.df_image, path=folder + "/df_image.csv"
                )

            # results
            # add results and save it on disk
            if algorithm.df_results is not None and not algorithm.df_results.empty:
                path_df_results = folderRoot + "/results.csv"
                df_results = (
                    pd.read_csv(path_df_results)
                    .drop(["Unnamed: 0"], axis=1, errors="ignore")
                    .append(algorithm.df_results)
                    if os.path.exists(path_df_results)
                    else algorithm.df_results.copy(deep=True)
                )
                df_results.to_csv(r"" + path_df_results)

            # yearly occurrence
            # add results to yearly occurrence
            if (
                yearly
                and isinstance(algorithm.df_image, pd.DataFrame)
                and not algorithm.df_image.empty
            ):
                df_image = algorithm.df_image.astype(str)
                df_yearly_occurrence = (
                    pd.read_csv(path_df_yearly_occurrence)
                    .drop(["Unnamed: 0"], axis=1, errors="ignore")
                    .append(df_image)
                    if os.path.exists(path_df_yearly_occurrence)
                    else df_image.copy(deep=True)
                )
                df_yearly_occurrence.to_csv(r"" + path_df_yearly_occurrence)

                # clear memory
                del df_yearly_occurrence
                gc.collect()

            # clear memory
            del algorithm
            gc.collect()

        # check if it should build yearly occurrences and save it
        if yearly and os.path.exists(path_df_yearly_occurrence):

            # fix/build yearly dataframe
            df_yearly_occurrence = pd.read_csv(path_df_yearly_occurrence).drop(
                [
                    "Unnamed: 0",
                    "pixel",
                    "index",
                    "doy",
                    "ndvi",
                    "fai",
                    "ndvi_median",
                    "fai_median",
                    "ndvi_diff",
                    "fai_diff",
                ],
                axis=1,
                errors="ignore",
            )
            df_yearly_occurrence.rename(columns={"label_0": "occurrence"}, inplace=True)
            df_yearly_occurrence["year"] = pd.to_datetime(
                df_yearly_occurrence["date"], format="%Y-%m-%d", errors="ignore"
            ).dt.year.astype(dtype=np.int64, errors="ignore")
            df_yearly_occurrence["cloud"] = df_yearly_occurrence["cloud"].astype(
                dtype=np.int64, errors="ignore"
            )
            df_yearly_occurrence["not_occurrence"] = 0

            # sum all occurrences, clouds and non_occurrence by year, lat and lon
            df_yearly_occurrence.loc[
                (
                    (df_yearly_occurrence["occurrence"] == 0.0)
                    & (df_yearly_occurrence["cloud"] == 0.0)
                ),
                "not_occurrence",
            ] = 1
            df_yearly_occurrence = (
                df_yearly_occurrence.groupby(["lat", "lon", "year"]).sum().reset_index()
            )

            # add porcentage of occurrence and cloud
            df_yearly_occurrence["pct_occurrence"] = (
                df_yearly_occurrence["occurrence"]
                / (
                    df_yearly_occurrence["occurrence"]
                    + df_yearly_occurrence["not_occurrence"]
                )
            ) * 100
            df_yearly_occurrence["pct_cloud"] = (
                df_yearly_occurrence["cloud"]
                / (
                    df_yearly_occurrence["occurrence"]
                    + df_yearly_occurrence["not_occurrence"]
                    + df_yearly_occurrence["cloud"]
                )
            ) * 100
            df_yearly_occurrence["instants"] = (
                df_yearly_occurrence["occurrence"]
                + df_yearly_occurrence["not_occurrence"]
                + df_yearly_occurrence["cloud"]
            )

            # fix problem with 1 instant pixels for
            # series average bigger than 10 instants
            if df_yearly_occurrence["instants"].mean() > 10:
                df_yearly_occurrence.loc[
                    (df_yearly_occurrence["instants"] == 1), "pct_occurrence"
                ] = np.nan
                df_yearly_occurrence["pct_occurrence"].fillna(
                    method="ffill", inplace=True
                )

            # folder to save results from algorithm at
            folder = (
                folderRoot
                + "/"
                + dt.now().strftime("%Y%m%d_%H%M%S")
                + "[v="
                + str(__version__)
                + "-"
                + str(name)
                + "]"
            )
            if not os.path.exists(folder):
                os.mkdir(folder)

            # create new ABD instance
            algorithm = Abd(
                lat_lon=lat_lon,
                date=dates.split(",")[0],
                days_threshold=days_threshold,
                model=model,
                sensor=sensor,
                scaler="robust",
                cache_path=folderCache,
                force_cache=force_cache,
                remove_outliers=True,
                reduce_dimensionality=False,
                morph_op=None,
                morph_op_iters=1,
                convolve=False,
                convolve_radius=1,
                outliers_zscore=outliers_zscore,
                attributes=attributes.split(","),
                attribute_doy=attribute_doy,
                cloud_threshold=cloud_threshold,
            )

            # save occurrences plot
            algorithm.save_occurrences_plot(
                df=df_yearly_occurrence, path=folder + "/yearly_svm.png"
            )

            # save occurrences geojson
            algorithm.save_occurrences_geojson(
                df=df_yearly_occurrence, path=folder + "/yearly_svm.json"
            )

        # ### Script termination notice
        script_time_all = time.time() - start_time
        debug = (
            "***** Script execution completed successfully (-- %s seconds --) *****"
            % script_time_all
        )
        print()
        print(debug)

    except Exception:

        # ### Script execution error warning

        # Execution
        print()
        print()
        debug = "***** Error on script execution: " + str(traceback.format_exc())
        print(debug)

        # Removes the folder created initially with the result of execution
        script_time_all = time.time() - start_time
        debug = (
            "***** Script execution could not be completed (-- %s seconds --) *****"
            % script_time_all
        )
        print(debug)


if __name__ == '__main__':
    detection()

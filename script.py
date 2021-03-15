#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################################################################
# ### ABD - Anomalous Behaviour Detection
# ### Module responsable for detecting inland waters anomalies based on images from Google Earth Engine API and machine learning
# ### Python 3.7 64Bits is required!
#
# ### Change History
# - Version 21.0-2: 
# - Mean and standard deviation were tested with STD, STD / 2 and STD + (STD / 2) in versions 21.0, 21.1 and 21.2 respectively
#
# - Version 22.0: 
# - Trend was removed using the medium image and subtracting its pixels from the time series images, adding the use of randomizedsearchcv
#
# - Version 22.1: 
# - Creating script for cloud separation in json, correction in the display of maps
#
# - Version 22.2: 
# - Modification to support mosaic of images of the same date with cut (Sentinel)
#
# - Version 23.0: 
# - Merge test between GridSearch and RandomizedSearch
# - Removing the poly kernel (optimization)
# - Removal of the Outlier parameter - set to False, that is, remove
# - Enabled Modis satellite
# - added image error control, if not found, including Modis water mask
# - Corrections in the generation of the recurrences graph
#
# - Version 23.1:
# - trend removal correction
# - Added try except when extracting pixels from images
#
# - Version 23.2:
# - Built-in image caching system for increased performance
#
# - Version 24:
# - Added Random Forest support for RandomizedSearchCV
# - New ROIs were selected to validate the algorithm with in situ data
# - New function for outlier removal
# - Added support for bigger study areas
# - Added support for Deep Learning with TensorFlow AutoEncoder 
#
# - Version 25:
# - Added SABI Index
# - Added Isolation Forest support
#
# - Version 26:
# - Tests with NDVI and FAI only
# - Created yearly outputs
###############################################################################################################################

# ### Version
version = "V26"



# ### Module imports

# Main
import ee
import pandas as pd
import numpy as np
import math
import requests
import time
import warnings
import os
import sys
import argparse
import logging
import traceback
import itertools
import gc

# Sub
from datetime import datetime as dt
from datetime import timedelta

# Extras modules
from modules import misc, gee, abd

# ignore warnings
warnings.filterwarnings('ignore')


# ### Script args parsing

# starting arg parser
parser = argparse.ArgumentParser(description=version)

# create arguments
parser.add_argument('--lat_lon', dest='lat_lon', action='store', default="-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091",
                   help="Two diagnal points (Latitude 1, Longitude 1, Latitude 2, Longitude 2) of the study area")
parser.add_argument('--dates', dest='dates', action='store', default="2019-06-03,2019-07-01,2019-08-19,2019-09-24",
                   help="Comma-separated date to be applied the algorithm")
parser.add_argument('--name', dest='name', action='store', default="erie",
                   help="Place where to save generated files")
parser.add_argument('--days_threshold', dest='days_threshold', action='store', type=int, default=180,
                   help="Days threshold used to build the timeseries and training set: 90, 180 ou 365")
parser.add_argument('--model', dest='model', action='store', default=None,
                   help="Select the desired module: ocsvm, rf, ae, if or None for all")
parser.add_argument('--sensor', dest='sensor', action='store', default="modis",
                   help="Define the selected sensor where images will be downloaded from: landsat, sentinel, modis")
parser.add_argument('--save_collection', dest='save_collection', action='store_true',
                   help="Save collection images (tiff and png)")
parser.add_argument('--save_train', dest='save_train', action='store_true',
                   help="Enable saving the training dataset (csv)")
parser.add_argument('--force_cache', dest='force_cache', action='store_true',
                   help="Force cache reseting to prevent image errors")
parser.add_argument('--attributes', dest='attributes', action='store', default="ndvi,fai",
                   help="Define the attributes used in the modelling process")
parser.add_argument('--outliers_zscore', dest='outliers_zscore', action='store', type=float, default=3.0,
                   help="Define the Z-Score used in the median outlier removal threshold")
parser.add_argument('--attribute_doy', dest='attribute_doy', action='store_true',
                   help="Define if the doy attribute will be used in the modelling process")
parser.add_argument('--roi', dest='roi', action='store', default="",
                   help="Select the roi version used in the validation process")
parser.add_argument('--cloud_threshold', dest='cloud_threshold', action='store', type=float, default=None,
                   help="Allow CLOUD threshold customization by user")

# parsing arguments
args = parser.parse_args()


# ### Start

try:

  # Start script time counter
  start_time = time.time()

  # Google Earth Engine API initialization
  ee.Initialize()



  # ### Working directory

  # Data path
  folderRoot = os.path.dirname(os.path.realpath(__file__))+'/data'
  if not os.path.exists(folderRoot):
    os.mkdir(folderRoot)

  # Images path
  folderCache = os.path.dirname(os.path.realpath(__file__))+'/cache'
  if not os.path.exists(folderCache):
    os.mkdir(folderCache)



  # ### ABD execution

  # check if date range was selected and build time series based on it
  yearly = False
  if "*" in args.dates:
    yearly = True
    path_df_yearly_occurrence = folderRoot+'/df_yearly_occurrence.csv'
    start = dt.strptime(args.dates.split("*")[0], "%Y-%m-%d")
    end = dt.strptime(args.dates.split("*")[1], "%Y-%m-%d")
    dates = [(start + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end-start).days+1)]
    args.dates = ",".join(dates)

  # go through each parameter
  for date in args.dates.split(","):

    # folder to save results from algorithm at
    folder = folderRoot+'/'+dt.now().strftime("%Y%m%d_%H%M%S")+'[v='+str(version)+'-'+str(args.name)+',d='+str(date)+',t='+str(args.days_threshold)+',m='+str(args.model)+',s='+str(args.sensor)+',attr='+str(args.attributes)+']'
    if not os.path.exists(folder):
      os.mkdir(folder)

    # create ABD algorithm
    algorithm = abd.Abd(lat_lon=args.lat_lon,
                        date=date,
                        days_threshold=args.days_threshold,
                        model=args.model,
                        sensor=args.sensor,
                        scaler='robust',
                        cache_path=folderCache,
                        force_cache=args.force_cache,
                        remove_outliers=True,
                        reduce_dimensionality=False,
                        morph_op=None,
                        morph_op_iters=1,
                        convolve=False,
                        convolve_radius=1,
                        outliers_zscore=args.outliers_zscore,
                        attributes=args.attributes.split(","),
                        attribute_doy=args.attribute_doy,
                        cloud_threshold=args.cloud_threshold)

    # preprocessing
    algorithm.process_timeseries_data()

    # creating train and grid datasets
    algorithm.process_training_data(df=algorithm.df_timeseries)

    # start training process
    algorithm.train()

    # apply detection algorithm
    algorithm.detect()

    # validate using ROI
    algorithm.validate_using_roi(path='users/pedroananias/'+str(args.name), rois=[date+'_'+args.sensor+'_regular'+args.roi, date+'_'+args.sensor+'_anomaly'+args.roi], labels=[0, 1])

    # save geojson
    algorithm.save_geojsons(folder=folder+"/geojson")

    # save results
    algorithm.save_dataset(df=algorithm.df_results, path=folder+'/results.csv')

    # save collection images
    if isinstance(algorithm.df_image, pd.DataFrame) and not algorithm.df_image.empty:
      algorithm.save_detection_plot(path=folder+'/detection.png')
      algorithm.save_image_png(image=algorithm.image_clip, date=dt.strptime(date, "%Y-%m-%d"), path=folder+"/"+date+".png")
      algorithm.save_image_tiff(image=algorithm.image_clip, date=dt.strptime(date, "%Y-%m-%d"), path=folder+"/"+date+".zip", folderName=args.name)
      
    # save collection images (tiff and png)
    if args.save_collection:
      algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries.png')
      algorithm.save_collection_tiff(folder=folder+"/tiff", folderName=args.name)
      algorithm.save_collection_png(folder=folder+"/png")

    # save preprocessing results
    if args.save_train:
      algorithm.save_dataset(df=algorithm.df_timeseries, path=folder+'/timeseries.csv')
      algorithm.save_dataset(df=algorithm.df_train, path=folder+'/df_train.csv')
      algorithm.save_dataset(df=algorithm.df_gridsearch, path=folder+'/df_gridsearch.csv')
      algorithm.save_dataset(df=algorithm.df_image, path=folder+'/df_image.csv')

    # results
    # add results and save it on disk
    if not algorithm.df_results is None and not algorithm.df_results.empty:
      path_df_results = folderRoot+'/results.csv'
      df_results = pd.read_csv(path_df_results).drop(['Unnamed: 0'], axis=1, errors="ignore").append(algorithm.df_results) if os.path.exists(path_df_results) else algorithm.df_results.copy(deep=True)
      df_results.to_csv(r''+path_df_results)

    # yearly occurrence
    # add results to yearly occurrence
    if yearly and isinstance(algorithm.df_image, pd.DataFrame) and not algorithm.df_image.empty:
      df_image = algorithm.df_image.astype(str)
      df_yearly_occurrence = pd.read_csv(path_df_yearly_occurrence).drop(['Unnamed: 0'], axis=1, errors="ignore").append(df_image) if os.path.exists(path_df_yearly_occurrence) else df_image.copy(deep=True)
      df_yearly_occurrence.to_csv(r''+path_df_yearly_occurrence)

      # clear memory
      del df_yearly_occurrence
      gc.collect()

    # clear memory
    del algorithm
    gc.collect()

  # check if it should build yearly occurrences and save it
  if yearly and os.path.exists(path_df_yearly_occurrence):

    # fix/build yearly dataframe
    df_yearly_occurrence = pd.read_csv(path_df_yearly_occurrence).drop(['Unnamed: 0','pixel','index','doy','ndvi','fai','ndvi_median','fai_median','ndvi_diff','fai_diff'], axis=1, errors="ignore")
    df_yearly_occurrence.rename(columns={"label_0": "occurrence"}, inplace=True)
    df_yearly_occurrence['year'] = pd.to_datetime(df_yearly_occurrence['date'], format='%Y-%m-%d', errors='ignore').dt.year.astype(dtype=np.int64, errors='ignore')
    df_yearly_occurrence['cloud'] = df_yearly_occurrence['cloud'].astype(dtype=np.int64, errors='ignore')
    df_yearly_occurrence['not_occurrence'] = 0

    # sum all occurrences, clouds and non_occurrence by year, lat and lon
    df_yearly_occurrence.loc[((df_yearly_occurrence['occurrence']==0.0) & (df_yearly_occurrence['cloud']==0.0)), 'not_occurrence'] = 1
    df_yearly_occurrence = df_yearly_occurrence.groupby(['lat','lon','year']).sum().reset_index()

    # add porcentage of occurrence and cloud
    df_yearly_occurrence['pct_occurrence']   = (df_yearly_occurrence['occurrence']/(df_yearly_occurrence['occurrence']+df_yearly_occurrence['not_occurrence']))*100
    df_yearly_occurrence['pct_cloud']        = (df_yearly_occurrence['cloud']/(df_yearly_occurrence['occurrence']+df_yearly_occurrence['not_occurrence']+df_yearly_occurrence['cloud']))*100
    df_yearly_occurrence['instants']         = df_yearly_occurrence['occurrence']+df_yearly_occurrence['not_occurrence']+df_yearly_occurrence['cloud']

    # fix problem with 1 instant pixels for series average bigger than 10 instants
    if df_yearly_occurrence['instants'].mean()>10:
      df_yearly_occurrence.loc[(df_yearly_occurrence['instants']==1), 'pct_occurrence'] = np.nan
      df_yearly_occurrence['pct_occurrence'].fillna(method='ffill', inplace=True)

    # folder to save results from algorithm at
    folder = folderRoot+'/'+dt.now().strftime("%Y%m%d_%H%M%S")+'[v='+str(version)+'-'+str(args.name)+']'
    if not os.path.exists(folder):
      os.mkdir(folder)

    # create new ABD intance
    algorithm = abd.Abd(lat_lon=args.lat_lon,
                        date=args.dates.split(",")[0],
                        days_threshold=args.days_threshold,
                        model=args.model,
                        sensor=args.sensor,
                        scaler='robust',
                        cache_path=folderCache,
                        force_cache=args.force_cache,
                        remove_outliers=True,
                        reduce_dimensionality=False,
                        morph_op=None,
                        morph_op_iters=1,
                        convolve=False,
                        convolve_radius=1,
                        outliers_zscore=args.outliers_zscore,
                        attributes=args.attributes.split(","),
                        attribute_doy=args.attribute_doy,
                        cloud_threshold=args.cloud_threshold)

    # save occurrences plot
    algorithm.save_occurrences_plot(df=df_yearly_occurrence, path=folder+'/yearly_svm.png')

    # save occurrences geojson
    algorithm.save_occurrences_geojson(df=df_yearly_occurrence, path=folder+'/yearly_svm.json')

  # ### Script termination notice
  script_time_all = time.time() - start_time
  debug = "***** Script execution completed successfully (-- %s seconds --) *****" %(script_time_all)
  print()
  print(debug)

except:

    # ### Script execution error warning

    # Execution
    print()
    print()
    debug = "***** Error on script execution: "+str(traceback.format_exc())
    print(debug)

    # Removes the folder created initially with the result of execution
    script_time_all = time.time() - start_time
    debug = "***** Script execution could not be completed (-- %s seconds --) *****" %(script_time_all)
    print(debug)
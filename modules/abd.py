#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################################
# ### ABD - Anomalous Behaviour Detection
# ### Module responsible for applying the Anomalous Behaviour Detection algorithm
##################################################################################################################

# Dependencyies
# Base
import ee
import numpy as np
import pandas as pd
import hashlib
import PIL
import requests
import os
import joblib
import gc
import sys
import traceback
import math
import scipy
import os
import time
import geojson
import warnings
import multiprocessing
from io import BytesIO
from datetime import datetime as dt
from datetime import timedelta as td

# Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle, Patch
from matplotlib import cm

# Machine Learning
from sklearn import preprocessing, svm, model_selection, metrics, feature_selection, ensemble, decomposition, cluster

# Deep Learning
import tensorflow

# Local
import modules.misc as misc
import modules.gee as gee

# ignore warnings
warnings.filterwarnings('ignore')

# Anomalous Behaviour Detection
class Abd:

  # configuration
  anomaly                     = 1
  dummy                       = -99999
  cloud_threshold             = 0.50 # only images where clear pixels are greater than 50%
  reduction_threshold         = 0.70 # reduce half of the data
  max_tile_pixels             = 10000000 # if higher, will split the geometry into tiles
  indices_thresholds          = {'red': 0.0, 'green': 0.0, 'blue': 0.0, 'nir': 0.0, 'swir': 0.0, 'ndwi': 0.3, 'ndvi': -0.15, 'sabi': -0.10, 'fai': -0.004, 'slope': -0.05}
  n_cores                     = int(multiprocessing.cpu_count()*0.75) # only 75% of available cores
  random_state                = 123 # random state used in numpy and related shuffling problems

  # attributes
  attributes                  = []
  attributes_extra            = []
  attributes_inverse          = ['ndwi']
  attributes_full             = ['cloud', 'red', 'green', 'blue', 'nir', 'swir', 'ndwi', 'ndvi', 'sabi', 'fai', 'slope']

  # supports
  dates_timeseries            = [None,None]
  dates_timeseries_interval   = []
  scaler                      = None
  sensor_params               = None
  force_cache                 = None
  median_attributes           = None
  outliers_zscore             = None

  # masks
  water_mask                  = None
  
  # collections
  collection                  = None
  collection_water            = None

  # clips
  image_clip                  = None
  image_is_good               = 0

  # sample variables
  sample_total_pixel          = None
  sample_clip                 = None
  sample_lon_lat              = [[0,0],[0,0]]
  splitted_geometry           = []

  # dataframes
  df_train                    = None
  df_gridsearch               = None
  df_timeseries               = None
  df_image                    = None
  df_results                  = None
  df_columns                  = []
  df_columns_results          = ['model', 'date_detection', 'date_execution', 'time_execution', 'runtime', 'days_threshold', 'size_image', 'size_train', 'size_dates', 'roi', 'scaler', 'remove_outliers', 'reduce_dimensionality', 'attribute_doy', 'morph_op', 'morph_op_iters', 'convolve', 'convolve_radius', 'acc', 'f1score', 'kappa', 'vkappa', 'tau', 'vtau', 'mcc', 'p_value', 'fp', 'fn', 'tp', 'tn']
  
  # classifiers
  classifiers                 = {}
  classifiers_runtime         = {}

  # hash
  hash_string                 = "abd-2020111801"

  # constructor
  def __init__(self,
               lat_lon:               str,
               date:                  dt,
               days_threshold:        int           = 180,
               model:                 str           = None,
               sensor:                str           = "modis",
               scale:                 int           = None,
               scaler:                str           = 'robust',
               cache_path:            str           = None, 
               force_cache:           bool          = False,
               remove_outliers:       bool          = True,
               reduce_dimensionality: bool          = False,
               morph_op:              str           = None,
               morph_op_iters:        int           = 1,
               convolve:              bool          = False,
               convolve_radius:       int           = 1,
               outliers_zscore:       float         = 3.0,
               attributes:            list          = ['ndvi', 'fai'],
               attribute_doy:         bool          = False,
               cloud_threshold:       float         = None):
    
    # get sensor parameters
    self.sensor_params    = gee.get_sensor_params(sensor)
    self.scale            = self.sensor_params['scale'] if not scale else scale

    # warning
    print()
    print("Selected sensor: "+self.sensor_params['name'])

    # user defined parameters
    self.geometry               = gee.get_geometry_from_lat_lon(lat_lon)
    self.date                   = dt.strptime(date, "%Y-%m-%d")
    self.days_threshold         = days_threshold
    self.model                  = model
    self.sensor                 = sensor
    self.cache_path             = cache_path
    self.lat_lon                = lat_lon
    self.force_cache            = force_cache
    self.remove_outliers        = remove_outliers
    self.reduce_dimensionality  = reduce_dimensionality
    self.attribute_doy          = attribute_doy
    self.morph_op               = morph_op
    self.morph_op_iters         = morph_op_iters
    self.convolve               = convolve
    self.convolve_radius        = convolve_radius
    self.outliers_zscore        = outliers_zscore

    # check selected attributes
    if 'cloud' not in attributes:
      self.attributes = ['cloud'] + list(attributes)
    else:
      self.attributes = list(attributes)

    # check if it has to use doy attribute
    if self.attribute_doy:
      self.attributes_extra.append('doy')

    # check if it has to change CLOUD threshold
    if not cloud_threshold is None:
      self.cloud_threshold = cloud_threshold

    # correct dataframe columns
    self.df_columns = ['pixel','index','date','doy','lat','lon']+self.attributes

    # create scaler
    if scaler == 'minmax':
      self.scaler_str = 'minmax'
      self.scaler     = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    elif scaler == 'robust':
      self.scaler_str = 'robust'
      self.scaler     = preprocessing.RobustScaler(quantile_range=(25, 75), copy=True)
    else:
      self.scaler_str = 'standard'
      self.scaler     = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    # reset classifier
    self.classifiers             = {}
    self.classifiers_runtime     = {}

    # creating initial sensor precollection
    collection = gee.get_sensor_collections(geometry=self.geometry, sensor=self.sensor, dates=[dt.strftime(self.date-td(days=16), "%Y-%m-%d"), dt.strftime(self.date+td(days=16), "%Y-%m-%d")])

    # correcting image, checking if selected exist in dataset
    last = None
    dates_list = [dt.strptime(d, "%Y-%m-%d") for d in collection[0].map(lambda image: ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})).distinct('date').aggregate_array("date").getInfo()]
    for d in dates_list:
      if dt.strftime(self.date, "%Y-%m-%d") == dt.strftime(d, "%Y-%m-%d"):
        break
      elif d > self.date and last:
        self.date = d if abs((d - self.date).days) <= abs((self.date - last).days) else last
        break
      last = d

    # time series expansion
    self.dates_timeseries[1]  = self.date + td(days=self.days_threshold)
    self.dates_timeseries[0]  = self.date - td(days=self.days_threshold)

    # correct time series expansion (sensor end date)
    if self.dates_timeseries[1] > dt.now():
      self.dates_timeseries[0] -= td(days=(self.dates_timeseries[1] - dt.now()).days)
      self.dates_timeseries[1] = dt.now()

    # correct time series expansion (sensor start date)
    if self.dates_timeseries[0] < self.sensor_params['start']:
      self.dates_timeseries[1] += td(days=(self.sensor_params['start'] - self.dates_timeseries[0]).days)
      self.dates_timeseries[0] = self.sensor_params['start']

    # creating final sensor collection
    collection, collection_water      = gee.get_sensor_collections(geometry=self.geometry, sensor=self.sensor, dates=[dt.strftime(self.dates_timeseries[0], "%Y-%m-%d"), dt.strftime(self.dates_timeseries[1], "%Y-%m-%d")])

    # create useful time series
    self.collection                   = collection
    self.collection_water             = collection_water
    self.dates_timeseries_interval    = misc.remove_duplicated_dates([dt.strptime(d, "%Y-%m-%d") for d in self.collection.map(lambda image: ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})).distinct('date').aggregate_array("date").getInfo()])

    # preprocessing - water mask extraction
    self.water_mask                   = self.create_water_mask(self.morph_op, self.morph_op_iters)

    # count sample pixels and get sample min max coordinates
    self.sample_clip                  = self.clip_image(ee.Image(abs(self.dummy)))
    self.sample_total_pixel           = gee.get_image_counters(image=self.sample_clip.select("constant"), geometry=self.geometry, scale=self.scale)["constant"]
    coordinates_min, coordinates_max  = gee.get_image_min_max(image=self.sample_clip, geometry=self.geometry, scale=self.scale)
    self.sample_lon_lat               = [[float(coordinates_min['latitude']),float(coordinates_min['longitude'])],[float(coordinates_max['latitude']),float(coordinates_max['longitude'])]]

    # split geometry in tiles
    self.splitted_geometry            = self.split_geometry()

    # preprocessing - images clips
    self.image_clip                   = self.clip_image(self.extract_image_from_collection(self.date), geometry=self.geometry, scale=self.scale)

    # check if requested date is good for processing
    print()
    print("Check if requested date is good for processing ["+str(self.date.strftime("%Y-%m-%d"))+"]...")
    self.image_is_good = 0
    for i, geometry in enumerate(self.splitted_geometry):

      # geometry
      print("Extracting geometry ("+str(self.image_is_good)+") "+str(i+1)+" of "+str(len(self.splitted_geometry))+"...")
      
      # counters
      if not self.image_clip is None:
      
        # count image pixels
        counters            = self.image_clip.select([self.sensor_params['red'],'water','water_nocloud']).reduceRegion(reducer=ee.Reducer.count(), geometry=geometry, scale=self.scale).getInfo()
        total_pixel         = int(counters[self.sensor_params['red']])
        total_water_pixel   = int(counters['water'])
        total_water_nocloud = int(counters['water_nocloud'])
        
        # calculate pixel score
        if (len(self.splitted_geometry) > 1):
          sample_total_pixel = gee.get_image_counters(image=self.clip_image(ee.Image(abs(self.dummy))).select("constant"), geometry=geometry, scale=self.scale)["constant"]
        else:
          sample_total_pixel = self.sample_total_pixel

        # score total pixel and cloud
        pixel_score         = 0.0 if total_pixel == 0 else round(total_pixel/sample_total_pixel,5)
        water_nocloud_score = 0.0 if total_water_nocloud == 0 else round(total_water_nocloud/total_water_pixel, 5)
        
        # warning
        print("Pixel and cloud score for geometry #"+str(i+1)+": "+str(pixel_score)+" and "+str(water_nocloud_score)+"!")

        # check if image is good for processing
        if pixel_score >= 0.50 and water_nocloud_score >= self.cloud_threshold:
          self.image_is_good += 1

    # check image result
    if self.image_is_good:
      print("Ok, image is good for processing: "+str(self.image_is_good))
    else:
      print("Error! Please, pick another date and try again.")
    print()

    # warning
    print("Statistics: scale="+str(self.scale)+" meters, pixels="+str(self.sample_total_pixel)+", date_start='"+self.dates_timeseries[0].strftime("%Y-%m-%d")+"', date_end='"+self.dates_timeseries[1].strftime("%Y-%m-%d")+"', interval_images='"+str(self.collection.size().getInfo())+"', interval_unique_images='"+str(len(self.dates_timeseries_interval))+"', water_mask_images='"+str(self.collection_water.size().getInfo())+"', days_threshold='"+str(self.days_threshold)+"', morph_op='"+str(self.morph_op)+"', morph_op_iters='"+str(self.morph_op_iters)+"', convolve='"+str(self.convolve)+"', convolve_radius='"+str(self.convolve_radius)+"', scaler='"+str(self.scaler_str)+"', model='"+str(self.model)+"', outliers_zscore='"+str(self.outliers_zscore)+"', attributes='"+str(self.attributes)+"', attribute_doy='"+str(self.attribute_doy)+"'")


  # create the water mask
  def create_water_mask(self, morph_op: str = None, morph_op_iters: int = 1):

    # water mask
    if self.sensor == "modis":
      water_mask = self.collection_water.mode().select('water_mask').eq(1)
    elif "landsat" in self.sensor:
      water_mask = self.collection_water.mode().select('water').eq(2)
    else:
      water_mask = self.collection_water.mode().select('water').gt(0)

    # morphological operations
    if not morph_op is None and morph_op != '':
      if morph_op   == 'closing':
        water_mask = water_mask.focal_max(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters).focal_min(kernel=ee.Kernel.circle(radius=1), iterations=morph_op_iters)
      elif morph_op == 'opening':
        water_mask = water_mask.focal_min(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters).focal_max(kernel=ee.Kernel.circle(radius=1), iterations=morph_op_iters)
      elif morph_op == 'dilation':
        water_mask = water_mask.focal_max(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters)
      elif morph_op == 'erosion':
        water_mask = water_mask.focal_min(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters)

    # build image with mask
    return ee.Image(0).blend(ee.Image(abs(self.dummy)).updateMask(water_mask))


  # clipping image
  def clip_image(self, image: ee.Image, scale: int = None, geometry: ee.Geometry = None):
    geometry  = self.geometry if geometry is None else geometry
    scale     = self.sensor_params['scale'] if scale is None else scale
    if image is None:
      return None
    clip      = image.clipToBoundsAndScale(geometry=geometry, scale=scale)
    if self.sensor_params['scale'] > self.scale:
      return clip.resample('bicubic').reproject(crs=image.projection(),scale=self.scale)
    else:
      return clip


  # applying water mask to indices
  def apply_water_mask(self, image: ee.Image, remove_empty_pixels = False):
    for i, attribute in enumerate(self.attributes_full):
      image = gee.apply_mask(image, self.water_mask, attribute,  attribute+"_water", remove_empty_pixels)
    return image
  

  # extract image from collection
  def extract_image_from_collection(self, date):
    try:
      collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d")))
      if int(collection.size().getInfo()) == 0:
        collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=2)).strftime("%Y-%m-%d")))
        if int(collection.size().getInfo()) == 0:
          return None
      image = ee.Image(collection.max())
      return self.apply_water_mask(image.set('system:id', collection.first().get('system:id').getInfo()), False)
    except:
      return None


  # split images into tiles
  def split_geometry(self):

    # check total of pixels
    total = self.sample_total_pixel*(len(self.attributes_full)+2)
    if total > self.max_tile_pixels:

      # total of tiles needed
      tiles = math.ceil(total/self.max_tile_pixels)

      # lat and lons range
      latitudes       = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=tiles+1)
      longitudes      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=tiles+1)

      # go through all latitudes and longitudes
      geometries = []
      for i, latitude in enumerate(latitudes[:-1]):
        for j, longitude in enumerate(longitudes[:-1]):
          x1 = [i,j]
          x2 = [i+1,j+1]
          geometry = gee.get_geometry_from_lat_lon(str(latitudes[x1[0]])+","+str(longitudes[x1[1]])+","+str(latitudes[x2[0]])+","+str(longitudes[x2[1]]))
          geometries.append(geometry)

      # return all created geometries
      return geometries

    else:
      
      # return single geometry
      return [gee.get_geometry_from_lat_lon(self.lat_lon)]


  # decode from OneClassSVM labels
  def decode_ocsvm_labels(self, y):
    y = np.array(y)
    return np.where(y==-1, self.anomaly, np.where(y==1, 0, y))


  # encode to OneClassSVM labels
  def encode_ocsvm_labels(self, y):
    y = np.array(y)
    return np.where(y==self.anomaly, -1, np.where(y==0, 1, y))


  # decode from Isolation Forest labels
  def decode_if_labels(self, y):
    y = np.array(y)
    return np.where(y==-1, self.anomaly, np.where(y==1, 0, y))


  # encode to Isolation Forest labels
  def encode_if_labels(self, y):
    y = np.array(y)
    return np.where(y==self.anomaly, -1, np.where(y==0, 1, y))


  # apply K-Means dimenisonality reduction
  def apply_kmeans(self, df: pd.DataFrame):

    # jump line
    print()
    print("Starting dimensionality reduction using K-Means...")

    # dif attributes
    diff_attributes   = self.attributes_extra+[a+"_diff" for a in self.attributes if a != 'cloud']

    # get gridsearch data
    X                 = self.scaler.fit_transform(df[diff_attributes].values.reshape((-1, len(diff_attributes))))

    # create kmeans clustering
    kmeans            = cluster.MiniBatchKMeans(n_clusters=20, max_iter=1000, tol=0.0001, batch_size=100, random_state=self.random_state)
    y_groups          = kmeans.fit_predict(X)
    unique, counts    = np.unique(y_groups, return_counts=True)

    # split groups and sort
    groups = {k: v for k, v in sorted(dict(zip(unique, counts)).items(), key=lambda item: item[1], reverse=True)}

    # select data
    summ      = 0
    total     = len(df)
    n_groups  = []
    for group, value in groups.items():
      summ += value
      prop = summ/total
      if prop <= self.reduction_threshold:
        n_groups.append(group)

    # select only groups meeting reduction threshold
    df['group'] = y_groups
    df = df[df['group'].isin(n_groups)].drop(['group'], axis=1)

    # jump line
    print("Dimensionality reduction finished! Parametrization dataframe was reduced from "+str(total)+" -> "+str(len(df))+" samples")
    return df


  # get cache files for datte
  def get_cache_files(self, date):
    prefix            = self.hash_string.encode()+self.lat_lon.encode()+self.sensor.encode()+str(self.morph_op).encode()+str(self.morph_op_iters).encode()+str(self.convolve).encode()+str(self.convolve_radius).encode()
    hash_image        = hashlib.md5(prefix+(date.strftime("%Y-%m-%d")+'original').encode())
    hash_timeseries   = hashlib.md5(prefix+(self.dates_timeseries[0].strftime("%Y-%m-%d")+self.dates_timeseries[1].strftime("%Y-%m-%d")).encode())
    hash_classifiers  = hashlib.md5(prefix+(self.dates_timeseries[0].strftime("%Y-%m-%d")+self.dates_timeseries[1].strftime("%Y-%m-%d")).encode()+('classifier').encode())
    hash_runtime      = hashlib.md5(prefix+(self.dates_timeseries[0].strftime("%Y-%m-%d")+self.dates_timeseries[1].strftime("%Y-%m-%d")).encode()+('runtime').encode())
    return [self.cache_path+'/'+hash_image.hexdigest(), self.cache_path+'/'+hash_timeseries.hexdigest(), self.cache_path+'/'+hash_classifiers.hexdigest(), self.cache_path+'/'+hash_runtime.hexdigest()]


  # extract image's coordinates and pixels values
  def extract_image_pixels(self, image: ee.Image, date):

    # warning
    print("Processing date ["+str(date.strftime("%Y-%m-%d"))+"]...")

    # attributes
    lons_lats_attributes        = None
    cache_files                 = self.get_cache_files(date)
    df_timeseries               = pd.DataFrame(columns=self.df_columns)

    # trying to find image in cache
    try:

      # warning - user disabled cache
      if self.force_cache:
        raise Exception()

      # extract pixel values from cache
      lons_lats_attributes      = joblib.load(cache_files[0])
      if lons_lats_attributes is None:
        raise Exception()

      # check if image is empty and return empty dataframe
      if len(lons_lats_attributes) == 0:

        # clear memory
        del lons_lats_attributes
        gc.collect()

        # return empty dataframe
        return df_timeseries

    # error finding image in cache
    except:

      # image exists
      try:

        # go through each tile
        lons_lats_attributes  = np.array([]).reshape(0, len(self.attributes_full)+2)
        for i, geometry in enumerate(self.splitted_geometry):

          # geometry
          print("Extracting geometry ("+str(len(lons_lats_attributes))+") "+str(i+1)+" of "+str(len(self.splitted_geometry))+"...")

          # counters
          clip = self.clip_image(self.extract_image_from_collection(date=date), geometry=geometry, scale=self.scale)
          if not clip is None:
            counters            = clip.select([self.sensor_params['red'],'water','water_nocloud']).reduceRegion(reducer=ee.Reducer.count(), geometry=geometry, scale=self.scale).getInfo()

            # count image pixels
            total_pixel         = int(counters[self.sensor_params['red']])
            total_water_pixel   = int(counters['water'])
            total_water_nocloud = int(counters['water_nocloud'])

            # calculate pixel score
            if (len(self.splitted_geometry) > 1):
              sample_total_pixel = gee.get_image_counters(image=self.clip_image(ee.Image(abs(self.dummy))).select("constant"), geometry=geometry, scale=self.scale)["constant"]
            else:
              sample_total_pixel = self.sample_total_pixel

            # score total pixel and cloud
            pixel_score         = 0.0 if total_pixel == 0 else round(total_pixel/sample_total_pixel,5)
            water_nocloud_score = 0.0 if total_water_nocloud == 0 else round(total_water_nocloud/total_water_pixel, 5)

            # warning
            print("Pixel and cloud score for geometry #"+str(i+1)+": "+str(pixel_score)+" and "+str(water_nocloud_score)+"!")

            # check if image is good for processing
            if pixel_score >= 0.50 and water_nocloud_score >= self.cloud_threshold:
              geometry_lons_lats_attributes = gee.extract_latitude_longitude_pixel(image=clip, geometry=geometry, bands=[a+"_water" for a in self.attributes_full], scale=self.scale)
              lons_lats_attributes = np.vstack((lons_lats_attributes, geometry_lons_lats_attributes))
              del geometry_lons_lats_attributes
              gc.collect()
        
        # empty image
        if len(lons_lats_attributes) == 0:

          # warning
          print("Image is not good for processing and it was discarded!")
          
          # Clear pixels, image is not good
          lons_lats_attributes = None
          joblib.dump(np.array([]), cache_files[0], compress=3)

          # clear memory
          del lons_lats_attributes
          gc.collect()

          # return empty dataframe
          return df_timeseries

        # save in cache
        else:
          joblib.dump(lons_lats_attributes, cache_files[0], compress=3)

      # error in the extraction process
      except:
        
        # warning
        print("Error while extracting pixels from image "+str(date.strftime("%Y-%m-%d"))+": "+str(traceback.format_exc()))

        # reset attributes
        lons_lats_attributes = None

        # remove cache file
        if os.path.exists(cache_files[0]):
          os.remove(cache_files[0])

    # check if has indices to process
    try:

      # check if they are valid
      if lons_lats_attributes is None:
        raise Exception()

      # fix image array
      index_delete = []
      for i in range(0, len(self.attributes_full)):
        if self.attributes_full[i] not in self.attributes:
          index_delete.append(i+2)
      lons_lats_attributes = np.delete(lons_lats_attributes, index_delete, 1)

      # build arrays
      extra_attributes        = np.array(list(zip([0]*len(lons_lats_attributes),[0]*len(lons_lats_attributes),[date.strftime("%Y-%m-%d")]*len(lons_lats_attributes),[0]*len(lons_lats_attributes))))
      #lons_lats_attributes    = np.concatenate((extra_attributes, lons_lats_attributes), axis=1)
      lons_lats_attributes    = np.hstack((extra_attributes, lons_lats_attributes))

      # build dataframe and fix column index values
      df_timeseries           = pd.DataFrame(data=lons_lats_attributes, columns=self.df_columns).infer_objects().sort_values(['lat','lon']).reset_index(drop=True)
      df_timeseries['pixel']  = range(0,len(df_timeseries))

      # show example of time series
      print(df_timeseries.head())

      # gabagge collect
      del lons_lats_attributes, extra_attributes
      gc.collect()

      # return all pixels in an three pandas format
      return df_timeseries

    # no data do return
    except:

      # warning
      print("Error while extracting pixels from image "+str(date.strftime("%Y-%m-%d"))+": "+str(traceback.format_exc()))

      # remove cache file
      if os.path.exists(cache_files[0]):
        os.remove(cache_files[0])
      
      # clear memory
      del lons_lats_attributes
      gc.collect()

      # return empty dataframe
      return df_timeseries


  # merge two or more timeseries
  def merge_timeseries(self, df_list: list):
    df = pd.concat(df_list, ignore_index=True, sort=False)
    gc.collect()
    return self.fix_timeseries(df)

  # fix timeseries values
  def fix_timeseries(self, df: pd.DataFrame):
    df['index']             = np.arange(start=0, stop=len(df), step=1, dtype=np.int64)
    df['pixel']             = df['pixel'].astype(dtype=np.int64, errors='ignore')
    df[['lat','lon']]       = df[['lat','lon']].astype(dtype=np.float64, errors='ignore')
    df['date']              = pd.to_datetime(df['date'], errors='ignore')
    df['doy']               = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='ignore').dt.dayofyear.astype(dtype=np.int64, errors='ignore')
    df[self.attributes]     = df[self.attributes].apply(pd.to_numeric, errors='ignore')
    gc.collect()
    return df.sort_values(by=['date', 'pixel'])


  # normalize indices
  def normalize_indices(self, df: pd.DataFrame):
    if 'ndwi' in self.attributes:
      df.loc[df['ndwi']<-1, 'ndwi'] = -1
      df.loc[df['ndwi']>1, 'ndwi'] = 1
    if 'ndvi' in self.attributes:
      df.loc[df['ndvi']<-1, 'ndvi'] = -1
      df.loc[df['ndvi']>1, 'ndvi'] = 1
    if 'sabi' in self.attributes:
      df.loc[df['sabi']<-1, 'sabi'] = -1
      df.loc[df['sabi']>1, 'sabi'] = 1
    return df

  # process a timeseries
  def process_timeseries_data(self, force_cache: bool = False):

    # warning
    if self.image_is_good:
      print()
      print("Starting time series processing ...")

      # attributes
      df_timeseries             = pd.DataFrame(columns=self.df_columns)
      attributes                = [a for a in self.attributes if a != 'cloud']
      attributes_median         = [a+"_median" for a in attributes]

      # check timeseries is already on cache
      cache_files               = self.get_cache_files(date=self.date)
      try:

        # warning
        print("Trying to extract it from the cache...")

        # warning 2
        if self.force_cache or force_cache:
          print("User selected option 'force_cache'! Forcing building of time series...")
          raise Exception()

        # extract dataframe from cache
        df_timeseries = self.fix_timeseries(df=joblib.load(cache_files[1]))

      # if do not exist, process normally and save it in the end
      except:

        # warning
        print("Error trying to get it from cache: either doesn't exist or is corrupted! Creating it again...")

        # process all dates in time series
        for date in self.dates_timeseries_interval:

          # extract pixels from image
          # check if is good image (with pixels)
          try:
            df_timeseries_ = self.extract_image_pixels(image=self.extract_image_from_collection(date=date), date=date)
            if df_timeseries_.size > 0:
              df_timeseries = self.merge_timeseries(df_list=[df_timeseries, df_timeseries_])
            gc.collect()
          except:
            pass

        # get only good dates
        # fix dataframe index
        if not df_timeseries is None:
          df_timeseries['index'] = range(0,len(df_timeseries))

          # save in cache
          if self.cache_path:
            joblib.dump(df_timeseries, cache_files[1], compress=3)

      # remove dummies
      for attribute in attributes:
        df_timeseries = df_timeseries[(df_timeseries[attribute]!=abs(self.dummy))]

      # change cloud values
      df_timeseries.loc[df_timeseries['cloud'] == abs(self.dummy), 'cloud'] = 0.0

      # remove duplicated values
      df_timeseries.drop_duplicates(subset=['pixel','date','lat','lon']+attributes, keep='last', inplace=True)

      # create median timeseries columns
      df_timeseries_median = df_timeseries.copy(deep=True).groupby(['pixel']).median().reset_index().drop(['index', 'doy', 'cloud'], axis=1).sort_values(by=['pixel'])
      for attribute in attributes_median:
        df_timeseries[attribute] = np.nan
      first_date = dt.strptime(df_timeseries['date'].unique()[0].astype(str).split("T")[0], "%Y-%m-%d")
      pixels = df_timeseries[(df_timeseries['pixel'].isin(df_timeseries_median['pixel'].values)) & (df_timeseries['date']==first_date)].groupby(['pixel']).median().reset_index()['pixel'].values
      df_timeseries.loc[(df_timeseries['pixel'].isin(pixels)) & (df_timeseries['date']==first_date), attributes_median] = df_timeseries_median[df_timeseries_median['pixel'].isin(pixels)][attributes].values
      df_timeseries = df_timeseries.sort_values(by=['pixel','date']).fillna(method='ffill').sort_values(by=['date', 'pixel'])
        
      # save modified dataframe to its original variable
      self.df_timeseries = df_timeseries[self.df_columns+attributes_median].dropna()

      # correction of empty dates
      self.dates_timeseries_interval = [dt.strptime(date.astype(str).split("T")[0],"%Y-%m-%d") for date in self.df_timeseries['date'].unique()]

      # garbagge collect
      del df_timeseries, df_timeseries_median
      gc.collect()

      # warning
      print("finished!")


  # process training dataset
  def process_training_data(self, df: pd.DataFrame):

    # warning
    if self.image_is_good:
      print()
      print("Processing training data...")

      # attributes
      attributes = [a for a in self.attributes if a != 'cloud']

      # show statistics
      print(df.describe())

      # warning
      print()
      print("Removing median trend...")

      # remove median trend
      for attribute in attributes:
        df[attribute+"_diff"] = df[attribute] - df[attribute+"_median"]

      # separate dataframes
      self.df_gridsearch   = df[df['date'] != self.date.strftime("%Y-%m-%d")]
      self.df_train        = df[df['date'] != self.date.strftime("%Y-%m-%d")]
      self.df_image        = df[df['date'] == self.date.strftime("%Y-%m-%d")]

      # show statistics
      print(self.df_train.describe())

      # check if it has to remove outliers
      if self.remove_outliers:

        # warning
        print()
        print("Removing outliers on training dataframe...")

        # remove outliers (min than 3 stddev)
        self.df_train = self.df_train[(np.abs(scipy.stats.zscore(self.df_train[['index']+attributes])) < self.outliers_zscore).all(axis=1)]

        # show statistics
        print(self.df_train.describe())

      # warning
      print()
      print("Median modelling...")

      # median modeling
      df_queries = []
      for attribute in attributes:

        # create median modeling queries to extract training data
        # only pixels' values within mean +- std
        mean              = self.df_train[attribute].mean()
        std               = self.df_train[attribute].std()
        df_queries.append('('+attribute+' >= '+str(mean-std)+' and '+attribute+' <= '+str(mean+std)+')')

      # apply median modeling query
      self.df_train = self.df_train.query(" and ".join(df_queries))

      # new column to hold pixel labels (targetting training data with non-anomaly and gridsearch data with both, non-anomaly and anomaly)
      self.df_train['label']                                                                             = 0
      self.df_gridsearch['label']                                                                        = self.anomaly
      self.df_gridsearch.loc[self.df_gridsearch['index'].isin(self.df_train['index'].values),'label']    = 0

      # show statistics
      print(self.df_train.describe())

      # check if it should apply aggregation algorithm to reduce dimensinality
      if self.reduce_dimensionality:
        self.df_gridsearch = self.apply_kmeans(df=self.df_gridsearch)
      
      # warning
      print()
      print("Total of "+str(len(self.df_train))+" and "+str(len(self.df_gridsearch))+" pixels for training and parametrization dataframes, respectively!")
      print("finished!")


  # start training process
  def train(self):

    # jump line
    if self.image_is_good:
      print()
      print("Starting the training process...")

      # attributes
      attributes = self.attributes_extra+[a+"_diff" for a in self.attributes if a != 'cloud']

      # split data
      X_gridsearch  = self.scaler.fit_transform(self.df_gridsearch[attributes].values.reshape((-1, len(attributes))))
      y_gridsearch  = self.df_gridsearch['label'].values.reshape(1,-1)[0]
      X_train       = self.scaler.transform(self.df_train[attributes].values.reshape((-1, len(attributes))))
      y_train       = self.df_train['label'].values.reshape(1,-1)[0]

      ########################################################################
      # 1) OneClass Support Vector Machines

      # check if model was selected in options
      if self.model is None or self.model == "ocsvm":

        # gridsearch data
        X_gridsearch_, _, y_gridsearch_, _ = model_selection.train_test_split(X_gridsearch, self.encode_ocsvm_labels(y_gridsearch), train_size=0.01, shuffle=True, random_state=self.random_state)

        # jump line
        print()
        print("Creating the OneClass Support Vector Machines with RandomizedSearchCV parameterization model...")

        # RandomizedSearch params
        random_grid = [
          {
            'kernel':     ['rbf'],
            'gamma':      scipy.stats.expon(scale=.100),
            'nu':         scipy.stats.expon(scale=.100),
            'degree':     [3],
            'coef0':      [0.0],
            'shrinking':  [True, False],
          },
          {
            'kernel':     ['rbf'],
            'gamma':      ['auto','scale'],
            'nu':         scipy.stats.expon(scale=.100),
            'degree':     [3],
            'coef0':      [0.0],
            'shrinking':  [True, False],
          }
        ]

        # warning
        print("Starting RandomizedSearch ("+str(len(X_gridsearch_))+" pixels) parameters selection...")

        # initialize searching process for OneClassSVM best parameters selection
        start_time = time.time()
        rs = model_selection.RandomizedSearchCV(estimator=svm.OneClassSVM(verbose=False), param_distributions=random_grid, scoring={'accuracy':metrics.make_scorer(metrics.accuracy_score), 'kappa': metrics.make_scorer(metrics.cohen_kappa_score)}, refit='kappa', n_iter=50, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch_, y_gridsearch_)

        # train data
        X_train_, X_test, y_train_, y_test = model_selection.train_test_split(X_train, self.encode_ocsvm_labels(y_train), train_size=0.01, shuffle=True, random_state=self.random_state)

        # initialize training process for OneClassSVM
        ocsvm = rs.best_estimator_
        ocsvm.fit(X_train_, y_train_)

        # model name
        str_model = "OneClassSVM (kernel="+str(rs.best_params_['kernel'])+",gamma="+str(rs.best_params_['gamma'])+",nu="+str(rs.best_params_['nu'])+",shrinking="+str(rs.best_params_['shrinking'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = ocsvm

        # test data
        _, X_test, _, y_test = model_selection.train_test_split(X_gridsearch, y_gridsearch, test_size=0.30, shuffle=True, random_state=self.random_state)

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_true, y_pred  = y_test, self.decode_ocsvm_labels(ocsvm.predict(X_test))
        measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])
        print(metrics.classification_report(y_true, y_pred))

        # warning
        print("finished!")

      ########################################################################


      ########################################################################
      # 2) Random Forest

      # check if model was selected in options
      if self.model is None or self.model == "rf":

        # gridsearch data
        X_gridsearch_, _, y_gridsearch_, _ = model_selection.train_test_split(X_gridsearch, y_gridsearch, train_size=0.01, shuffle=True, random_state=self.random_state)

        # jump line
        print()
        print("Creating the Random Forest Classifier with RandomizedSearchCV parameterization model...")

        # RandomizedSearchCV
        random_grid = {
          'n_estimators':       np.linspace(1, 250, num=51, dtype=int, endpoint=True),
          'max_depth':          np.linspace(1, 30, num=30, dtype=int, endpoint=True),
          'min_samples_split':  np.linspace(2, 20, num=10, dtype=int, endpoint=True),
          'min_samples_leaf':   np.linspace(2, 20, num=10, dtype=int, endpoint=True),
          'max_features':       ['auto', 'sqrt', 1.0, 0.75, 0.50],
          'bootstrap':          [True, False]
        }

        # warning
        print("Starting RandomizedSearch ("+str(len(X_gridsearch_))+" pixels) parameters selection...")

        # initialize searching process for Random Forest best parameters selection
        start_time = time.time()
        rs = model_selection.RandomizedSearchCV(estimator=ensemble.RandomForestClassifier(n_jobs=self.n_cores, random_state=self.random_state), param_distributions=random_grid, scoring={'accuracy':metrics.make_scorer(metrics.accuracy_score), 'kappa': metrics.make_scorer(metrics.cohen_kappa_score)}, refit='kappa', n_iter=50, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch_, y_gridsearch_)

        # train data
        X_train_, X_test, y_train_, y_test = model_selection.train_test_split(X_gridsearch, y_gridsearch, train_size=0.70, shuffle=True, random_state=self.random_state)

        # initialize training process for Random Forest
        rf = rs.best_estimator_
        rf.fit(X_train_, y_train_)

        # model name
        str_model = "Random Forest (n_estimators="+str(rs.best_params_['n_estimators'])+",max_features="+str(rs.best_params_['max_features'])+",max_depth="+str(rs.best_params_['max_depth'])+",min_samples_leaf="+str(rs.best_params_['min_samples_leaf'])+",min_samples_split="+str(rs.best_params_['min_samples_split'])+",bootstrap="+str(rs.best_params_['bootstrap'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = rf

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_true, y_pred  = y_test, rf.predict(X_test)
        measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])
        print(metrics.classification_report(y_true, y_pred))

        # warning
        print("finished!")

      ########################################################################


      ########################################################################
      # 3) TensorFlow Autoencoder

      # check if model was selected in options
      if self.model == "ae":

        # gridsearch data
        X_gridsearch_, _, y_gridsearch_, _ = model_selection.train_test_split(X_gridsearch, y_gridsearch, train_size=0.01, shuffle=True, random_state=self.random_state)

        # jump line
        print()
        print("Creating the TensorFlow AutoEncoder with RandomizedSearchCV parameterization model...")

        #################################
        # Custom AutoEncoder Model
        # Change KerasClassifier
        class KerasClassifierModified(tensorflow.keras.wrappers.scikit_learn.KerasClassifier):
          def predict(self, x, **kwargs):
            kwargs = self.filter_sk_params(tensorflow.keras.Sequential.predict_classes, kwargs)
            classes = (self.model.predict(x, **kwargs) > 0.5).astype("int32")
            return self.classes_[classes]

        # Model Function
        def autoencoder(optimizer='adam'):
          autoencoder = tensorflow.keras.Sequential(
            [
              tensorflow.keras.layers.Input(shape=(X_train.shape[1], )),
              tensorflow.keras.layers.Dense(128, activation="tanh", activity_regularizer=tensorflow.keras.regularizers.l1(1e-3)),
              tensorflow.keras.layers.Dense(64, activation="tanh"),
              tensorflow.keras.layers.Dense(32, activation="tanh"),
              tensorflow.keras.layers.Dense(16, activation="tanh"),
              tensorflow.keras.layers.Dense(1, activation="sigmoid")
            ]
          )
          autoencoder.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer=optimizer)
          return autoencoder
        ##################################

        # RandomizedSearchCV
        random_grid = {
          'batch_size':   [2048, 4096],
          'epochs':       [10, 50, 100, 200, 300, 400, 500],
          'optimizer':    ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        }

        # warning
        print("Starting RandomizedSearch ("+str(len(X_gridsearch_))+" pixels) parameters selection...")

        # apply RandomizedSearchCV
        start_time = time.time()
        rs = model_selection.RandomizedSearchCV(estimator=KerasClassifierModified(build_fn=autoencoder, verbose=0), param_distributions=random_grid, scoring={'accuracy':metrics.make_scorer(metrics.accuracy_score), 'kappa': metrics.make_scorer(metrics.cohen_kappa_score)}, refit='kappa', n_iter=50, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch_, y_gridsearch_)

        # train data
        X_train_, X_test, y_train_, y_test = model_selection.train_test_split(X_gridsearch, y_gridsearch, train_size=0.70, shuffle=True, random_state=self.random_state)

        # initialize training process for Random Forest
        ae = rs.best_estimator_
        ae.fit(X_train_, y_train_)

        # model name
        str_model = "TensorFlow AutoEncoder (batch_size="+str(rs.best_params_['batch_size'])+",epochs="+str(rs.best_params_['epochs'])+",optimizer="+str(rs.best_params_['optimizer'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = ae

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_true, y_pred  = y_test, ae.predict(X_test)
        measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])
        print(metrics.classification_report(y_true, y_pred))

        # warning
        print("finished!")

      ########################################################################


      ########################################################################
      # 4) Isolation Forest

      # check if model was selected in optionsf
      if self.model is None or self.model == "if":

        # gridsearch data
        X_gridsearch_, _, y_gridsearch_, _ = model_selection.train_test_split(X_gridsearch, self.encode_if_labels(y_gridsearch), train_size=0.01, shuffle=True, random_state=self.random_state)

        # jump line
        print()
        print("Creating the Isolation Forest Classifier with RandomizedSearchCV parameterization model...")

        # RandomizedSearchCV
        random_grid = {
          'n_estimators':       np.linspace(1, 250, num=51, dtype=int, endpoint=True),
          'max_samples':        ['auto', 1.0, 0.75, 0.50, 0.25, 0.10, 0.05],
          'contamination':      ['auto', 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
          'max_features':       [1.0, 0.75, 0.50],
          'bootstrap':          [True, False]
        }

        # warning
        print("Starting RandomizedSearch ("+str(len(X_gridsearch_))+" pixels) parameters selection...")

        # initialize searching process for Random Forest best parameters selection
        start_time = time.time()
        rs = model_selection.RandomizedSearchCV(estimator=ensemble.IsolationForest(n_jobs=self.n_cores, random_state=self.random_state), param_distributions=random_grid, scoring={'accuracy':metrics.make_scorer(metrics.accuracy_score), 'kappa': metrics.make_scorer(metrics.cohen_kappa_score)}, refit='kappa', n_iter=50, cv=5, verbose=1, random_state=self.random_state, n_jobs=self.n_cores)
        rs.fit(X_gridsearch_, y_gridsearch_)

        # train data
        X_train_, X_test, _, y_test = model_selection.train_test_split(X_gridsearch, y_gridsearch, train_size=0.70, shuffle=True, random_state=self.random_state)

        # initialize training process for Random Forest
        iff = rs.best_estimator_
        iff.fit(X_train_)

        # model name
        str_model = "Isolation Forest (n_estimators="+str(rs.best_params_['n_estimators'])+",max_samples="+str(rs.best_params_['max_samples'])+",contamination="+str(rs.best_params_['contamination'])+",max_features="+str(rs.best_params_['max_features'])+",bootstrap="+str(rs.best_params_['bootstrap'])+")"
        self.classifiers_runtime[str_model] = time.time() - start_time
        self.classifiers[str_model] = iff

        # warning
        print()
        print("Evaluating the "+str(str_model)+" model...")

        # get predictions on test set
        y_true, y_pred  = y_test, self.decode_if_labels(iff.predict(X_test))
        measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

        # reports
        print("Report for the "+str(str_model)+" model: "+measures['string'])
        print(metrics.classification_report(y_true, y_pred))

        # warning
        print("finished!")

      ########################################################################


  # start the detection process
  # first, split detection data and then apply classifier
  def detect(self):

    # warning
    if isinstance(self.df_image, pd.DataFrame) and not self.df_image.empty:
      print()
      print("Starting "+self.date.strftime("%Y-%m-%d")+" image detection process ("+str(len(self.df_image))+")...")

      # attributes
      attributes = self.attributes_extra+[a+"_diff" for a in self.attributes if a != 'cloud']
      
      # show statistics
      print(self.df_image.describe())

      # data
      X = self.scaler.transform(self.df_image[attributes].values.reshape((-1, len(attributes))))

      # classify!
      for i, model in enumerate(self.classifiers):
        start_time = time.time()
        y = self.classifiers[model].predict(X)
        if 'OneClassSVM' in model:
          y = self.decode_ocsvm_labels(y)
        self.df_image['label_'+str(i)] = y
        self.classifiers_runtime[model] = self.classifiers_runtime[model] + (time.time() - start_time)
      
      # warning
      print("finished!")


  # save geojson of pixels classified and spectral indices
  def save_geojsons(self, folder: str):

    # warning
    if isinstance(self.df_image, pd.DataFrame) and not self.df_image.empty:
      print()
      print("Saving GeoJSONs to folder '"+folder+"'...")

      # attributes
      attributes  = [a for a in self.attributes if a != 'cloud']

      # check folder exists
      if not os.path.exists(folder):
        os.mkdir(folder)
      
      # save pixels
      for i, model in enumerate(self.classifiers):
        if 'OneClassSVM' in model:
          model = 'ocsvm'
        elif 'Random Forest' in model:
          model = 'rf'
        elif 'TensorFlow AutoEncoder' in model:
          model = 'ae'
        elif 'Isolation Forest' in model:
          model = 'if'
        features = []
        for index, row in self.df_image.iterrows():
          features.append(geojson.Feature(geometry=geojson.Point((float(row['lat']), float(row['lon']))), properties={"label": int(row['label_'+str(i)]), "cloud": int(row['cloud'])}))
        fc = geojson.FeatureCollection(features)
        f = open(folder+"/"+self.date.strftime("%Y-%m-%d")+"_"+str(model)+".json","w")
        geojson.dump(fc, f)
        f.close()

      # save indices
      for i, attribute in enumerate(attributes):
        features = []
        for index, row in self.df_image.iterrows():
          if attribute in self.attributes_inverse:
            label = self.anomaly if row[attribute]<self.indices_thresholds[attribute] else 0
          else:
            label = self.anomaly if row[attribute]>self.indices_thresholds[attribute] else 0
          features.append(geojson.Feature(geometry=geojson.Point((float(row['lat']), float(row['lon']))), properties={"label": int(label), "cloud": int(row['cloud'])}))
        fc = geojson.FeatureCollection(features)
        f = open(folder+"/"+self.date.strftime("%Y-%m-%d")+"_"+str(attribute)+".json","w")
        geojson.dump(fc, f)
        f.close()

      # warning
      print("finished!")


  # save detection plot
  def save_detection_plot(self, path: str):

    # warning
    if isinstance(self.df_image, pd.DataFrame) and not self.df_image.empty:
      print()
      print("Creating detection plot to file '"+path+"'...")

      # attributes
      attributes  = [a for a in self.attributes if a != 'cloud']

      # configuration
      columns     = len(attributes)+3
      rows        = len(self.classifiers)

      # axis ticks
      xticks      = np.linspace(self.df_image['lat'].min(), self.df_image['lat'].max(), num=4)
      yticks      = np.linspace(self.df_image['lon'].min(), self.df_image['lon'].max(), num=4)

      # matplot fig configuration
      fig         = plt.figure(figsize=(16,8), dpi=300)
      fig.suptitle('Anomalous Behaviour Detection  ('+self.date.strftime("%Y-%m-%d")+')', fontsize=12, y=0.64)
      plt.rc('xtick',labelsize=5)
      plt.rc('ytick',labelsize=5)

      # marker size
      multiplier  = math.ceil(self.scale/100)
      multiplier  = multiplier if multiplier >= 1 else 1
      markersize  = (72./fig.dpi)*multiplier

      # cmap #1
      my_cmap = cm.get_cmap('Reds', 20)
      my_cmap = ListedColormap(my_cmap(np.linspace(0.3, 1, 256)))
      my_cmap.set_under('cyan')

      # cmap #2
      my_cmap2 = cm.get_cmap('Blues', 20)
      my_cmap2 = ListedColormap(my_cmap2(np.linspace(0.3, 1, 256)))
      my_cmap2.set_under('cyan')

      # handles legend
      handles = [Patch(color='cyan', label='Regular'), Patch(color='red', label='Anomaly'), Patch(color='gray', label='Cloud'), Patch(color='white', label='Land/Absence')]

      # normalize indexes
      self.df_image = self.normalize_indices(df=self.df_image)

      # blank image
      imageIO_blank = PIL.Image.open(BytesIO(requests.get(self.clip_image(ee.Image([99999,99999,99999])).select(['constant','constant_1','constant_2']).getThumbUrl({'min':0, 'max':99999, 'dimensions': 500}), timeout=60).content))

      # go through each classifier
      count = 0
      for i, model in enumerate(self.classifiers):

        # show rgb plot
        count += 1
        try:
          image_rgb_clip = self.image_clip.select(self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue']).getThumbUrl({'min':0, 'max':3000, 'dimensions': 500})
          imageIO = PIL.Image.open(BytesIO(requests.get(image_rgb_clip, timeout=600).content))
        except:
          imageIO = imageIO_blank
    
        # RGB
        c = fig.add_subplot(rows,columns,count)
        c.set_title("Natural color [Red, Green, Blue]", fontdict = {'fontsize' : 7})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.imshow(imageIO, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        c.margins(x=0,y=0)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.autoscale(False)

        # show false-color plot
        count += 1
        try:
          image_false_clip = self.image_clip.select(self.sensor_params['nir'], self.sensor_params['red'], self.sensor_params['green']).getThumbUrl({'min':0, 'max':3000, 'dimensions': 500})
          imageIO = PIL.Image.open(BytesIO(requests.get(image_false_clip, timeout=600).content))
        except:
          imageIO = imageIO_blank

        # FALSE-COLOR
        c = fig.add_subplot(rows,columns,count)
        c.set_title("False-color [NIR, Red, Green]", fontdict = {'fontsize' : 7})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.imshow(imageIO, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        c.margins(x=0,y=0)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.autoscale(False)

        # creating spectral indices subplots
        for attribute in attributes:
          count += 1
          c = fig.add_subplot(rows,columns,count)
          c.set_title(attribute.upper(), fontdict = {'fontsize' : 7})
          c.set_xticks([])
          c.set_yticks([])
          s = c.scatter(self.df_image['lat'], self.df_image['lon'], s=markersize, c=self.df_image[attribute], marker='s', cmap=my_cmap2 if attribute in self.attributes_inverse else my_cmap)
          s.set_clim([self.indices_thresholds[attribute], self.df_image[attribute].max()])
          cb = plt.colorbar(s, cax=make_axes_locatable(c).append_axes("right", size="5%", pad=0.1), format=FormatStrFormatter('%.3f'), ticks=np.linspace(self.indices_thresholds[attribute], self.df_image[attribute].max(), num=5))
          cb.ax.tick_params(labelsize=5)
          c.scatter(self.df_image[self.df_image['cloud']==1]['lat'], self.df_image[self.df_image['cloud']==1]['lon'], marker='s', s=markersize, c="gray")
          c.imshow(imageIO_blank, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
          c.margins(x=0,y=0)
          c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          c.autoscale(False)

        # creating detection plot
        count += 1
        c = fig.add_subplot(rows,columns,count)
        c.set_title("Anomalous Behaviour Detection (ABD)", fontdict = {'fontsize' : 7})
        c.set_xticks(xticks)
        c.set_yticks(yticks)
        c.yaxis.tick_right()
        c.scatter(self.df_image[self.df_image['label_'+str(i)]!=self.anomaly]['lat'], self.df_image[self.df_image['label_'+str(i)]!=self.anomaly]['lon'], marker='s', color='cyan', s=markersize)
        c.scatter(self.df_image[self.df_image['label_'+str(i)]==self.anomaly]['lat'], self.df_image[self.df_image['label_'+str(i)]==self.anomaly]['lon'], marker='s', color='red', s=markersize)
        c.scatter(self.df_image[self.df_image['cloud']==1]['lat'], self.df_image[self.df_image['cloud']==1]['lon'], marker='s', color='gray', s=markersize)
        if count >= (columns*rows):
          c.legend(loc='right', bbox_to_anchor=(1.6, 0.5), fancybox=True, shadow=True, ncol=1, fontsize="x-small", handles=handles)
        c.imshow(imageIO_blank, extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])
        c.margins(x=0,y=0)
        c.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        c.autoscale(False)
      
      # save figure
      plt.subplots_adjust(wspace=0.4, hspace=0.4)
      plt.tight_layout()
      fig.savefig(path, bbox_inches='tight')

      # warning
      print("finished!")


  # save timeseries plot
  def save_timeseries_plot(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Creating time series plot to file '"+path+"'...")

    # attributes
    attributes = [a for a in self.attributes if a != 'cloud']

    # creating plot figure
    fig = plt.figure(figsize=(10,6), dpi=300)
    plt.tight_layout()
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    # separe each indice in a plot (vertically)
    i = 0
    for attribute in attributes:
      i += 1
      ax = fig.add_subplot(len(attributes),1,i)
      ax.set_title(attribute.upper()+' Spread')
      ax.set_ylabel(attribute.upper())
      if i < len(attributes):
        ax.set_xlabel('')
        ax.set_xticks([])
      else:
        ax.set_xlabel('Pixels')
        ax.set_xticks((0, math.floor(df['index'].max()/2), df['index'].max()))
      ax.plot(df['index'], df[attribute], linewidth=0.2)

    # save it to file
    fig.savefig(path)

    # warning
    print("finished!")


  # validate imagem using a ROI from GEE
  # GEE Code Console: https://code.earthengine.google.com
  def validate_using_roi(self, path: str,  rois: list, labels: list = [], df_pred: pd.DataFrame = None):

    # warning
    if isinstance(self.df_image, pd.DataFrame) and not self.df_image.empty:
      print()
      print("Validating classifier using a ROI from GEE web plataform (https://code.earthengine.google.com)...")

      # atributes
      df_columns          = ['pixel', 'lat', 'lon', 'label']
      df_true             = pd.DataFrame(columns=df_columns)

      # go over each roi
      for i, roi in enumerate(rois):

        # extract FeatureCollection from roi selected through the GEE Code Console and process image
        print("Processing ROI '"+path+"/"+roi+"' with label '"+str(labels[i])+"'...")

        # get each geometry pixel values
        lons_lats_attributes  = np.array([]).reshape(0, 3)
        try:

          # go through each geometry and get pixels
          for geometry in self.splitted_geometry:
            clip                            = self.clip_image(image=ee.Image(2).toByte().paint(ee.FeatureCollection(path+'/'+roi), labels[i]), geometry=geometry, scale=self.scale)
            geometry_lons_lats_attributes   = gee.extract_latitude_longitude_pixel(image=clip, geometry=geometry, bands=['constant'], scale=self.scale)
            lons_lats_attributes            = np.vstack((lons_lats_attributes, geometry_lons_lats_attributes))

          # concatenate geometry pixel values
          extra_attributes     = np.array(list(zip(range(0,len(lons_lats_attributes)))))
          lons_lats_attributes = np.hstack((extra_attributes, lons_lats_attributes))

          # append new data from roi to dataframe
          df_true = df_true.append(pd.DataFrame(data=lons_lats_attributes,columns=df_columns).infer_objects(), ignore_index = True)

        # error: no data in ROI or it does not exist
        except:
          print("Error while extracting roi pixels: "+str(traceback.format_exc()))

      # fix columns labels
      df_true = df_true.apply(pd.to_numeric, errors='ignore')
      
      # prepare the datas
      df_true = df_true[df_true['label']!=2.0][['pixel', 'lat', 'lon', 'label']]
      df_pred_raw = df_pred.copy(deep=True) if df_pred is not None else self.df_image.copy(deep=True)

      # get rois label list
      array_labels = df_true.groupby('label').count().values[:,0].astype(str)

      # reset results
      dict_results = []

      ########################################################################
      # 1) Classifiers
      
      # go through each model
      for i, model in enumerate(self.classifiers):

        # warning
        print()
        print("Evaluating the "+str(model)+" model...")

        # start counter
        start_time = time.time()

        # prepare model data
        df_pred = df_pred_raw[['lat', 'lon', 'label_'+str(i)]].rename(columns={'label_'+str(i): 'label'})

        # merge true and pred dataframes, finding matched pair of latitude and longitude
        merged = pd.merge(df_true, df_pred, on=['lat','lon'], how='inner', suffixes=('_true', '_pred'))

        # get predictions on test set
        y_true, y_pred  = merged['label_true'].values.astype(int), merged['label_pred'].values.astype(int)
        measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

        # reports
        try:
          print("Report for the "+str(model)+" model: "+measures['string'])
          print(metrics.classification_report(y_true, y_pred))
        except:
          print("Error while extracting classification report!")

        # add results
        dict_results.append({
          'model':                  str(model),
          'date_detection':         self.date.strftime("%Y-%m-%d"),
          'date_execution':         dt.now().strftime("%Y-%m-%d"),
          'time_execution':         dt.now().strftime("%H:%M:%S"),
          'runtime':                str(self.classifiers_runtime[model]),
          'days_threshold':         str(self.days_threshold),
          'size_image':             str(self.sample_total_pixel),
          'size_train':             str(len(self.df_train)),
          'size_dates':             str(len(self.dates_timeseries_interval)),
          'roi':                    str(rois)+","+str(",".join(array_labels)),
          'scaler':                 str(self.scaler_str),
          'remove_outliers':        str(self.remove_outliers),
          'reduce_dimensionality':  str(self.reduce_dimensionality),
          'attribute_doy':          str(self.attribute_doy),
          'morph_op':               str(self.morph_op),
          'morph_op_iters':         str(self.morph_op_iters),
          'convolve':               str(self.convolve),
          'convolve_radius':        str(self.convolve_radius),
          'acc':                    float(measures['acc']),
          'f1score':                float(measures['f1score']),
          'kappa':                  float(measures["kappa"]),
          'vkappa':                 float(measures["vkappa"]),
          'tau':                    float(measures["tau"]),
          'vtau':                   float(measures["vtau"]),
          'mcc':                    float(measures['mcc']),
          'p_value':                float(measures['p_value']),
          'fp':                     int(measures["fp"]),
          'fn':                     int(measures["fn"]),
          'tp':                     int(measures["tp"]),
          'tn':                     int(measures["tn"]),
        })

      ########################################################################


      ########################################################################
      # 1) Indices Thresholds

      # go through each index
      for indice in self.indices_thresholds:

        # check indice exists in attributes array
        if indice in self.attributes:

          # warning
          print()
          print("Evaluating the "+str(indice).upper()+" indice...")

          # start counter
          start_time = time.time()

          # prepare model data
          #df_pred = df_pred_raw[['pixel', indice]].rename(columns={indice: 'label'})
          df_pred = df_pred_raw[['lat', 'lon', indice]].rename(columns={indice: 'label'})
          df_pred.loc[df_pred['label']>self.indices_thresholds[indice], 'label'] = self.anomaly if indice not in self.attributes_inverse else 0
          df_pred.loc[df_pred['label']<=self.indices_thresholds[indice], 'label'] = 0 if indice not in self.attributes_inverse else self.anomaly

          # merge true and pred dataframes, finding matched pair of latitude and longitude
          #merged = pd.merge(df_true, df_pred, on=['pixel'], how='inner', suffixes=('_true', '_pred'))
          merged = pd.merge(df_true, df_pred, on=['lat','lon'], how='inner', suffixes=('_true', '_pred'))

          # get predictions on test set
          y_true, y_pred  = merged['label_true'].values.astype(int), merged['label_pred'].values.astype(int)
          measures        = misc.concordance_measures(metrics.confusion_matrix(y_true, y_pred), y_true, y_pred)

          # endcounter
          end_time = (time.time() - start_time)

          # reports
          try:
            print("Report for the "+str(indice)+" indice: "+measures['string'])
            print(metrics.classification_report(y_true, y_pred))
          except:
            print("Error while extracting classificatio report!")

          # add results
          dict_results.append({
            'model':                  str(indice),
            'date_detection':         self.date.strftime("%Y-%m-%d"),
            'date_execution':         dt.now().strftime("%Y-%m-%d"),
            'time_execution':         dt.now().strftime("%H:%M:%S"),
            'runtime':                str(end_time),
            'days_threshold':         str(self.days_threshold),
            'size_image':             str(self.sample_total_pixel),
            'size_train':             str(len(self.df_train)),
            'size_dates':             str(len(self.dates_timeseries_interval)),
            'roi':                    str(rois)+","+str(",".join(array_labels)),
            'scaler':                 str(self.scaler_str),
            'remove_outliers':        str(self.remove_outliers),
            'reduce_dimensionality':  str(self.reduce_dimensionality),
            'attribute_doy':          str(self.attribute_doy),
            'morph_op':               str(self.morph_op),
            'morph_op_iters':         str(self.morph_op_iters),
            'convolve':               str(self.convolve),
            'convolve_radius':        str(self.convolve_radius),
            'acc':                    float(measures['acc']),
            'f1score':                float(measures['f1score']),
            'kappa':                  float(measures["kappa"]),
            'vkappa':                 float(measures["vkappa"]),
            'tau':                    float(measures["tau"]),
            'vtau':                   float(measures["vtau"]),
            'mcc':                    float(measures['mcc']),
            'p_value':                float(measures['p_value']),
            'fp':                     int(measures["fp"]),
            'fn':                     int(measures["fn"]),
            'tp':                     int(measures["tp"]),
            'tn':                     int(measures["tn"]),
          })

      ########################################################################

      # save results
      self.df_results = pd.DataFrame(dict_results, columns=self.df_columns_results)

      # warning
      print("finished!")


  # save occurrences plot
  def save_occurrences_plot(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Saving occurrences plot to image '"+path+"'...")
    
    # years list
    years_list  = df.groupby('year')['year'].agg('mean').values

    # build date string
    str_date    = str(int(min(years_list))) + ' to ' + str(int(max(years_list)))

    # number of columns
    columns     = 6 if len(years_list)>=6 else len(years_list)
    rows        = math.ceil(len(years_list)/columns)
    fig_height  = 16/columns
  
    # axis ticks
    xticks      = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=4)
    yticks      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=4)

    # colorbar tixks
    colorbar_ticks_max          = 100
    colorbar_ticks              = np.linspace(0, colorbar_ticks_max if colorbar_ticks_max > 1 else 2, num=5, dtype=int)
    colorbar_ticks_labels       = [str(l) for l in colorbar_ticks]
    colorbar_ticks_labels[-1]   = str(colorbar_ticks_labels[-1])

    # create the plot
    fig = plt.figure(figsize=(20,rows*fig_height), dpi=300)
    fig.suptitle('Algal Bloom Yearly Occurrences  ('+str_date+')', fontsize=14, y=1.04)
    fig.autofmt_xdate()
    plt.rc('xtick',labelsize=6)
    plt.rc('ytick',labelsize=6)

     # marker size
    multiplier  = math.ceil(self.scale/100)
    multiplier  = multiplier if multiplier >= 1 else 1
    markersize  = (72./fig.dpi)*multiplier

    # go through each year
    images = []
    for i, year in enumerate(years_list):

      # filter year data
      df_year = df[(df['year'] == year)]

      # add plot
      ax = fig.add_subplot(rows,columns,i+1)
      ax.grid(True, linestyle='dashed', color='#909090', linewidth=0.1)
      ax.title.set_text(str(int(year)))
      s = ax.scatter(df_year['lat'], df_year['lon'], s=markersize, c=df_year['pct_occurrence'], cmap=plt.get_cmap('jet'))
      s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
      ax.margins(x=0,y=0)
      ax.set_xticks(xticks)
      ax.set_yticks(yticks)
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      images.append(s)

    # figure add cmap
    cbar = fig.colorbar(images[-1], cax=fig.add_axes([0.6, -0.05, 0.39, 0.05]), ticks=colorbar_ticks, orientation='horizontal')
    cbar.set_label("% of occurrence")

    # save it to file
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')

    # warning
    print("finished!")


  # save occurrences geojsons
  def save_occurrences_geojson(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Saving occurrences geojsons to file '"+path+"'...")
    
    # years list
    years_list  = df.groupby('year')['year'].agg('mean').values

    # fillna
    df.fillna(0, inplace=True)

    # build features list
    features = []
    for index, row in df.iterrows():
      features.append(geojson.Feature(geometry=geojson.Point((float(row['lat']), float(row['lon']))), properties={"year": int(row['year']), "cloud": int(row['cloud']), "occurrence": int(row['occurrence']), "not_occurrence": int(row['not_occurrence']), "pct_occurrence": int(row['pct_occurrence']), "pct_cloud": int(row['pct_cloud']), "instants": int(row['instants'])}))
    fc = geojson.FeatureCollection(features)
    f = open(path,"w")
    geojson.dump(fc, f)
    f.close()

    # warning
    print("finished!")


  # save a image to file
  def save_image_png(self, image: ee.Image, date: dt, path: str, options: dict = {'min':0, 'max': 3000, 'dimensions': 500}):
    
    # warning
    print()
    print("Saving image to file '"+path+"'...")

    # get sensor name
    image_collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d")))

    # default to RGB bands
    bands = [self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue']]
    if self.sensor_params["sensor"] == "landsat578":
      sensor = image_collection.first().get(self.sensor_params["property_id"]).getInfo()
      if 'LT05' in sensor:
        bands = [gee.get_sensor_params("landsat5")['red'], gee.get_sensor_params("landsat5")['green'], gee.get_sensor_params("landsat5")['blue']]
      elif 'LE07' in sensor:
        bands = [gee.get_sensor_params("landsat7")['red'], gee.get_sensor_params("landsat7")['green'], gee.get_sensor_params("landsat7")['blue']]
      elif 'LC08' in sensor:
        bands = [gee.get_sensor_params("landsat")['red'], gee.get_sensor_params("landsat")['green'], gee.get_sensor_params("landsat")['blue']]

    # extract imagem from GEE using getThumbUrl function and saving it
    try:
      imageIO = PIL.Image.open(BytesIO(requests.get(image.select(bands).getThumbUrl(options), timeout=60).content))
      imageIO.save(path)
    except:
      print("Error while saving png image: "+str(traceback.format_exc()))
    
    # warning
    print("finished!")


  # save a image to file
  def save_image_tiff(self, image: ee.Image, date: dt, path: str, folderName: str, options: dict = {'min':0, 'max': 3000}):
    
    # warning
    print()
    print("Saving image in tiff to file '"+path+"' (first try, based on image size) or to your Google Drive at folder '"+str(folderName)+"'...")

    # attributes
    attributes = [a for a in self.attributes]

    # check if its landsat merge
    bands = [self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue'], self.sensor_params['nir'], self.sensor_params['swir']]+attributes
    if self.sensor_params["sensor"] == "landsat578":
      sensor = image.get(self.sensor_params["property_id"]).getInfo()
      if 'LT05' in sensor:
        bands = [gee.get_sensor_params("landsat5")['red'], gee.get_sensor_params("landsat5")['green'], gee.get_sensor_params("landsat5")['blue'], gee.get_sensor_params("landsat5")['nir'], gee.get_sensor_params("landsat5")['swir']]+attributes
      elif 'LE07' in sensor:
        bands = [gee.get_sensor_params("landsat7")['red'], gee.get_sensor_params("landsat7")['green'], gee.get_sensor_params("landsat7")['blue'], gee.get_sensor_params("landsat7")['nir'], gee.get_sensor_params("landsat7")['swir']]+attributes
      elif 'LC08' in sensor:
        bands = [gee.get_sensor_params("landsat")['red'], gee.get_sensor_params("landsat")['green'], gee.get_sensor_params("landsat")['blue'], gee.get_sensor_params("landsat")['nir'], gee.get_sensor_params("landsat")['swir']]+attributes

    # First try, save in local folder
    try:
      print("Trying to save "+date.strftime("%Y-%m-%d")+" GeoTIFF to local folder...")
      image_download_url = image.select(bands).getDownloadUrl({"name": date.strftime("%Y-%m-%d"), "region":self.geometry, "filePerBand": True})
      open(path, 'wb').write(requests.get(image_download_url, allow_redirects=True).content)
      print("finished!")

    # Second try, save in Google Drive
    except:
      print("Error! It was not possible to save GeoTIFF localy. Trying to save it in Google Drive...")
      for band in bands:
        task = ee.batch.Export.image.toDrive(image=image.select(band), folder=folderName, description=date.strftime("%Y-%m-%d")+"_"+str(band), region=self.geometry)
        task.start()
        print(task.status())
    
    # warning
    print("finished!")


  # save a collection in tiff (zip) to folder (time series)
  def save_collection_tiff(self, folder: str, folderName: str):

    # build Google Drive folder name where tiffs will be saved in
    folderName = "abd_"+str(folderName)+".tiff"
    
    # warning
    print()
    print("Saving image collection in tiff to folder '"+str(folder)+"' (first try, based on image size) or to your Google Drive at folder '"+str(folderName)+"'...")

    # check if folder exists
    if not os.path.exists(folder):
      os.mkdir(folder)

    # go through all the collection
    for date in self.dates_timeseries_interval:
      image = self.clip_image(self.extract_image_from_collection(date=date), geometry=self.geometry)
      self.save_image_tiff(image=image, date=date, path=folder+'/'+date.strftime("%Y-%m-%d")+'.zip', folderName=folderName)

    # warning
    print("finished!")
  

  # save a collection in png to folder (time series)
  def save_collection_png(self, folder: str):
    
    # warning
    print()
    print("Saving image collection in png to folder '"+folder+"'...")

    # check if folder exists
    if not os.path.exists(folder):
      os.mkdir(folder)

    # go through all the collection
    for date in self.dates_timeseries_interval:

      # check if folder exists
      path_image = folder+'/'+date.strftime("%Y-%m-%d")
      if not os.path.exists(path_image):
        os.mkdir(path_image)

      # save geometries in folder
      for i, geometry in enumerate(self.splitted_geometry):
        self.save_image_png(image=self.clip_image(self.extract_image_from_collection(date=date), geometry=geometry), date=date, path=path_image+"/"+date.strftime("%Y-%m-%d")+"_"+str(i)+".png")
    
    # warning
    print("finished!")


  # save a dataset to file
  def save_dataset(self, df: pd.DataFrame, path: str):
    
    # warning
    if not df is None and not df.empty:
      print()
      print("Saving dataset to file '"+path+"'...")

      # saving dataset to file
      df.to_csv(r''+path)
      
      # warning
      print("finished!")
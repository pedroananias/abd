# Anomalous Behaviour Detection

Module responsable for detecting anomalies in case studies of algal bloom occurences based on images from Google Earth Engine API and machine learning



### Dependencies

- Python 3.7.7 64-bit ou superior
- Modules: oauth2client earthengine-api matplotlib pandas numpy requests pillow natsort geojson argparse logging joblib



### Attention, before running:

Before running the script and after installing the libraries, you must authenticate with the Google Earth Engine API using the following command:

```
earthengine authenticate
```



### How to execute the default script?

python /path/to/abd/script.py --lat_lon=-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091 --dates=2019-06-03 --name=erie --model=ocsvm --sensor=modis




### What are the results?

The script will detect the occurrence of anomalies with a case study of algae blooming in the inserted study area and selected date. Therefore, a folder located in 'data' is created and named based on the date and version of the script executed. Example: /path/to/abd/data/20201122_123014[v=V26-erie,d=2019-06-03,t=180,m=ocsvm,s=modis].

The following results are generated:

- geojson (GeoJSONs with pixels and labels)
- 2019-06-03.png (RGB image)
- 2017-12-31.zip (Bands as TIFFs files)
- detection.png (Graph of the detection result)
- results.csv (Accuracy results of validation process in CSV format -if a ROI located in GEE plataform was selected)
- timeseries.png (Graphical visualization of the built timeseries)


### Exporting GeoTIFFs to Google Drive

When using the 'save_collection' function, if the script can not save images locally, will send them to Google Drive to a folder called 'abd_name.tiff' for user who is authenticated.



### Example

```
# Import
import ee
from modules import abd

# Initialize Google Earth Engine
ee.Initialize()

# folder where to save results
folder = "/path/to/desired/folder"

# create algorithm object
algorithm = abd.Abd(lat_lon="-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091,
                    date="2019-06-03",
                    days_threshold=180,
                    model="ocsvm",
                    sensor="modis")

# creating timeseries based on days_thresholds
algorithm.process_timeseries_data()

# creating train and grid datasets
algorithm.process_training_data(df=algorithm.df_timeseries)

# start training process
algorithm.train()

# apply detection algorithm
algorithm.detect()

# validate using ROI - example (https://code.earthengine.google.com/)
algorithm.validate_using_roi(path='users/gee-user-name/erie', rois=['2019-06-03_modis_regular', '2019-06-03_modis_anomaly'], labels=[0, 1])

# save geojson
algorithm.save_geojsons(folder=folder+"/geojson")

# save results
algorithm.save_dataset(df=algorithm.df_results, path=folder+'/results.csv')

# save collection images
algorithm.save_timeseries_plot(df=algorithm.df_timeseries, path=folder+'/timeseries.png')
if isinstance(algorithm.df_image, pd.DataFrame) and not algorithm.df_image.empty:
    algorithm.save_detection_plot(path=folder+'/detection.png')
    algorithm.save_image_png(image=algorithm.image_clip, date=dt.strptime(date, "%Y-%m-%d"), path=folder+"/"+date+".png")
    algorithm.save_image_tiff(image=algorithm.image_clip, date=dt.strptime(date, "%Y-%m-%d"), path=folder+"/"+date+".zip", folderName=args.name)
    
# save collection images (tiff and png)
if args.save_collection:
    algorithm.save_collection_tiff(folder=folder+"/tiff", folderName="abd_erie.tiff")
    algorithm.save_collection_png(folder=folder+"/png")

# save preprocessing results
if args.save_train:
    algorithm.save_dataset(df=algorithm.df_timeseries, path=folder+'/timeseries.csv')
    algorithm.save_dataset(df=algorithm.df_train, path=folder+'/df_train.csv')
    algorithm.save_dataset(df=algorithm.df_gridsearch, path=folder+'/df_gridsearch.csv')
    algorithm.save_dataset(df=algorithm.df_image, path=folder+'/df_image.csv')
```
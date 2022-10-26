# Anomalous Behaviour Detection

Module responsable for detecting anomalies in case studies of algal bloom occurences based on images from Google Earth Engine API and machine learning



## Dependencies

- Python >= 3.7.7 64-bit, < 3.10 64-bit
- Google Earth Engine enabled account: see https://earthengine.google.com/


## Instalation

To install this script and all its dependencies, execute the follow commands:

1) Create a virtual environment: `python3 -m venv venv`
2) Enable it: `source venv/bin/activate`
2) Install the script dependencies: `pip install -e .`


## Attention, before running this script:

Before running the script and after installing the libraries, you must authenticate with the Google Earth Engine API using ONE of the following commands:

```bash
# from local command line
earthengine authenticate

# from inside Docker container
earthengine authenticate --auth_mode=notebook

# from inside Jupyter Notebook
import ee
ee.Authenticate()
```

In some versions of macOS, it might be necessary to run this command using `sudo`.

Additionally, make sure that folders `cache` and `results` have writing permissions:

```bash
chmod 777 /path/to/abd/cache
chmod 777 /path/to/abd/results
```

## Docker image

There is also a Docker image which provides this script with all necessary dependencies easy and ready. To use it, run:

```bash
docker run -p 8888:8888 phmananias/abd:latest
```

or you can build it locally and then run it:
```bash
docker build -t abd:latest .
docker run -p 8888:8888 abd:latest abd
```



## Command line tool

This module brings a default command line `adb` for you. To see available parameters, please run:

```bash
> abd --help
Usage: abd [OPTIONS]

Options:
  --lat_lon TEXT            Two diagnal points (Latitude 1, Longitude 1,
                            Latitude 2, Longitude 2) of the study area
  --dates TEXT              Comma-separated date to be applied the algorithm
  --name TEXT               Place where to save generated files
  --days_threshold INTEGER  Days threshold used to build the timeseries and
                            training set: 90, 180 ou 365
  --model TEXT              Select the desired model: ocsvm, rf, if or None
                            for all
  --sensor TEXT             Define the selected sensor where images will be
                            downloaded from: landsat, sentinel, modis
  --save_collection TEXT    Save collection images (tiff and png)
  --save_train TEXT         Enable saving the training dataset (csv)
  --force_cache TEXT        Force cache resetting to prevent image errors
  --attributes TEXT         Define the attributes used in the modelling
                            process
  --outliers_zscore FLOAT   Define the Z-Score used in the median outlier
                            removal threshold
  --attribute_doy TEXT      Define if the doy attribute will be used in the
                            modelling process
  --roi TEXT                Select the GEE roi version used in the validation
                            process. E.g. users/pedroananias/roi
  --cloud_threshold TEXT    Allow CLOUD threshold customization by user
  --output_folder           Specify desired results output folder
  --help                    Show this message and exit.

```


## How to execute the default script?

```bash
abd --lat_lon=-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091 --dates=2019-06-03 --name=erie --model=ocsvm --sensor=modis --output_folder=abd
```


## What are the results?

The script will detect the occurrence of anomalies with a case study of algae blooming in the inserted study area and selected date. Therefore, a folder located in `results` is created and named based on the date and version of the script executed. Example: 

```bash
/path/to/abd/results/20221023_133118[v=v0.26.0-erie,d=2019-06-03,t=180,m=ocsvm,s=modis,attr=ndvi,fai]
```

The following results are generated:

- geojson (GeoJSONs with pixels and labels)
- 2019-06-03.png (RGB image)
- 2017-12-31.zip (Bands as TIFFs files)
- detection.png (Graph of the detection result)
- results.csv (Accuracy results of validation process in CSV format -if a ROI located in GEE plataform was selected)
- timeseries.png (Graphical visualization of the built timeseries)


### Exporting GeoTIFFs to Google Drive

When using the 'save_collection' function, if the script can not save images locally, will send them to Google Drive to a folder called 'abd_name.tiff' for user who is authenticated.



## Sandbox example

This script comes with a Jupyter Notebook sandbox example file. To open it, please run the command below inside the script's root directory:

```bash
jupyter-lab
```
#!/bin/bash 

# THIS DIR
BASEDIR="$( cd "$( dirname "$0" )" && pwd )"

# ARGUMENTS GENERAL
PYTHON=${1:-"python"}
SCRIPT="script.py"

# SHOW BASE DIR
echo "$PYTHON $BASEDIR/$SCRIPT"

############################################################################################
## LAKE ERIE (MODIS)

# ARGUMENTS
NAME="erie"
SENSOR="modis"
DATES="2019-06-03,2019-07-01,2019-08-19,2019-09-24"
LAT_LON="-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091"
MODELS="ocsvm,rf"

# EXECUTIONS
for model in $(echo $MODELS | tr "," "\n")
do
	for date in $(echo $DATES | tr "," "\n")
	do
		for day_threshold in 180
		do
			eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --dates=$date --name=$NAME --days_threshold=$day_threshold --model=$model --sensor=$SENSOR"
		done
	done
done

############################################################################################


# ############################################################################################
# ## LAKE ERIE (LANDSAT)

# ARGUMENTS
NAME="erie"
SENSOR="landsat"
DATES="2015-09-21"
LAT_LON="-83.50124371805877,41.88435023280987,-83.07548096199702,41.65275061592091"
MODELS="ocsvm,rf"

# EXECUTIONS
for model in $(echo $MODELS | tr "," "\n")
do
	for date in $(echo $DATES | tr "," "\n")
	do
		for day_threshold in 180
		do
			eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --dates=$date --name=$NAME --days_threshold=$day_threshold --model=$model --sensor=$SENSOR"
		done
	done
done

############################################################################################


############################################################################################
## LAKE TAIHU (LANDSAT)

# ARGUMENTS
NAME="taihu"
SENSOR="landsat"
DATES="2016-09-13,2016-12-02"
LAT_LON="119.88067005824256,31.273892900245198,120.12089092261887,31.125992693422525"
MODELS="ocsvm,rf"

# EXECUTIONS
for model in $(echo $MODELS | tr "," "\n")
do
	for date in $(echo $DATES | tr "," "\n")
	do
		for day_threshold in 180
		do
			eval "$PYTHON $BASEDIR/$SCRIPT --lat_lon=$LAT_LON --dates=$date --name=$NAME --days_threshold=$day_threshold --model=$model --sensor=$SENSOR"
		done
	done
done

############################################################################################
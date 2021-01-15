# MR Multi-Resident Tracking

The MR multi-resident tracking algorithm processes ambient sensors collected
in a smart environment.  MR uses motion sensor readings to track residents
through the space. The number of residents is a specified constant. Based on the
tracking information, the input file of sensor readings is split into n files,
one per resident. The output files can be used to recognize activities and
understand behavior patterns for individuals within the group of environment
residents.

To perform multi-resident tracking, the data are first processed to map the
sensors to a multi-dimensional latent space. In the latent space, a pair of
sensors are close to each other if the pair of sensors appear consecutively
relatively frequently in the input data. Sensor readings are assigned to the
same resident if they are close to each other in the latent space.

Author: Dr. Diane J. Cook, School of Electrical Engineering and
Computer Science, Washington State University, email: djcook@wsu.edu.

Support: This material is based on upon work supported by the National Science
Foundation under Grant No. 3820-5484 and by the National Institutes of Health
under Grant No. R01AG065218.


# Running MR

MR requires packages `tensorflow`, `keras`, `numpy`, `matplotlib`, and `scipy`.
To install the packages for your environment, run:
```
pip install tensorflow keras numpy matplotlib scipy
```

MR is run using the following command-line format (requires Python 3):

```
python mr.py [options]
```

The options, input file format, and output are described below.


# Options

The following AL options are currently available. If not provided, each option will be set 
to its default value.

```
--num_residents <num>
```

Specify the number of residents that are in the environment. Based on the
input data file, `<num>` output files will be created containing sensor readings,
one per specified resident. The default value is 2.

```
--save_model
```

Specify that the embeddings model should be saved to a file for later use.
This is the model that maps sensors to the multidimensional latent space.
By default the value is False, meaning the model is not saved. If the model
is saved, the saved location is in a 'model' directory.

```
--load_model
```

Specify that the embeddings model should be loaded from a saved file
in a 'model' directory.  By default the value is False, meaning the model is
not loaded from a file but is instead created from the input data.

```
--plot_embeddings
```

Specify that the generated embeddings are visualized by a 3D graph and
an animated 3D graph. The visualizations are stored in a 'plots' directory.

```
--data <filename>
```
Specify `filename` containing the input sequence of ambient sensor readings.
The default filename is 'data'.

# Input File(s)

The input file(s) contains time-stamped sensor readings. An example is in the
file data. Each line of the input file contains a reading for
a single ambient sensor.
```
2016-12-01 00:01:41.858318 DiningRoomAArea ON Other_Activity
2016-12-01 00:01:42.939997 DiningRoomAArea OFF Other_Activity
2016-12-01 00:01:50.972779 KitchenADiningChair ON Other_Activity
2016-12-01 00:01:51.009226 DiningRoomAArea ON Other_Activity
2016-12-01 00:01:55.104980 KitchenADiningChair OFF Other_Activity
2016-12-01 00:01:55.169992 KitchenAArea ON Other_Activity
2016-12-01 00:01:56.620043 DiningRoomAArea OFF Other_Activity
```

The general format for the data contains 5 fields per line. The fields are:

* date: `yyyy-mm-dd`
* time: `hh:mm:ss.ms`
* sensor name
* sensor reading
* label: this is a string indicating the activity label

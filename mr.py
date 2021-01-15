#!/usr/bin/python

# python mr.py <option>+
#
# Performs multi-resident separation of a given file into
# one file per resident.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2021. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import os
from datetime import datetime

import numpy as np
from scipy.spatial.distance import mahalanobis

import config
import create_embeddings


def get_datetime(date, time):
    """ Convert a pair of date and time strings to a datetime structure.
    """
    dtstr = date + ' ' + time
    try:
        dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S.%f")
    except:
        dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S")
    return dt


def find_sensor(cf, sensor):
    """ Return the index of a specific sensor name in the list of sensors.
        If the name is not found in the list, add it to the list.
    """
    try:
        i = cf.sensornames.index(sensor)
        return i
    except:
        print("   adding sensor ", sensor)
        cf.sensornames.append(sensor)
        return len(cf.sensornames) - 1


def read_data(cf):
    """ Read the data file containing timestamped sensor readings.
    """
    datafile = open(cf.data_filename, "r")
    for line in datafile:
        words = line.split()
        date = words[0]
        time = words[1]
        sensorid1 = words[2]
        sensorid2 = words[3]
        message = words[4]
        if len(words) > 5:
            alabel = words[5]
        else:
            alabel = ""
        dt = get_datetime(date, time)
        snum = find_sensor(cf, sensorid1)
        cf.data.append([dt, snum, sensorid2, message, alabel])
        cf.sequence.append(snum)
    datafile.close()


def distance(p1, p2):
    """ Compute Euclidean distance between two points.
    """
    return np.sqrt(((p1[0] - p2[0]) * (p1[0] - p2[0])) +
                   ((p1[1] - p2[1]) * (p1[1] - p2[1])) +
                   ((p1[2] - p2[2]) * (p1[2] - p2[2])))


def compute_distance(stream1, stream2):
    """ Calculate the Mahalanobis distance between feature vectors for
    two streams.
    """
    cv = np.cov(np.array([stream1, stream2]).T)  # covariance matrix
    icv = np.linalg.pinv(cv)  # inverse of the covariance matrix
    d = mahalanobis(stream1, stream2, icv)
    return d


def extract_features(cf, stream, num_days):
    """ Extract vectors of features for a list of streams.
    Features include the number of readings, the location of the first reading
    of the day, and the normalized location counts by time of day
    (four time ranges).
    """
    k = len(cf.sensornames) * 4 + 2
    feature_list = []
    features = np.zeros((k))
    n = len(stream)
    features[0] = n  # number of readings
    features[1] = cf.data[0][1]  # location of first reading of day
    for i in stream:  # extract features for each resident
        reading = cf.data[i]
        dt = reading[0]
        loc = reading[1]
        index = (dt.hour % 6) * loc + 2
        features[loc + 2] += 1
    for i in range(2, k):  # normalize location counts
        features[i] = float(features[i]) / float(n)
    if num_days > 1:
        features[0] /= num_days
    return features


def align_day_streams(cf, day_streams):
    """ Merge streams from multiple days into one set of streams.
    """
    streams = day_streams[0]
    num_days = []
    for i in range(len(streams)):
        num_days.append(0)
    for day in range(1, len(day_streams)):  # merge each day into overall streams
        stream_features = []
        for r in range(len(streams)):
            num_days[r] += 1
            stream_features.append(extract_features(cf, streams[r], num_days[r]))
        day_features = []
        day_data = day_streams[day]
        for r in range(len(day_data)):
            day_features.append(extract_features(cf, day_data[r], 1))
        assignments = []
        for i in range(len(day_features)):  # consider each stream in current day
            min_value = -1
            min_index = -1
            for j in range(len(stream_features)):
                if j not in assignments:
                    d = compute_distance(day_features[i], stream_features[j])
                    if min_value == -1 or d < min_value:
                        min_value = d
                        min_index = j
            if min_index == -1:  # add new stream to list
                streams.append(day_data[i])
                num_days.append(0)
            else:
                assignments.append(min_index)
                streams[min_index].extend(day_data[i])
                num_days[min_index] += 1
    return streams


def assign_reading(cf, streams, index, prevdt, embeddings):
    """ Assign reading to an existing or new stream.
    """
    reading = cf.data[index]  # reading that corresponds to the input index
    dt = reading[0]  # current date and time
    if reading[1] > len(embeddings):
        loc = embeddings[len(embeddings) - 1]
    else:
        loc = embeddings[reading[1]]  # current location in latent space
    close_streams = []
    for i in range(len(streams)):  # fit point to current streams
        n = len(streams[i])  # number of current resident streams
        last_reading = cf.data[streams[i][n - 1]]
        prevloc = embeddings[last_reading[1]]
        d = distance(prevloc, loc)  # distance from previous to current locations
        timediff = reading[0] - prevdt  # time delay between readings
        if i == 0:  # determine which stream is closest to reading
            min_distance = d
            min_index = 0
            td = timediff
        else:
            if d < min_distance:
                min_distance = d
                min_index = i
                td = timediff
        if d == 0.0 and i != min_index:  # reading belongs to more than one resident
            close_streams.append(i)
    if td.total_seconds() <= cf.time_threshold and \
            min_distance <= cf.distance_threshold:
        streams[min_index].append(index)
        for j in close_streams:  # reading in more than one stream
            if j != min_index:
                streams[min_index].append(index)
    elif len(streams) == cf.num_residents:  # cannot create another stream
        streams[min_index].append(index)
        for j in close_streams:  # reading in more than one stream
            if j != min_index:
                streams[min_index].append(index)
    else:  # create a new stream, assign there
        newstream = [index]
        streams.append(newstream)
    return streams


def assign_day(cf, streams, begin, end, dt, embeddings):
    """ Assign one day of readings to one stream per resident.
    """
    streams = []
    stream = [begin]
    streams.append(stream)
    for i in range(begin + 1, end):
        streams = assign_reading(cf, streams, i, dt, embeddings)
        dt = cf.data[i][0]
    return streams


def assign_data(cf, embeddings):
    """ Partition data into one day sequences before assigning readings
    for each day to streams. A stream is maintained as a set of indices
    into the original data array.
    """
    streams = []
    day_begin = 0
    day_end = 0
    current_date = cf.data[day_begin][0].date()
    for i in range(0, len(cf.data)):
        new_date = cf.data[i][0].date()
        if current_date != new_date:  # new day
            day_end = i
            day_streams = assign_day(cf, streams, day_begin, day_end,
                                     cf.data[day_begin][0], embeddings)
            streams.append(day_streams)
            day_begin = i
        current_date = cf.data[day_end][0].date()
    if len(cf.data) > (day_begin + 1):
        day_streams = assign_day(cf, streams, day_begin, len(cf.data),
                                 cf.data[day_begin][0], embeddings)
        streams.append(day_streams)
    return streams


def create_streams(cf, embeddings):
    """ Create files corresponding to data substreams.
    """
    files = []
    for i in range(cf.num_residents):
        outfile = open(cf.data_filename + ".r" + str(i + 1), "w")
        files.append(outfile)
    day_streams = assign_data(cf, embeddings)  # Assign data to streams for each day
    streams = align_day_streams(cf, day_streams)  # Merge days into streams
    for i in range(len(streams)):
        stream = streams[i]
        for j in range(len(stream)):
            reading = cf.data[stream[j]]
            outstr = reading[0].strftime("%Y-%m-%d %H:%M:%S.%f")
            outstr += ' ' + cf.sensornames[reading[1]] + ' '
            outstr += reading[2] + ' ' + reading[3] + ' ' + reading[4] + '\n'
            files[i].write(outstr)
    for i in range(cf.num_residents):
        files[i].close()


def mean_distance(embeddings):
    """ Compute the mean distance in the latent space
    among all pairs of sensors in the environment.
    """
    count = 0
    total = 0
    n = len(embeddings)
    distances = np.zeros((n * (n - 1)))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                distances[count] = distance(embeddings[i], embeddings[j])
                count += 1
    mean_distance_val = np.mean(distances)
    stdev = np.std(distances)
    return mean_distance_val - stdev


def main():
    cf = config.Config()
    cf.set_parameters()
    read_data(cf)
    if len(cf.data) < 500 or len(cf.sensornames) < 10:
        print("Sample too small, assign to one resident.")
        exit()
    if cf.load_model:
        print('Using saved model')
        embeddings_dir = 'model'
        embeddings_filename = os.path.join(embeddings_dir, 'embeddings.npy')
        embeddings = np.load(embeddings_filename)
    else:
        print('Building new model.')
        embeddings = create_embeddings.generate_sensor_embeddings(cf)
    if cf.plot_embeddings:
        create_embeddings.plot_embeddings(embeddings, cf)
    md = mean_distance(embeddings)
    cf.distance_threshold = md
    create_streams(cf, embeddings)
    print('Created streams')


if __name__ == "__main__":
    main()

import gzip
import numpy as np
import json
import sys

def json_to_text(path):
    name = path[:path.find('.')]
    try:
        fdata = open(path)
    except FileNotFoundError:
        sys.exit('amazon json file not found in ' + path + '.')
    fwrite = open(name+'.txt', 'w')
    lines = fdata.readlines()
    for i, line in enumerate(lines):
        fwrite.write(json.loads(line)['reviewText'])
        if i != len(lines) - 1:
            fwrite.write('\n')

def read_amazon_data(data_path, num_of_lines):
    try:
        fdata = open(data_path, 'r')
    except FileNotFoundError:
        sys.exit('amazon data file not found in ' + data_path + '.')
    return fdata.readlines()[:num_of_lines]

def read_sst_data(data_path):
    try:
        fdata = open(data_path, 'r')
    except FileNotFoundError:
        sys.exit('sst data file not found ' + data_path + '.')
    return fdata.readlines()

def read_sst_label(label_path):
    try:
        flabel = open(label_path, 'r')
    except FileNotFoundError:
        sys.exit('sst label file not found ' + label_path + '.')
    y = flabel.readlines()
    return np.array([int(l) for l in y])

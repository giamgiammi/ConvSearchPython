"""
Initialize common objects, logs, pyterrier and load configuration.

# Configuration
The configuration is loaded from a file named 'conf.ini' inside the current directory at execution.

It's structured in 5 sections.

## Sections

### General

threads 
:   number of threads to use during index creation. This will not affect run parallelization.
    Note that if you need a deterministic index it should be set to 1
    
buffer_size_bytes
:   number of bytes for buffering dataset files during index creation

workdir
:    path to the work directory. Here's where run will be saved

cache_dir
:   path to the directory where cache can be saved. Not setting disable cache


### MsMarco
These value are needed to load the msmarco dataset

collection
:   path to collection tsv file

duplicates
:   path to duplicates txt file


### Trec CAR
These value are needed to load the Trec CAR dataset

collection
:   path to collection cbor file


### Terrier
These values are used to setup terrier

JAVA_HOME
:   path to the java JRE home directory to use for terrier. You are required to provide this 
    (except in a singularity environment)

version
:   pyterrier version to load

### Indexes
Here are defined the indexes to use with this application.
For every index_name:

index_name.path
:   The path to the folder where the index resides. This parameter is **required**

For every property that you want to set on the index use:

index_name.prop_name
:   Value of the property 'prop_name' of index 'index_name'

You are free to set no properties, the pyterrier defaults will be used.

## Example
<details>
    <summary>config.ini</summary>
    
```
[GENERAL]
threads = 1
buffer_size_bytes = 10240
workdir = workdir
cache_dir = workdir/cache

[MSMARCO]
collection = /home/user/Datasets/cast2019/MSMARCO/collection.tsv
duplicates = /home/user/Datasets/cast2019/duplicate_list_v1.0.txt

[TREC_CAR]
collection=/home/user/Datasets/cast2019/TREC_CAR/paragraphCorpus/dedup.articles-paragraphs.cbor

[TERRIER]
JAVA_HOME=/lib/jvm/default
version=5.5-custom

[INDEXES]
custom = /home/user/Indexes/myCastIndex
custom.stopwords.filename = ./indri-stopwords.txt
custom.termpipelines = Stopwords,KrovetzStemmer

default.path = /home/user/Indexes/DefaultCastIndex

lucene_like.path = /home/user/Indexes/LuceneLikeCastIndex
lucene_like.stopwords.filename = ./lucene_stopwords.txt
lucene_like.termpipelines = Stopwords,PorterStemmer
```
</details>
# Singularity environment
If a singularity environment (by checking if the environment variable 'SINGULARITY_ENVIRONMENT' is set)
is detected then some extra passage are executed:

- It searches for an additional file 'conf_singularity.ini' and add it to the conf (if exists)
overriding already presents value (make easier to have a different configuration for singularity)
- If the environment variable 'JAVA_HOME' is set, then its value is used instead of the one in conf.ini

# Usage
```
# load pyterrier
import convSearchPython.basics

# obtain main config
from convSearchPython.basics import conf  # conf is of type `configparser.ConfigParser`
```
"""
import logging
from configparser import ConfigParser
from os import environ
from pathlib import Path
from sys import stdout

import pyterrier

conf_file = 'config.ini'
if not Path(conf_file).exists():
    raise Exception(f'config file {conf_file} does not exist. Normally this happen if the current directory is not '
                    'the project top level')
conf = ConfigParser()
conf.read(conf_file)

QUIET = bool(environ.get('WORKER_PROCESS'))

LOG_FORMAT = '%(asctime)s %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=LOG_FORMAT, stream=stdout)
logging.root.setLevel(logging.INFO)

SINGULARITY = bool(environ.get('SINGULARITY_ENVIRONMENT'))
if SINGULARITY:
    if not QUIET:
        print('Detected singularity environment')
    if Path('conf_singularity.ini').exists():
        conf.read('conf_singularity.ini')
        if not QUIET:
            print('added conf_singularity.ini to conf')
    if bool(environ['JAVA_HOME']):
        if not QUIET:
            print(f'using default JAVA_HOME: {environ["JAVA_HOME"]}')
    else:
        environ['JAVA_HOME'] = conf.get('TERRIER', 'JAVA_HOME')
else:
    environ['JAVA_HOME'] = conf.get('TERRIER', 'JAVA_HOME')


if not QUIET:
    print('###############       LOADED CONF     #######################')
    for section in conf.sections():
        print('# [{}]'.format(section))
        for k, v in conf[section].items():
            print('#  {} = {}'.format(k, v))
    print('#############################################################')


if not pyterrier.started():
    pyterrier.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"],
                   version=conf.get("TERRIER", "version"), mem=environ.get('HEAP'))
    if not pyterrier.started():
        raise Exception('something\'s wrong with pyterrier init')


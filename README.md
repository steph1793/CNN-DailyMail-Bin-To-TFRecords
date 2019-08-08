# CNN-DailyMail-Bin-To-TFRecords

The CNN Daily MAil Dataset available online is stored in binary files. We have to use old tensorflow APIs to read and parse the artciles and abstract stores in them, and manage the threads used for this task, which can be unefficient.
With the release of tf-2.0.0, tensorflow proposes new APIs to efficiently manage datasets and build strong data pipelines.
In this context, it is interesting to transform the .bin files into .tfrecords files. When using tf.data API to create dataset it is much more easier (even more suitable) to deal directly with .tfrecods files.

To transform CNN Dailymail .bin chunk files into .tfrecords, follow the next steps.

## - Download the Dataset:
With this link (https://drive.google.com/file/d/0BzQ6rtO2VN95a0c3TlZCWkl3aU0/view?usp=sharing), you can download the preprocessed dataset compressed into a zip file.

## - Unzip the Downloaded file

## - Launch this project
With the following command launch the project to transform your bin files into tfrecords files
python transform.py --data_path *<the_contaning_the_bins_to_be_transformed>* --new_data_path *<the_new_folder_for_tfrecods_files>*


## Requirements
- ntpath
- os
- glob
- struct
- tensorflow >= 1.14
- argparse

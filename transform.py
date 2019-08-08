import ntpath
import os
import glob
import struct
import tensorflow as tf # tensorflow 1.14, 2.0.0-alpha, 2.0.0-beta1
from tensorflow.core.example import example_pb2
import argparse

def example_generator(file):
	"""
		This is a generator that yields parsed exampled_pb2 objects contaning an article and an abstract
		args : 
			file : .bin file from which the generator extracts the example_pb2 objects
	"""
  
	while True:
		len_bytes = file.read(8)
		if not len_bytes: break # finished reading this file
		str_len = struct.unpack('q', len_bytes)[0]
		example_str = struct.unpack('%ds' % str_len, file.read(str_len))[0]
		yield example_pb2.Example.FromString(example_str)



def art_abs_example(article, abstract):
  	"""
		Builds a tf.train.Example object from an article and an abstract
		args:	
			article : string bytes 
			abstract : string bytes
  	"""

	def _bytes_feature(value):
		"""Returns a bytes_list from a string / byte."""
		if isinstance(value, type(tf.constant(0))):
  			value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

	feature = {
	  'article': _bytes_feature(article),
	  'abstract': _bytes_feature(abstract)
	}

	return tf.train.Example(features=tf.train.Features(feature=feature))



def make_TFRecords(data_path, new_data_path):

	"""
		This methods tranforms the CNN dailymail bin files into TFRecords files. 
		It works recursively by transforming even bin files stores in subfolders of data_path
		args:
			data_path : The folder in which are stored the bin files
			new_data_path : The new folder where to store the tfrecords files
	"""

	print("Starting ...")
  	if not os.path.exists(new_data_path): # if new_data_ath doesn't exist, we create it
    	os.makedirs(new_data_path)

  	filelist = glob.glob(data_path+"/**/*.bin", recursive=True) # get the list of datafiles, even those in subfolders
  	assert filelist, "No binary files"
  
  	# The next two lines allows us to extract the names of the files in data_path and the names of the files in subfolders of data_path
  	# (with the subfolder attached to the latter). example : [test000.bin, test001.bin, subfold/test000.bin , ...]
  	common_path = os.path.commonpath(filelist)
  	files = [os.path.splitext(x.replace(common_path, ""))[0]  for x in filelist]
  
  	# WE iterate over each binary file
  	for f, filename in zip(filelist, files):
    	try:
      		file =  open(f, 'rb')
		except:
      		print("Cannot open file : {}".format(f))
      		continue
     
     	# we build the name of the new tfrecord file
    	record_file = '{}/{}.tfrecords'.format(new_data_path, filename)

    	#For the bin files stored in subfolders, we make sure to create equivalent subfolders in the new_data_path
    	record_dir = os.path.dirname(record_file)
    	if not os.path.exists(record_dir):
      		os.makedirs(record_dir)

  		# Create a writer for the current bin file
    	with tf.io.TFRecordWriter(record_file) as writer:
    		# We get all the (article, abstract ) pairs stored in the bin file one by one . The pair is an example_pb2 object
      		for e in example_generator(file):
        		try:
        			# Article and abstract extraction
          			article_text = e.features.feature['article'].bytes_list.value[0].decode()
          			abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode()
          
          			# Build tf.Train.Example object and write it in the current tfrecord file
          			tf_example = art_abs_example(article_text, abstract_text)
          			writer.write(tf_example.SerializeToString())

        		except ValueError:
          			tf.logging.error('Failed to get article or abstract from example')
          			continue
        		if len(article_text) == 0   :
          			tf.logging.warning('Found an example with empty article text. Skipping it.')
          
    	print("Chunked file {} processed and saved to {}".format(f, record_file))



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", help="The folder containing the binary files")
	parser.add_argument("--new_data_path", help="The new folder where to store the tfrecords files")
	args = parser.parse_args()

	assert parser.data_path, "Data path needed"
	assert parser.new_data_path, "New data path needed"

	make_TFRecords(parser.data_path, parser.new_data_path)
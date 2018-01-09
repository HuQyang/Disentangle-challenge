import re, os
import tensorflow as tf
import numpy as np
import glob

def get_files(folder, file_regexp):
  files = []
  for root, dirnames, filenames in os.walk(folder):
      for filename in filenames:
        if re.match(file_regexp, filename):
          files.append(os.path.join(root, filename))
  return sorted(files)

def get_files_cached(folder, type, regexp, reload=False):
  f = os.path.join(folder, '{}_files.txt'.format(type))
  if os.path.isfile(f) and not reload:
    return [filename.strip() for filename in open(f).readlines()]
  else:
    filelist = get_files(folder, regexp)
    with open(f, 'w') as file_write:
      file_write.write('\n'.join(filelist))
    return filelist


def get_rendered_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'rendered', '.*r_\d{3}_' + size_suffix + '\.png$')


def get_albedo_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'albedo', '.*r_\d{3}_albedo\.png0001_' + size_suffix + '\.png$')


def get_normal_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'normal', '.*r_\d{3}_normal\.png0001_' + size_suffix + '\.png$')


def get_depth_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'depth', '.*r_\d{3}_depth\.png0001_' + size_suffix + '\.png$')


def get_sketch_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'sketch', '.*r_\d{3}_sketch_' + size_suffix + '.png$')


def preprocess(image_tensor, img_size,resize_size, whiten=True, color=False,
               augment=False, augment_color=False, augment_translation=False,grayscale=False):
  # Use same seed for flipping for every tensor, so they'll be flipped the same.
  seed = 42
  if color:       
    
    out = tf.reshape(image_tensor, [img_size, img_size, 3])
    out = tf.image.resize_images(out,[resize_size,resize_size])
  else:
    out = tf.reshape(image_tensor, [img_size, img_size, 1])
    out = tf.image.resize_images(out,[resize_size,resize_size],method=1)
  if grayscale==True:
    out = tf.image.rgb_to_grayscale(out)
  if whiten :
    # Bring to range [-1, 1]
    out = tf.cast(out, tf.float32) * (2. / 255) - 1

  return out


def make_image_producer(files, epochs, name, img_size,resize_size, shuffle, whiten, color, filename_seed=233,
                        augment=True, capacity=256, augment_color=False):
  with tf.variable_scope(name) as scope:
    gray_filename_queue = tf.train.string_input_producer(files, num_epochs=epochs, seed=filename_seed,
                                                         capacity=capacity, shuffle=shuffle)
    _, gray_files = tf.WholeFileReader(scope.name).read(gray_filename_queue)
    channels = 3 if color else 1
    img = tf.image.decode_jpeg(gray_files, channels)
    return preprocess(img, img_size,resize_size,
                         whiten=whiten, color=color, augment=augment, augment_color=augment_color)


def read_tensor_record(filename_queue, img_size, resize_size,img_channels,grayscale):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={'image': tf.FixedLenFeature([], tf.string),
                'imageR': tf.FixedLenFeature([], tf.string)})

  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([img_size * img_size * img_channels])

  is_color_img = img_channels == 3
  
  image = preprocess(image, img_size,resize_size,whiten=True, color=is_color_img,grayscale=grayscale)

  imageR = tf.decode_raw(features['imageR'], tf.uint8)
  imageR.set_shape([img_size * img_size * img_channels])

  imageR = preprocess(imageR, img_size,resize_size,
                      whiten=True, color=is_color_img,grayscale=grayscale)
  return  image,imageR


def get_pipeline_training_from_dump(dump_file, batch_size, epochs,
                                          image_size=128,resize_size = 128, img_channels=3, min_queue_size=100, read_threads=4,grayscale=False):
  with tf.variable_scope('dump_reader'):
    with tf.device('/cpu:0'):
      all_files = glob.glob(dump_file + '*')
      filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs,shuffle=True)

      example_list = [read_tensor_record(filename_queue, image_size,resize_size, img_channels,grayscale=grayscale)
                  for _ in range(read_threads)]
      
      return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=min_queue_size + batch_size * 16,
                                         min_after_dequeue=min_queue_size)


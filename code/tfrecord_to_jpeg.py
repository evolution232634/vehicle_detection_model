# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
"""
  savedi:图片的路径
  start:图片名称开始的坐标
"""
def decode_from_tfrecord(savedi,start):
    features={
      'image/encoded': tf.FixedLenFeature([],tf.string),
      'image/format': tf.FixedLenFeature([],tf.string),
      'image/class/label': tf.FixedLenFeature([],tf.int64),
      'image/height': tf.FixedLenFeature([],tf.int64),
      'image/width': tf.FixedLenFeature([],tf.int64)
    }    
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer(savedir)
    _,example=reader.read(filename_queue)
    features=tf.parse_single_example(example,features=features)
    
    image_target = tf.image.decode_jpeg(features['image/encoded'])
    labels=tf.cast(features['image/class/label'],tf.int64)
    width=tf.cast(features['image/width'],tf.int64)
    height = tf.cast(features['image/height'],tf.int64)
    image_format=features['image/format']
    
    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(40):
            image,lable,f,h,w = sess.run([image_target,labels,image_format,height,width])
            sample = sess.run(tf.reshape(image, [h, w, 3]))
            image= Image.fromarray(sample,'RGB')
            index = start+i
            imageName = str(index)
            if index < 9:
                imageName = "00"+imageName
            elif index< 99:
                imageName = "0"+imageName
            image.save("/home/linux/code/data/image/"+imageName+'.jpg')
            
        coord.request_stop()
        coord.join(threads)
savedir = ["/home/linux/code/items/models/pj_vehicle_train_00000-of-00004.tfrecord"]
decode_from_tfrecord(savedir,1)
savedir = ["/home/linux/code/items/models/pj_vehicle_train_00001-of-00004.tfrecord"]
decode_from_tfrecord(savedir,41)
savedir = ["/home/linux/code/items/models/pj_vehicle_train_00002-of-00004.tfrecord"]
decode_from_tfrecord(savedir,81)
savedir = ["/home/linux/code/items/models/pj_vehicle_train_00003-of-00004.tfrecord"]
decode_from_tfrecord(savedir,121)

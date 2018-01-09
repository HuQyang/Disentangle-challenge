import os
import time

import tensorflow as tf


from ops import *
from utils import *
from input_pipeline_rendered_data import get_pipeline_training_from_dump

import math
import numpy as np
import scipy.io as sio

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=256, sample_size = 28, image_shape=[28, 28, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.model_name = "DCGAN.model"
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.image_shape = image_shape
        self.image_size = image_shape[0]

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z = None

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.cg_dim = cg_dim


        self.g_s_bn5 = batch_norm(is_train,convolutional=False, name='g_s_bn5')

        self.build_model(is_train) 


    def build_model(self, is_train):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
        
        self.abstract_size = self.sample_size // 2 ** 4 


        images, imagesR = get_pipeline_training_from_dump('mnist_train/mnist_train.tfrecords',
                                                                 self.batch_size,
                                                                 100, image_size=28,resize_size=28,
                                                                 img_channels=3)

        third_image, _ = get_pipeline_training_from_dump('mnist_train/mnist_train.tfrecords',
                                                                 self.batch_size,
                                                                 100, image_size=28,resize_size=28,
                                                                 img_channels=3)


        test_image_1, test_image_1R = get_pipeline_training_from_dump('mnist_test/mnist_test.tfrecords1',
                                                                 self.batch_size,
                                                                 1, image_size=28,resize_size=28,
                                                                 img_channels=3)

        test_image_2, _ = get_pipeline_training_from_dump('mnist_test/mnist_test.tfrecords8',
                                                                 self.batch_size,
                                                                 1000, image_size=28,resize_size=28,
                                                                 img_channels=3)

        test_image_3, _ = get_pipeline_training_from_dump('mnist_test/mnist_test.tfrecords9',
                                                                 self.batch_size,
                                                                 1000, image_size=28,resize_size=28,
                                                                 img_channels=3)



        self.images = images
        self.imagesR = imagesR
        
        self.third_image = third_image

        self.test_image_1 = test_image_1
        self.test_image_2 = test_image_2
        self.test_image_3 = test_image_3
        self.test_image_1R = test_image_1R

        self.entry_size = 64
        self.feature_size = 128


        with tf.variable_scope('generator') as scope:

            self.image_merge = tf.concat(axis=0,values=[self.images, self.imagesR])
            self.image_merge = tf.concat(axis=0,values=[self.image_merge,self.third_image])

            self.G_merge,self.representation_merge = self.generator(self.image_merge,batch_size=self.batch_size*3)
            entry_size = self.entry_size   

            self.G = self.G_merge[0:self.batch_size,:]
            self.GR = self.G_merge[self.batch_size:self.batch_size+self.batch_size,:]
            self.G3 = self.G_merge[self.batch_size+self.batch_size:,:]

            self.representation = self.representation_merge[0:self.batch_size,:]
            self.representationR = self.representation_merge[self.batch_size:self.batch_size+self.batch_size,:]
            self.representation3 = self.representation_merge[self.batch_size+self.batch_size:,:]
            
            self.ang = tf.reshape(self.representation[:,0:entry_size],[self.batch_size,-1])
            self.obj = self.representation[:,entry_size:]

            self.angR = tf.reshape(self.representationR[:,0:entry_size], [self.batch_size,-1])
            self.objR = self.representationR[:,entry_size:]

            self.IR = tf.concat(axis=1, values=[self.angR, self.obj])
            self.I = tf.concat(axis=1, values=[self.ang, self.objR])

            self.ang3 = tf.reshape(self.representation3[:,0:entry_size],[self.batch_size, -1])
            self.obj3 = self.representation3[:,entry_size:]

            self.D_IR, _ = self.generator(self.IR, from_abstract_representation=True,batch_size=self.batch_size)
            self.D_I, _ = self.generator(self.I, from_abstract_representation=True,batch_size=self.batch_size)
          
            self.v3c1 = tf.concat(axis=1, values=[self.ang3,self.obj])
            self.D_v3c1, _ = self.generator(self.v3c1, from_abstract_representation=True,batch_size=self.batch_size)

            self.image_merge_test = tf.concat(axis=0,values=[self.test_image_1, self.test_image_2])
            self.image_merge_test = tf.concat(axis=0,values=[self.image_merge_test, self.test_image_3])

            scope.reuse_variables()
            self.G_merge_test,self.representation_merge_test = self.generator(self.image_merge_test,batch_size=self.batch_size*3)

            self.real_representation1 = self.representation_merge_test[0:self.batch_size,:]
            self.real_representation2 = self.representation_merge_test[self.batch_size:self.batch_size*2,:]
            self.real_representation3 = self.representation_merge_test[self.batch_size*2:,:]

            self.real_ang1 = tf.reshape(self.real_representation1[:,0:entry_size],[self.batch_size, -1])
            self.real_obj1 = self.real_representation1[:,entry_size:]

            self.real_ang2 = tf.reshape(self.real_representation2[:,0:entry_size],[self.batch_size, -1])
            self.real_obj2 = self.real_representation2[:,entry_size:]

            self.real_ang3 = tf.reshape(self.real_representation3[:,0:entry_size],[self.batch_size, -1])
            self.real_obj3 = self.real_representation3[:,entry_size:]

            self.fea_test21 = tf.concat(axis=1,values=[self.real_ang2, self.real_obj1])
            self.fea_test31 = tf.concat(axis=1,values=[self.real_ang3, self.real_obj1])
            self.fea_test12 = tf.concat(axis=1,values=[self.real_ang1, self.real_obj2])

            scope.reuse_variables()
            self.D_fea_test12, _ = self.generator(self.fea_test12, from_abstract_representation=True,batch_size=self.batch_size)
            scope.reuse_variables()
            self.D_fea_test21, _ = self.generator(self.fea_test21, from_abstract_representation=True,batch_size=self.batch_size)
            scope.reuse_variables()
            self.D_fea_test31, _ = self.generator(self.fea_test31, from_abstract_representation=True,batch_size=self.batch_size)

            self.v12 = tf.concat(axis=1,values=[self.ang, self.angR])
            self.v13 = tf.concat(axis=1,values=[self.ang, self.ang3]) 



        with tf.variable_scope('discriminator') as scope:

            self.D = self.discriminator(self.images,self.imagesR)

            self.D_ = self.discriminator(self.D_v3c1,self.imagesR, reuse=True)


        with tf.variable_scope('discriminator_loss') as scope:
            self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
            self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
            self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('generator_loss') as scope:
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        with tf.variable_scope('L2') as scope:

            self.rec_loss = tf.reduce_mean(tf.square(self.D_I - self.images))

            self.recR_loss = tf.reduce_mean(tf.square(self.D_IR - self.imagesR))

            self.rec_loss_unswap = tf.reduce_mean(tf.square(self.G - self.images))

            self.recR_loss_unswap = tf.reduce_mean(tf.square(self.GR - self.imagesR))


        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_s_vars = [var for var in t_vars if 'g_s' in var.name]
        self.g_e_vars = [var for var in t_vars if 'g_en' in var.name]

        self.saver = tf.train.Saver(self.d_vars + self.g_vars +
                                    batch_norm.shadow_variables,
                                    max_to_keep=0)


    def train(self, config, run_string="???"):
        """Train DCGAN"""

        if config.continue_from_iteration:
            counter = config.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)
        
        # Learning rate of generator is gradually decreasing.
        self.g_lr = tf.train.exponential_decay(0.0002, global_step=global_step,
                                               decay_steps=20000,
                                               decay_rate=0.9,
                                               staircase=True)
        
        self.d_lr = tf.train.exponential_decay(0.0002, global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.9,
                                               staircase=True)


        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=config.beta1) \
                          .minimize(1*self.rec_loss+1*self.recR_loss +20*self.g_loss, var_list=self.g_vars)

        
        d_optim = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars, global_step=global_step)
        # # See that moving average is also updated with g_optim.
        with tf.control_dependencies([g_optim]): 
            g_optim = tf.group(self.bn_assigners) 

        tf.global_variables_initializer().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir, config.continue_from_iteration)

        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.make_summary_ops()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.summary_dir, graph_def=self.sess.graph_def)


        try:
            # Training
            while not coord.should_stop():
                # Update D and G network
                tic = time.time()
                
                self.sess.run([g_optim]) 
                self.sess.run([d_optim]) 

                toc = time.time()

                counter += 1
                duration = toc - tic
                
                if counter % 200 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)

                if np.mod(counter, 3000) == 2: 
                    
                    samples,samplesR,images,imagesR,D_fea_test12,D_fea_test21,D_fea_test31,third_image,test_image_1,test_image_2,test_image_3= self.sess.run([self.G, 
                        self.GR,self.images, self.imagesR,self.D_fea_test12,self.D_fea_test21,self.D_fea_test31,
                        self.third_image,self.test_image_1,self.test_image_2,self.test_image_3])

                    grid_size = np.ceil(np.sqrt(self.batch_size))
                    grid = [grid_size, grid_size]
                    save_images(samples, grid, os.path.join(config.summary_dir, '%s_train.png' % counter))
                    save_images(samplesR, grid, os.path.join(config.summary_dir, '%s_train_GR.png' % counter))
                    save_images(images, grid, os.path.join(config.summary_dir, '%s_train_images.png' % counter))
                    save_images(imagesR, grid, os.path.join(config.summary_dir, '%s_train_imageR.png' % counter))
                    save_images(third_image, grid, os.path.join(config.summary_dir, '%s_third_image.png' % counter))
)
                    save_images(test_image_1, grid, os.path.join(config.summary_dir, '%s_test1.png' % counter))
                    save_images(test_image_2, grid, os.path.join(config.summary_dir, '%s_test2.png' % counter))
                    save_images(test_image_3, grid, os.path.join(config.summary_dir, '%s_test3.png' % counter))
                    save_images(D_fea_test21, grid, os.path.join(config.summary_dir, '%s_test_comb21.png' % counter))
                    save_images(D_fea_test31, grid, os.path.join(config.summary_dir, '%s_test_comb31.png' % counter))
                    save_images(D_fea_test12, grid, os.path.join(config.summary_dir, '%s_test_comb12.png' % counter))


                if np.mod(counter, 2000) == 100:
                    self.save(config.checkpoint_dir, counter)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


    def discriminator(self, image1,image2, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        concated = tf.concat(axis=3, values=[image1, image2])

        conv0 = lrelu((conv2d(concated, 96,k_h=3, k_w=3, d_h=1, d_w=1,padding = 'VALID', name='d_3_s0_conv')))
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn0 = tf.nn.local_response_normalization(conv0,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name="d_3_lrn0")

        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        # k_h = 7; k_w = 7; s_h = 2; s_w = 2; padding = 'VALID' # version 1 
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID' # size 12
        self.s0 = tf.nn.max_pool(lrn0, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name="d_3_mp0")

        conv1 = conv2d(self.s0, 256,k_h=3, k_w=3, d_h=1, d_w=1, name='d_3_s1_conv') #version 2 size 10
        # conv1 = tf.nn.relu(conv1)
        conv1 = lrelu((conv1))
        #lrn2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,depth_radius=radius,alpha=alpha,beta=beta,bias=bias, name="d_3_lrn1")
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID' #version 2 size 4
        self.s1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="d_3_mp1")


        # #conv(3, 3, 384, 1, 1, name='conv3')            
        conv2 = conv2d(self.s1, 384,k_h=3, k_w=3, d_h=1, d_w=1, name='d_3_s2_conv') # version 2 size13
        self.s2 = lrelu((conv2))

        # #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        conv3 = conv2d(self.s2, 384,k_h=3, k_w=3, d_h=1, d_w=1, name='d_3_s3_conv') # version 2 size 13
        self.s3 = lrelu((conv3))


        fc = lrelu(linear(tf.reshape(self.s3, [self.batch_size, -1]), 1, 'd_3_fc6') )  

        return tf.nn.sigmoid(fc)


    def generator(self, sketches_or_abstract_representations,batch_size=64, z=None, y=None, from_abstract_representation=False):            
        if from_abstract_representation:
            # Used when feeding abstract representation directly, not deriving it from a sketch.
            used_abstract = sketches_or_abstract_representations
            tf.get_variable_scope().reuse_variables()
        else:
            self.s0 = lrelu(instance_norm(conv2d((sketches_or_abstract_representations), self.df_dim, name='g_s0_conv')))
            self.s1 = lrelu(instance_norm(conv2d(self.s0, self.df_dim * 2, name='g_s1_conv')))
            self.s2 = lrelu(instance_norm(conv2d(self.s1, self.df_dim * 4, name='g_s2_conv')))

            used_abstract = lrelu((linear(tf.reshape(self.s2, [batch_size, -1]), self.feature_size, 'g_en_fc')) )


        h = deconv2d(tf.reshape(used_abstract,[batch_size,1,1,self.feature_size]), [batch_size,4,4, self.gf_dim*8],k_h=4, k_w=4, d_h=1, d_w=1,padding = 'VALID',name='g_de_h')
        h = tf.nn.relu(instance_norm(h))

        h1 = deconv2d(h, [batch_size, 7, 7, self.gf_dim*4 ], name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [batch_size, 14, 14, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [batch_size, 28, 28, self.c_dim], name='g_h3')
        
        return tf.nn.tanh(h3), used_abstract
        
    def make_summary_ops(self):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        tf.summary.scalar('d_loss_real', self.d_loss_real)
        tf.summary.scalar('rec_loss', self.rec_loss)


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir) 

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, iteration=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name

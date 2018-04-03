# -*- coding: utf-8 -*-
import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

import config
import time
from tensorflow.examples.tutorials.mnist import input_data

def average_gradients(grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class Trainer:
    def __init__(self):
        print '[!] START INITIALIZING TRAINER'
        with tf.device('/cpu:0'):
            # global variables from all towers
            self.all_towers   =  []
            self.all_outputs  =  []
            self.all_gradients = []
            self.all_losses =    []

            # basic variables


            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.optimizer = config.MODEL_OPTIMIZER(learning_rate=self.learning_rate)



            self.input = tf.placeholder(
                dtype=tf.float32,
                shape=[
                    None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH
                ],
                name='input'
            )
            self.label = tf.placeholder(dtype=tf.float32, shape=[None,config.NUM_CLASS], name='label')

            # input and label data for each towers
            self.tower_inputs = tf.split(self.input, config.NUM_GPUS, 0)
            self.tower_labels = tf.split(self.label, config.NUM_GPUS, 0)

            # build towers for each GPUs
            for i in range(config.NUM_GPUS):
                self.build_tower(i, self.tower_inputs[i], self.tower_labels[i])

            # define your training step
            self.train_step = tf.group(
                self.optimizer.apply_gradients(
                    global_step=self.global_step,
                    grads_and_vars= average_gradients(self.all_gradients)
                )
            )

            # define statistical information
            self.global_loss = tf.reduce_mean(self.all_losses)


            is_correct = tf.equal(tf.argmax(tf.concat(self.all_outputs,  axis=0),1), tf.argmax(tf.concat(self.tower_labels, axis=0), 1))
            self.global_acc = tf.reduce_mean(tf.cast(is_correct,tf.float32))

            self.summary = self.define_summarizer()
        print '[!] INITIALIZING DONE'

    #tensorboard --logdir=/tmp/sample
    def define_summarizer(self):
        """
        Define summary information for tensorboard
        """
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.scalar('global_loss', self.global_loss)
        tf.summary.scalar('global_acc', self.global_acc)

        return tf.summary.merge_all()


    def build_tower(self, gpu_index, X, Y):
        print '[!] BUILD TOWER %d' % gpu_index
        with tf.device('/gpu:%d' % gpu_index), tf.name_scope('tower_%d' % gpu_index), tf.variable_scope(tf.get_variable_scope(), reuse= gpu_index is not 0):
            graph = config.MODEL_NETWORK.build_graph(X, Y, gpu_index is not 0 )
            loss = graph['loss']
            output = graph['prediction']
            gradients = self.optimizer.compute_gradients(loss)

            self.all_towers.append(graph)
            self.all_losses.append(loss)
            self.all_outputs.append(output)
            self.all_gradients.append(gradients)
            tf.get_variable_scope().reuse_variables()



    def run_train(self, data_manager, model_save_path):

        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        train_start_time = time.time()
        with tf.device('/cpu:0'), session.as_default():


            #Configurations for saving and loading your model
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            ckpt  = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path :
                print 'Restore trained model from ' + ckpt.model_checkpoint_path
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print 'Create new model'
                os.makedirs(model_save_path)
                session.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(os.path.join(model_save_path, 'logs'))



            #init variable for learning rate
            loss_array = []
            last_loss=999999
            lr_down_level=0
            weight_decay =float(config.TRAIN_WEIGHT_DECAY)
            weight_decay_level = 0
            init_lr= float(config.TRAIN_LEARNING_RATE)


            # Train your model iteratively
            while session.run(self.global_step) < config.TRAIN_MAX_STEPS:

                lr = init_lr*(config.TRAIN_DECAY **lr_down_level)
                lr -= lr*weight_decay*weight_decay_level

                step_index = int(session.run(self.global_step))


                # read input data from data manager
                X, Y = data_manager.train.next_batch(config.BATCH_SIZE)
                X = X.reshape((config.BATCH_SIZE, config.IMAGE_HEIGHT,config.IMAGE_WIDTH))


                _, loss, acc, summary = session.run(
                    fetches=[
                        self.train_step,
                        self.global_loss,
                        self.global_acc,
                        self.summary
                    ],
                    feed_dict={
                        self.input: X,
                        self.label: Y,
                        self.learning_rate: lr
                    }
                )

                loss_array.append(loss)
                #modurate learning rate
                if (step_index % (data_manager.train.num_examples/config.BATCH_SIZE) == 0 and step_index!=0):
                    print '[Step %5d] LR: %.5E, LOSS: %.5E, ACC: %.5E' % (step_index, lr, loss, acc)
                    epoch = int(step_index / (data_manager.train.num_examples / config.BATCH_SIZE))
                    print 'current epoch : %d' % epoch

                    cur_loss= np.average(loss_array)
                    loss_array=[]

                    print 'cur loss : %f , last loss : %f'%(cur_loss,last_loss)
                    if(epoch==30 or epoch ==60):
                        break
                        lr_down_level+=1
                        print 'lr_down_level up : %d' % lr_down_level
                        weight_decay_level=0
                    else:
                        #weight_decay_level +=1
                        print 'weight decay level up : %d' % weight_decay_level
                    if (epoch == 120):
                        break
                    last_loss=cur_loss

                # Save your summary information for tensor-board and model data
                summary_writer.add_summary(summary, global_step=step_index)


                if step_index % config.MODEL_SAVE_INTERVAL == 0 :
                    saver.save(session, os.path.join(model_save_path, 'tr'), global_step=step_index)


            test_batch_size = len(data_manager.test.images)
            testX=data_manager.test.images.reshape(test_batch_size,config.IMAGE_HEIGHT,config.IMAGE_WIDTH)
            testY=data_manager.test.labels
            print('Test Accuracy : ', session.run(self.global_acc, feed_dict={self.input: testX,
                        self.label: testY,}))

        train_end_time = time.time()
        print '--- All Training Done Successfully in %.7f seconds ----' % (train_end_time - train_start_time)




if __name__ == '__main__':

    #Load Database
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    #Init trainer and run training sessions
    trainer = Trainer()
    trainer.run_train(
        data_manager= mnist,
        model_save_path = os.path.join(config.MODEL_SAVE_FOLDER, config.MODEL_SAVE_NAME)
    )


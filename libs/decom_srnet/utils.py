import os
import sys
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve, auc
import configparser
import pandas as pd

from gen_data import *
from decom_srnet import copy_weights_to_decom_srnet

sys.path.append('../../third_party/tflib/')
from queues import *
from utils_multistep_lr import average_summary, AdamaxOptimizer
from utils_multistep_lr import Model as Model_ori
# from SRNet import *


class Model(Model_ori):
    """ expand the class utils_multistep_lr.Model """
    def __init__(self, is_training=None, data_format='NCHW'):
        super(Model, self).__init__(is_training, data_format)

    def _build_predict(self, labels):
        self.labels = tf.cast(labels, tf.int64)
        with tf.variable_scope('prob'):
           test_result = tf.nn.softmax(self.outputs)
           self.predict = test_result[:,1]

        return self.labels, self.predict

def train(model_class, train_gen, valid_gen, train_batch_size, \
          valid_batch_size, valid_ds_size, optimizer, boundaries, values, \
          train_interval, valid_interval, max_iter, \
          save_interval, log_path, acc_log_path, num_runner_threads=1, \
          load_path=None, config_log=None, wauc_flag=False, transfer_flag=False):
    tf.reset_default_graph()
    train_runner = GeneratorRunner(train_gen, train_batch_size * 10)
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 10)
    is_training = tf.get_variable('is_training', dtype=tf.bool, \
                                  initializer=True, trainable=False)
    if train_batch_size == valid_batch_size:
        batch_size = train_batch_size
        disable_training_op = tf.assign(is_training, False)
        enable_training_op = tf.assign(is_training, True)
    else:
        batch_size = tf.get_variable('batch_size', dtype=tf.int32, \
                                     initializer=train_batch_size, \
                                     trainable=False, \
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
        disable_training_op = tf.group(tf.assign(is_training, False), \
                                       tf.assign(batch_size, valid_batch_size))
        enable_training_op = tf.group(tf.assign(is_training, True), \
                                      tf.assign(batch_size, train_batch_size))
    img_batch, label_batch = queueSelection([valid_runner, train_runner], \
                                            tf.cast(is_training, tf.int32), \
                                            batch_size)
        
    model = model_class(is_training, 'NHWC')
    if model_class.__name__ != 'SRNetDecom':
        model._build_model(img_batch)
    else:
        assert config_log != None, 'SRNetDecom. config_log == None !'
        model._build_model(img_batch, config_log)

    labels, predicts = model._build_predict(label_batch)

    loss, accuracy = model._build_losses(label_batch)
    train_loss_s = average_summary(loss, 'train_loss', train_interval)
    train_accuracy_s = average_summary(accuracy, 'train_accuracy', \
                                       train_interval)
    valid_loss_s = average_summary(loss, 'valid_loss', \
                                   float(valid_ds_size) / float(valid_batch_size))
    valid_accuracy_s = average_summary(accuracy, 'valid_accuracy', \
                                       float(valid_ds_size) / float(valid_batch_size))
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)

    init_step = tf.assign(global_step, 0)
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)
    optimizer = optimizer(learning_rate)

    minimize_op = optimizer.minimize(loss, global_step)
    train_op = tf.group(minimize_op, train_loss_s.increment_op, \
                        train_accuracy_s.increment_op)
    increment_valid = tf.group(valid_loss_s.increment_op, \
                               valid_accuracy_s.increment_op)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    # The following code calculates the parameters for each convolutional layer and the total parameters
    # total_parameters = 0
    # for variable in tf.trainable_variables():
    #     # shape is an array of tf.Dimension
    #     shape = variable.get_shape()
    #
    #     # condition == 4, Focus only on convolutional layers;
    #     # condition > 0, Focus on all layers;
    #     if len(shape) == 4:
    #         variable_parameters = 1
    #         for dim in shape:
    #             variable_parameters *= dim.value
    #         print('variable_parameters:', variable_parameters)
    #         total_parameters += variable_parameters
    # print("total_parameters:", total_parameters)
    #
    # # for all data collection(gpu memory etc.)
    # run_metadata = tf.RunMetadata()
    #
    # # varlist = optimistic_restore_vars(load_path)
    # # saver = tf.train.Saver(varlist, max_to_keep=10000)
    # saver = tf.train.Saver(max_to_keep=10000)
    # with tf.Session() as sess:
    #     # sess.run(init_op)
    #
    #     # Write down all the data by using RunMetadata
    #     sess.run(init_op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
    #                                             output_partition_graphs=True), run_metadata=run_metadata)
    #     # write it to txt file
    #     with open("calpa_init_full_traces_srnetc64_256_bs2.txt", "w") as out:
    #         # with open("test.txt", "w") as out:
    #         out.write(str(run_metadata))
    #     with open("calpa_init_full_traces_srnetc64_256_bs2.txt", "r") as f:
    #         # with open("test.txt", "r") as f:
    #         requested_bytes = 0
    #         allocated_bytes = 0
    #         for line in f.readlines():
    #             line = line.strip('\n')
    #             if 'requested_bytes' in line:
    #                 data = line.split(' ')
    #                 requested_bytes += int(data[-1])
    #             if 'allocated_bytes' in line:
    #                 data = line.split(' ')
    #                 allocated_bytes += int(data[-1])
    #         print('requested_bytes:', requested_bytes)
    #         print('allocated_bytes:', allocated_bytes)
    #         print("log_gpu_memory_finished....")
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        with open(acc_log_path,'a+') as f:
            f.write("load_path: %s\n" % (load_path))
            f.write("is transfer: %s\n" % (transfer_flag))
        if load_path is not None:
            saver.restore(sess, load_path)
        if transfer_flag is True:
            assert load_path is not None, 'load_path is None!!!'
            sess.run(init_step)
        train_runner.start_threads(sess, num_runner_threads)
        valid_runner.start_threads(sess, 1)
        writer = tf.summary.FileWriter(log_path + '/LogFile/', \
                                       sess.graph)
        start = sess.run(global_step)
        sess.run(disable_training_op)
        sess.run([valid_loss_s.reset_variable_op, \
                  valid_accuracy_s.reset_variable_op, \
                  train_loss_s.reset_variable_op, \
                  train_accuracy_s.reset_variable_op])
        _time = time.time()
        for j in range(0, valid_ds_size, valid_batch_size):
            sess.run([increment_valid])
        _acc_val = sess.run(valid_accuracy_s.mean_variable)
        print ("initial accuracy on validation set:", _acc_val)
        print ("evaluation time on validation set:", time.time() - _time, "seconds")
        valid_accuracy_s.add_summary(sess, writer, start)
        valid_loss_s.add_summary(sess, writer, start)
        sess.run(enable_training_op)
        print ("network will be evaluatd every %i iterations on validation set" % valid_interval)

        model_path = os.path.join(log_path, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # save model 0
        if load_path is None:
            saver.save(sess, model_path + '/model_0.ckpt')

        all_labels = np.array([])
        all_predicts = np.array([])

        for i in range(start + 1, max_iter + 1):
            # start "time"
            sess.run(train_op)
            # end "time"
            if i % train_interval == 0:
                train_loss_s.add_summary(sess, writer, i)
                train_accuracy_s.add_summary(sess, writer, i)
                s = sess.run(lr_summary)
                writer.add_summary(s, i)
            if i % valid_interval == 0:
                sess.run(disable_training_op)
                for j in range(0, valid_ds_size, valid_batch_size):
                    sess.run([increment_valid])

                    valid_label, valid_predict = sess.run([labels, predicts])

                    all_labels = np.concatenate([all_labels, valid_label], axis=0)
                    all_predicts = np.concatenate([all_predicts, valid_predict], axis=0)

                valid_loss, valid_accuracy = sess.run([valid_loss_s.mean_variable, \
                                             valid_accuracy_s.mean_variable])
                AUC = roc_auc_score(all_labels, all_predicts)
                valid_loss_s.add_summary(sess, writer, i)
                valid_accuracy_s.add_summary(sess, writer, i)
                print ("iteration:", i)
                print ("accuracy on validation set:", valid_accuracy)
                if wauc_flag == True:
                    weight_auc = alaska_weighted_auc(all_labels, all_predicts)
                    with open(acc_log_path,'a+') as f:
                        f.write("Iter:%d, valid_acc: %s, valid_loss: %s, auc: %s, weight_auc: %s, time:%s\n" 
                                        %(i,  str(valid_accuracy), str(valid_loss), str(AUC), str(weight_auc),
                                        time.strftime("%Y%m%d-%H%M%S")))
                else:
                    with open(acc_log_path,'a+') as f:
                        f.write("Iter:%d, valid_acc: %s, valid_loss: %s, auc: %s, time:%s\n" 
                                        %(i,  str(valid_accuracy), str(valid_loss), str(AUC),
                                        time.strftime("%Y%m%d-%H%M%S")))
                sess.run(enable_training_op)
            if i % save_interval == 0:
                saver.save(sess, model_path + '/model_' + str(i) + '.ckpt')

def test_dataset(model_class, gen, batch_size, ds_size, load_path, test_log_path=None, config_log=None, csv_path=None):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NCHW')

    if model_class.__name__ != 'SRNetDecom':
        model._build_model(img_batch)
    else:
        assert config_log != None, 'SRNetDecom. config_log == None !'
        model._build_model(img_batch, config_log)

    labels, predicts = model._build_predict(label_batch)

    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss', \
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy', \
                                       float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)

        all_labels = np.array([])
        all_predicts = np.array([])

        for j in range(0, ds_size, batch_size):
            sess.run(increment_op)

            valid_label, valid_predict = sess.run([labels, predicts])
            all_labels = np.concatenate([all_labels, valid_label], axis=0)
            all_predicts = np.concatenate([all_predicts, valid_predict], axis=0)

        mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable, \
                                             accuracy_summary.mean_variable])

        AUC = roc_auc_score(all_labels, all_predicts)
        weight_auc = alaska_weighted_auc(all_labels, all_predicts)
        cur_iter = sess.run(global_step)
        if test_log_path != None:
            with open(test_log_path,'a+') as f:
                        f.write("Model:%d, test_acc: %s, test_loss: %s, auc: %s, weight_auc: %s, time:%s\n" 
                                        %(cur_iter,  str(mean_accuracy), str(mean_loss), str(AUC), str(weight_auc),
                                        time.strftime("%Y%m%d-%H%M%S")))


        if csv_path != None:
            # write to csv
            df = pd.DataFrame({'Predict': all_predicts, 'Label': all_labels})
            df.to_csv(csv_path, mode='w', index=True, sep=',', columns=['Predict', 'Label'],
                    header=0)

            print("cal P_FA")
            eval_criter = tf_roc(all_predicts, all_labels, 9999)
            pfa_30 = eval_criter.get_pfa(0.3)
            pfa_50 = eval_criter.get_pfa(0.5)
            pfa_70 = eval_criter.get_pfa(0.7)
            pmd_05 = eval_criter.get_pmd(0.05)
            print("P_FA(30%%): %f" % (pfa_30))
            print("P_FA(50%%): %f" % (pfa_50))
            print("P_FA(70%%): %f" % (pfa_70))
            print("P_MD(5%%): %f" % (pmd_05))
    print ("Accuracy:", mean_accuracy, " | Loss:", mean_loss)


from bisect import bisect_left

class tf_roc():
    def __init__(self, predicts, labels, threshold_num, predict_label_file=None):
        '''predict_score,label
        the predict_score should be between 0 and 1
        the label should be 0 or 1
        threshold_num: number of threshold will plot'''

        self.predicts = []
        self.labels = []
        self.total = 0

        if predict_label_file is not None:
            fd = open(predict_label_file)
            fdl = fd.readline()
            while len(fdl) > 0:
                fdl = fdl.replace('\n', '')
                val = fdl.split(',')
                # val[2] = val[2].split('\\')[0]
                self.predicts.append(float(val[1]))
                self.labels.append(True if int(eval(val[2])) == 1 else False)
                fdl = fd.readline()
                self.total += 1
            fd.close()
        else:
            if not isinstance(predicts, list):
                predicts = list(predicts)
            if not isinstance(labels, list):
                labels = list(labels)
            self.predicts = predicts
            self.labels = labels
            self.total = len(self.labels)
        print(self.total)
        self.threshold_num = threshold_num
        self.trues = 0  # total of True labels
        self.fpr = []  # false positive rate
        self.tpr = []  # true positive rate
        self.ths = []  # thresholds
        self.tn = []  # true negative
        self.tp = []  # true positive
        self.fp = []  # false positive
        self.fn = []  # false negative
        self.calc()

    def calc(self):
        for label in self.labels:
            if label:
                self.trues += 1
        # print 'self.trues:', self.trues
        threshold_step = 1. / self.threshold_num
        for t in range(self.threshold_num + 1):
            th = 1 - threshold_step * t
            tn, tp, fp, fn, fpr, tpr = self._calc_once(th)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.ths.append(th)
            self.tn.append(tn)
            self.tp.append(tp)
            self.fp.append(fp)
            self.fn.append(fn)

    def _calc_once(self, t):
        fp = 0
        tp = 0
        tn = 0
        fn = 0
        # print 't:', t
        for i in range(self.total):
            # print 'labels:', self.labels[i], ' predicts:', self.predicts[i]
            if not self.labels[i]:  # when labels[i] == 0 or false
                if self.predicts[i] >= t:
                    fp += 1  # false positive
                    # print 'fp == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
                else:
                    tn += 1  # true negative
                    # print 'tn == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
            elif self.labels[i]:
                if self.predicts[i] >= t:  # when labels[i] == 1 or true
                    tp += 1  # true positive
                    # print 'tp == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
                else:
                    fn += 1  # false negative
                    # print 'fn == ', 'labels:', self.labels[i], ' predicts:', self.predicts[i]
        fpr = fp / float(fp + tn)  # precision
        tpr = tp / float(self.trues)

        return tn, tp, fp, fn, fpr, tpr

    def get_pmd(self, thresh):
        # thresh is pre-defined FPR
        tpr = self.tpr
        fpr = self.fpr

        #  fnr = 1 - self.tpr

        if (thresh >= fpr[-1]):
            print('warning !!!')
            return 1 - tpr[-1]
        elif thresh <= fpr[0]:
            print('warning !!!')
            return 1 - tpr[0]
        pos = bisect_left(fpr, thresh)
        part_fpr = [fpr[pos - 1], fpr[pos]]
        part_pos = [pos - 1, pos]
        part_tpr = [tpr[pos - 1], tpr[pos]]
        if thresh in part_fpr:
            sp_pos = part_pos[part_fpr.index(thresh)]
            pmd = 1.0 - tpr[sp_pos]
        else:
            assert thresh in [0.05], "type error get P_MD(%f)" % (thresh)
            f1 = np.polyfit(part_fpr, part_tpr, 1)
            pmd = 1 - np.polyval(f1, thresh)
        print('%.2f part_tpr: %s' % (thresh, part_tpr))
        print('%.2f part_fpr: %s' % (thresh, part_fpr))

        return pmd

    def get_pfa(self, thresh):
        # args: thresh is pre-defined TPR

        tpr = self.tpr
        fpr = self.fpr

        if (thresh >= tpr[-1]):
            print('warning !!!')
            return fpr[-1]
        elif thresh <= tpr[0]:
            print('warning !!!')
            return fpr[0]
        pos = bisect_left(tpr, thresh)
        part_tpr = [tpr[pos - 1], tpr[pos]]
        part_pos = [pos - 1, pos]
        part_fpr = [fpr[pos - 1], fpr[pos]]
        if thresh in part_tpr:
            sp_pos = part_pos[part_tpr.index(thresh)]
            pfa = fpr[sp_pos]
        else:
            # pfa = get_FA(part_tpr, part_fpr, thresh)
            assert thresh in [0.3, 0.5, 0.7], "type error get P_FA(%f)" % (thresh)
            # p1 = np.poly1d(f1)

            f1 = np.polyfit(part_tpr, part_fpr, 1)
            pfa = np.polyval(f1, thresh)

        print('%.2f part_tpr: %s' % (thresh, part_tpr))
        print('%.2f part_fpr: %s' % (thresh, part_fpr))
        # print ('P_FA(%f): %f'%(thresh, pfa))

        return pfa


def compute_ma(batch_size, f_o, f_c):
    # L2 NORM
    ma_mat = np.zeros((batch_size, 1))

    ma_mat = np.linalg.norm(f_c - f_o, ord=2, axis=1) / np.linalg.norm(f_o, ord=2, axis=1)
    ma = np.sum(ma_mat)/batch_size
    print("ma_mat.shape: ", ma_mat.shape)   # (batch_size,)
    assert ma_mat.shape[0]==batch_size, "ma_mat.shape[0]=%d is not equal ot batch_size=%d"%(ma_mat.shape[0], batch_size)

    return ma_mat, ma

    
def get_assign_tensor_name(layer, index):

    if index == 1 and layer != 2:
        if (layer >= 3 and layer <=7) or layer == 12:
            new_conv_name = 'Layer' + str(layer) + '/Conv/weights:0'
            new_conv1_name = 'Layer' + str(layer) + '/Conv_1/weights:0'
            new_conv2_name = 'Layer' + str(layer) + '/Conv_2/weights:0'
            new_bias2_name = 'Layer' + str(layer) + '/Conv_2/biases:0'
            ori_conv1_name = 'Layer' + str(layer) + '/Conv_1/weights'
            new_conv3_name = 'Layer' + str(layer) + '/Conv_3/weights:0'
            ori_bias1_name = 'Layer' + str(layer) + '/Conv_1/biases'
            new_bias3_name = 'Layer' + str(layer) + '/Conv_3/biases:0'
        elif layer >= 8 and layer <= 11:
            new_conv_name = 'Layer' + str(layer) + '/Conv_1/weights:0'
            new_conv1_name = 'Layer' + str(layer) + '/Conv_2/weights:0'
            new_conv2_name = 'Layer' + str(layer) + '/Conv_3/weights:0'
            new_bias2_name = 'Layer' + str(layer) + '/Conv_3/biases:0'
            ori_conv1_name = 'Layer' + str(layer) + '/Conv_2/weights'
            new_conv3_name = 'Layer' + str(layer) + '/Conv_4/weights:0'
            ori_bias1_name = 'Layer' + str(layer) + '/Conv_2/biases'
            new_bias3_name = 'Layer' + str(layer) + '/Conv_4/biases:0'
        
        return new_conv_name, new_conv1_name, new_conv2_name, new_bias2_name, \
            ori_conv1_name, new_conv3_name, ori_bias1_name, new_bias3_name
    elif index == 2 or layer == 2:
        if (layer >= 3 and layer <=7) or layer == 12:
            new_conv_name = 'Layer' + str(layer) + '/Conv_1/weights:0'
            new_conv1_name = 'Layer' + str(layer) + '/Conv_2/weights:0'
            new_conv2_name = 'Layer' + str(layer) + '/Conv_3/weights:0'
            new_bias2_name = 'Layer' + str(layer) + '/Conv_3/biases:0'
        elif layer >= 8 and layer <= 11:
            new_conv_name = 'Layer' + str(layer) + '/Conv_2/weights:0'
            new_conv1_name = 'Layer' + str(layer) + '/Conv_3/weights:0'
            new_conv2_name = 'Layer' + str(layer) + '/Conv_4/weights:0'
            new_bias2_name = 'Layer' + str(layer) + '/Conv_4/biases:0'
        elif layer == 2:
            new_conv_name = 'Layer' + str(layer) + '/Conv/weights:0'
            new_conv1_name = 'Layer' + str(layer) + '/Conv_1/weights:0'
            new_conv2_name = 'Layer' + str(layer) + '/Conv_2/weights:0'
            new_bias2_name = 'Layer' + str(layer) + '/Conv_2/biases:0'

        return new_conv_name, new_conv1_name, new_conv2_name, new_bias2_name
    

def assign_sess(sess, load_path_ori, layer, index, first, core, last, var_bias):
    if index == 1 and layer != 2:
        new_conv_name, new_conv1_name, new_conv2_name, new_bias2_name, \
            ori_conv1_name, new_conv3_name, ori_bias1_name, new_bias3_name = \
            get_assign_tensor_name(layer, index)
        for v in tf.global_variables():
            if v.name == new_conv_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), first, validate_shape=False))
            elif v.name == new_conv1_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), core, validate_shape=False))
            elif v.name == new_conv2_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), last, validate_shape=False))
            elif v.name == new_bias2_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), var_bias, validate_shape=False))
            elif v.name == new_conv3_name:
                name_ori = ori_conv1_name
                var = tf.contrib.framework.load_variable(load_path_ori, name_ori)
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), var, validate_shape=False))
            elif v.name == new_bias3_name:
                name_ori = ori_bias1_name
                var = tf.contrib.framework.load_variable(load_path_ori, name_ori)
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), var, validate_shape=False))
            else:
                var = tf.contrib.framework.load_variable(load_path_ori, v.name[:-2])
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), var, validate_shape=False))
        
    elif index == 2 or layer == 2:
        new_conv_name, new_conv1_name, new_conv2_name, new_bias2_name = \
            get_assign_tensor_name(layer, index)
        for v in tf.global_variables():
            if v.name == new_conv_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), first, validate_shape=False))
            elif v.name == new_conv1_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), core, validate_shape=False))
            elif v.name == new_conv2_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), last, validate_shape=False))
            elif v.name == new_bias2_name:
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), var_bias, validate_shape=False)) 
            else:
                var = tf.contrib.framework.load_variable(load_path_ori, v.name[:-2])
                sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(v.name), var, validate_shape=False))

def get_decom_tensor_name(layer, index):
    """
        get decom tensor name of Conv and BN
    """
    if index == 1:
        if (layer >= 2 and layer <= 7) or layer == 12:
            ori_conv_name = 'Layer' + str(layer) + '/Conv/weights'
            ori_bias_name = 'Layer' + str(layer) + '/Conv/biases'
            ori_bn_name = 'Layer' + str(layer) + '/BatchNorm/FusedBatchNorm:0'
        elif layer >= 8 and layer <= 11:
            ori_conv_name = 'Layer' + str(layer) + '/Conv_1/weights'
            ori_bias_name = 'Layer' + str(layer) + '/Conv_1/biases'
            ori_bn_name = 'Layer' + str(layer) + '/BatchNorm_1/FusedBatchNorm:0'
    elif index == 2:
        if (layer >= 3 and layer <= 7) or layer == 12:
            ori_conv_name = 'Layer' + str(layer) + '/Conv_1/weights'
            ori_bias_name = 'Layer' + str(layer) + '/Conv_1/biases'
            ori_bn_name = 'Layer' + str(layer) + '/BatchNorm_1/FusedBatchNorm:0'
        elif layer >= 8 and layer <= 11:
            ori_conv_name = 'Layer' + str(layer) + '/Conv_2/weights'
            ori_bias_name = 'Layer' + str(layer) + '/Conv_2/biases'
            ori_bn_name = 'Layer' + str(layer) + '/BatchNorm_2/FusedBatchNorm:0'
    return ori_conv_name, ori_bias_name, ori_bn_name


def initial_and_decom_model(model_class, train_gen, valid_gen, train_batch_size, \
          valid_batch_size, valid_ds_size, optimizer, boundaries, values, \
          train_interval, valid_interval, save_path, Model_name, \
          load_path_ori, config_log, model_name_decom, num_runner_threads=1): #initial the model of decomposition
    tf.reset_default_graph()
    train_runner = GeneratorRunner(train_gen, train_batch_size * 10)
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 10)
    is_training = tf.get_variable('is_training', dtype=tf.bool, \
                                  initializer=True, trainable=False)
    if train_batch_size == valid_batch_size:
        batch_size = train_batch_size
        disable_training_op = tf.assign(is_training, False)
        enable_training_op = tf.assign(is_training, True)
    else:
        batch_size = tf.get_variable('batch_size', dtype=tf.int32, \
                                     initializer=train_batch_size, \
                                     trainable=False, \
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
        disable_training_op = tf.group(tf.assign(is_training, False), \
                                tf.assign(batch_size, valid_batch_size))
        enable_training_op = tf.group(tf.assign(is_training, True), \
                                tf.assign(batch_size, train_batch_size))
    img_batch, label_batch = queueSelection([valid_runner, train_runner], \
                                            tf.cast(is_training, tf.int32), \
                                            batch_size)
    model = model_class(is_training, 'NHWC') 
    if model_class.__name__ != 'SRNetDecom':
        model._build_model(img_batch)
    else:
        assert config_log != None, 'SRNetDecom. config_log == None !'
        model._build_model(img_batch, config_log)

    loss, accuracy = model._build_losses(label_batch)
    train_loss_s = average_summary(loss, 'train_loss', train_interval)
    train_accuracy_s = average_summary(accuracy, 'train_accuracy', \
                                       train_interval)
    valid_loss_s = average_summary(loss, 'valid_loss', \
                                   float(valid_ds_size) / float(valid_batch_size))
    valid_accuracy_s = average_summary(accuracy, 'valid_accuracy', \
                                       float(valid_ds_size) / float(valid_batch_size))
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)
    optimizer = optimizer(learning_rate) 
        
    minimize_op = optimizer.minimize(loss, global_step)
    train_op = tf.group(minimize_op, train_loss_s.increment_op, \
                        train_accuracy_s.increment_op)
    increment_valid = tf.group(valid_loss_s.increment_op, \
                               valid_accuracy_s.increment_op)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        train_runner.start_threads(sess, num_runner_threads)
        valid_runner.start_threads(sess, 1)
        start = sess.run(global_step)
        sess.run(disable_training_op)
        sess.run([valid_loss_s.reset_variable_op, \
                  valid_accuracy_s.reset_variable_op, \
                  train_loss_s.reset_variable_op, \
                  train_accuracy_s.reset_variable_op])
        load_path_new = os.path.join(save_path, Model_name)
        saver.save(sess, load_path_new)
    copy_weights_to_decom_srnet(load_path_ori, load_path_new, config_log, save_path, model_name_decom)

def stat_parameters():
    # calculates the parameters for each convolutional layer and the total parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()

        if len(shape) == 4:          # condition == 4, Focus only on convolutional layers; condition > 0, Focus on all layers;
            variable_parameters = 1
            for dim in shape:
                # print 'dim:', dim
                variable_parameters *= dim.value
            print('variable_parameters:', variable_parameters)
            total_parameters += variable_parameters
    print("total_parameters:", total_parameters)

def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2,   1]

    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        # pdb.set_trace()

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

def write_2_config_log(config_log, layer, index, input_rank, output_rank):
    cf = configparser.ConfigParser()
    cf.read(config_log)
    if index == 1:
        cf.remove_option('first_layer_input_ranks', 'layer' + str(layer))
        cf.set('first_layer_input_ranks', 'layer' + str(layer), str(input_rank))
        cf.remove_option('first_layer_output_ranks', 'layer' + str(layer))
        cf.set('first_layer_output_ranks', 'layer' + str(layer), str(output_rank))
    else:
        cf.remove_option('second_layer_input_ranks', 'layer' + str(layer))
        cf.set('second_layer_input_ranks', 'layer' + str(layer), str(input_rank))
        cf.remove_option('second_layer_output_ranks', 'layer' + str(layer))
        cf.set('second_layer_output_ranks', 'layer' + str(layer), str(output_rank))
    cf.write(open(config_log, 'w'))

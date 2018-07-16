from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple
import numpy as np
import os

import tensorflow as tf
from pprint import pprint

XYData = namedtuple("XYData", ["Xs", "Ys", "XMasks", "YMasks", "type_masks"])

def train_model(graph, var_dict, train_data, max_epoch, hyper_param, 
                output_dir, test_data=None, ex_printer=None, session=None,
                train_support_data=None, dev_support_data=None):
    """ train a model with provided data """
    learning_rate = hyper_param["learning_rate"]
    batch_size = hyper_param["batch_size"]
    log_file = os.path.join(output_dir, "train.log") if output_dir is not None else None

    with graph.as_default():
        # the saver to keep model
        saver = tf.train.Saver()
        last_best_accuracy = 0.

        # place holders for the model
        train_inputs = graph.get_tensor_by_name(var_dict["train_inputs"])
        train_outputs = graph.get_tensor_by_name(var_dict["train_outputs"])

        seq2seq_feed_previous = graph.get_tensor_by_name(var_dict["seq2seq_feed_previous"])
        input_mask = graph.get_tensor_by_name(var_dict["train_input_mask"])
        output_mask = graph.get_tensor_by_name(var_dict["train_output_mask"])
        type_masks = graph.get_tensor_by_name(var_dict["type_masks"])

        if 'train_inputs_2' in var_dict:
            train_inputs_2 = graph.get_tensor_by_name(var_dict["train_inputs_2"])
            train_outputs_2 = graph.get_tensor_by_name(var_dict["train_outputs_2"])

            seq2seq_feed_previous_2 = graph.get_tensor_by_name(var_dict["seq2seq_feed_previous_2"])
            input_mask_2 = graph.get_tensor_by_name(var_dict["train_input_mask_2"])
            output_mask_2 = graph.get_tensor_by_name(var_dict["train_output_mask_2"])
            type_masks_2 = graph.get_tensor_by_name(var_dict["type_masks_2"])


        input_switch = graph.get_tensor_by_name(var_dict["input_switch"]) if 'input_switch' in var_dict else None

        # operaters needed for testing
        token_accuracy = graph.get_tensor_by_name(var_dict["token_accuracy"])
        sentence_accuracy = graph.get_tensor_by_name(var_dict["sentence_accuracy"])
        total_loss = graph.get_tensor_by_name(var_dict["total_loss"])

        global_step = tf.placeholder(tf.int32, name="global_step")
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

        # gradient processing
        grad_clip_norm = hyper_param["gradient_clip_norm"] if "gradient_clip_norm" in hyper_param else None
        grad_noise = hyper_param["gradient_noise"] if "gradient_noise" in hyper_param else None
        grad_noise_gamma = hyper_param["gradient_noise_gamma"] if "gradient_noise_gamma" in hyper_param else None
        
        grads_and_vars = optimizer.compute_gradients(total_loss)
        (grads, variables) = zip(*grads_and_vars)

        if grad_clip_norm:
            print("clipping norm: {}".format(grad_clip_norm))
            capped_grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
            grads_and_vars = zip(capped_grads, variables)

        if grad_noise:
            if grad_noise_gamma:
                grad_noise /= tf.pow(1.0 + tf.to_float(global_step), grad_noise_gamma)
            grads_tmp = []
            for g in grads:
                if g is not None:
                    noisy_grad = g + tf.sqrt(grad_noise)*tf.random_normal(tf.shape(g))
                    grads_tmp.append(noisy_grad)
                else:
                    grads_tmp.append(g)
            print("noise added")
            grads_and_vars = zip(grads_tmp, variables)

        train_step = optimizer.apply_gradients(grads_and_vars)

        session = tf.Session() if session is None else session

        with session.as_default():
            if log_file is not None:
                # initialize the logfile
                output_log = open(log_file, "w")
                log_header = "tr_tok_acc, test_tok_acc, tr_sen_acc, test_sen_acc, tr_loss, test_loss"
                if test_data is None:
                    log_header = "tr_tok_acc, tr_sen_acc, tr_loss" 
                output_log.write(log_header)
                output_log.write("\n")

            session.run(tf.global_variables_initializer())
            nbatches = int(np.ceil(len(train_data.Xs) / float(batch_size)))
            num_meta_example = hyper_param["num_meta_example"]

            for n in range(max_epoch):
                print("================ epoch %d ==================" % n)
                print("PROGRESS: 00.00%")

                tr_token_accuracies = []
                tr_sentence_accuracies = []
                tr_losses = []

                for i in range(nbatches):
                    left = i * batch_size
                    right = min((i + 1) * batch_size, len(train_data.Xs))

                    Xt = train_data.Xs[left: right]
                    Yt = train_data.Ys[left: right]
                    XMasks = train_data.XMasks[left: right]
                    YMasks = train_data.YMasks[left: right]
                    ty_masks = train_data.type_masks[:, left: right, :]

                    Xt_support = np.concatenate([train_support_data[isupport].Xs[left: right] for isupport in range(num_meta_example)])
                    Yt_support = np.concatenate([train_support_data[isupport].Ys[left: right] for isupport in
                         range(num_meta_example)])
                    XMasks_support = np.concatenate([train_support_data[isupport].XMasks[left: right] for isupport in
                         range(num_meta_example)])
                    YMasks_support = np.concatenate([train_support_data[isupport].YMasks[left: right] for isupport in
                         range(num_meta_example)])
                    ty_masks_support =  np.concatenate([train_support_data[isupport].type_masks[:, left: right, :] for isupport in range(num_meta_example)], axis=1)

                    training_result = session.run([ token_accuracy,
                                                    sentence_accuracy, 
                                                    total_loss,
                                                    train_step ],
                                                  feed_dict={train_inputs: Xt_support, train_inputs_2: Xt,
                                                             train_outputs: Yt_support, train_outputs_2: Yt,
                                                             input_mask: XMasks_support, input_mask_2: XMasks,
                                                             output_mask: YMasks_support, output_mask_2: YMasks,
                                                             type_masks: ty_masks_support, type_masks_2: ty_masks,
                                                             seq2seq_feed_previous: False, seq2seq_feed_previous_2: False,
                                                             global_step: n,
                                                             input_switch: True})

                    tr_token_accuracies.append(training_result[0])
                    tr_sentence_accuracies.append(training_result[1])
                    tr_losses.append(training_result[2])

                print("training_loss = {:.5f}".format(np.mean(tr_losses)))
                print("train_token_accuracy = {:.5f}".format(np.mean(tr_token_accuracies)))
                print("train_sentence_accuracy = {:.5f}".format(np.mean(tr_sentence_accuracies)))

                if test_data is not None:
                    test_result = test_model(graph, var_dict, session, test_data, batch_size, 
                                             ex_printer=ex_printer if (n % 30 == 0 or n == max_epoch - 1) else None,
                                             test_support_data=dev_support_data[:num_meta_example])
                    
                    test_token_accuracy = test_result[0]
                    test_sentence_accuracy = test_result[1]
                    test_loss = test_result[2]

                    log_str = "{}, {}, {}, {}, {}, {}".format(
                                np.mean(tr_token_accuracies), test_token_accuracy,
                                np.mean(tr_sentence_accuracies), test_sentence_accuracy,
                                np.mean(tr_losses), test_loss)
                else:
                    log_str = "{}, {}, {}".format(np.mean(tr_token_accuracies),
                                                  np.mean(tr_sentence_accuracies),
                                                  np.mean(tr_losses))

                if output_dir:
                    if test_data is not None:
                        current_seq_acc = test_sentence_accuracy
                    else:
                        current_seq_acc = np.mean(tr_sentence_accuracies)

                    if current_seq_acc > last_best_accuracy:
                        last_best_accuracy = current_seq_acc
                        # add global step so that we can keep multiple models around
                        saver.save(session, os.path.join(output_dir, "table_nl_prog"), global_step=n)

                if log_file is not None:
                    # write corresponding data to log
                    output_log.write(log_str)
                    output_log.write("\n")

                if np.mean(tr_losses) < 0.05:
                    break

    return session


def test_model(graph, var_dict, session, test_data, batch_size, ex_printer=None, log_file=None,
               test_support_data=None):
    with graph.as_default():
        # place holders for the model
        train_inputs = graph.get_tensor_by_name(var_dict["train_inputs"])
        train_outputs = graph.get_tensor_by_name(var_dict["train_outputs"])

        seq2seq_feed_previous = graph.get_tensor_by_name(var_dict["seq2seq_feed_previous"])
        input_mask = graph.get_tensor_by_name(var_dict["train_input_mask"])
        output_mask = graph.get_tensor_by_name(var_dict["train_output_mask"])

        train_inputs_2 = graph.get_tensor_by_name(var_dict["train_inputs_2"])
        train_outputs_2 = graph.get_tensor_by_name(var_dict["train_outputs_2"])

        seq2seq_feed_previous_2 = graph.get_tensor_by_name(var_dict["seq2seq_feed_previous_2"])
        input_mask_2 = graph.get_tensor_by_name(var_dict["train_input_mask_2"])
        output_mask_2 = graph.get_tensor_by_name(var_dict["train_output_mask_2"])
        input_switch = graph.get_tensor_by_name(var_dict["input_switch"]) if 'input_switch' in var_dict else None
        type_masks = graph.get_tensor_by_name(var_dict["type_masks"])
        type_masks_2 = graph.get_tensor_by_name(var_dict["type_masks_2"])

        # operaters needed for testing
        token_accuracy = graph.get_tensor_by_name(var_dict["token_accuracy"])
        sentence_accuracy = graph.get_tensor_by_name(var_dict["sentence_accuracy"])
        total_loss = graph.get_tensor_by_name(var_dict["total_loss"])
        predicted_labels = graph.get_tensor_by_name(var_dict["predicted_labels"])

        num_meta_example = len(test_support_data) if test_support_data else 0

        with session.as_default():
            print("PROGRESS: 00.00%")
            # adding this so that we can include the last one as well
            nbatches = int(np.ceil(len(test_data.Xs) / float(batch_size)))

            test_tok_acc = []
            test_seq_acc = []
            test_loss = []

            if log_file is not None:
                # initialize the logfile
                pred_log_f = open(log_file, "w")

            for i in range(nbatches):
                left = i * batch_size
                right = min((i + 1) * batch_size, len(test_data.Xs))

                Xtest = test_data.Xs[left: right]
                XMasks = test_data.XMasks[left: right]
                ty_masks = test_data.type_masks[:, left: right, :]

                Xtest_support = np.concatenate(
                    [test_support_data[isupport].Xs[left: right] for isupport in
                     range(num_meta_example)])
                XMasks_support = np.concatenate(
                    [test_support_data[isupport].XMasks[left: right] for isupport in
                     range(num_meta_example)])
                ty_masks_support = np.concatenate(
                    [test_support_data[isupport].type_masks[:, left: right, :] for isupport in
                     range(num_meta_example)], axis=1)

                     # in some cases we don't have the access to the test data
                if test_data.Ys is not None and test_data.YMasks is not None:
                    Ytest = test_data.Ys[left: right]
                    YMasks = test_data.YMasks[left: right]

                    Ytest_support = np.concatenate(
                            [test_support_data[isupport].Ys[left: right] for isupport in
                         range(num_meta_example)])
                    YMasks_support = np.concatenate(
                            [test_support_data[isupport].YMasks[left: right] for isupport in range(num_meta_example)])

                    #### HERE feed prvious set to true to make it work #####
                    result = session.run([ token_accuracy,
                                           sentence_accuracy, 
                                           total_loss,
                                           predicted_labels ],
                                           feed_dict={train_inputs: Xtest_support, train_inputs_2: Xtest,
                                                    train_outputs: Ytest_support, train_outputs_2: Ytest,
                                                    input_mask: XMasks_support, input_mask_2: XMasks,
                                                    output_mask: YMasks_support, output_mask_2: YMasks,
                                                    type_masks: ty_masks_support, type_masks_2: ty_masks,
                                                    seq2seq_feed_previous: True,
                                                    seq2seq_feed_previous_2: True,
                                                    input_switch: True})
                    test_tok_acc.append(result[0])
                    test_seq_acc.append(result[1])
                    test_loss.append(result[2])
                    predictions = result[3]
                else:
                    result = session.run([ predicted_labels ],
                                           feed_dict={ train_inputs : Xtest,  
                                                       input_mask : XMasks,
                                                       type_masks: ty_masks,
                                                       seq2seq_feed_previous : True })
                    predictions = result[0]

                if ex_printer is not None and log_file is not None:
                    
                    if test_data.Ys is not None:
                        outs = ex_printer.print(Xtest, Ytest, predictions, k=999999)
                    else:
                        outs = ex_printer.print(Xtest, None, predictions, k=999999)
                    
                    for e in outs:
                        pred_log_f.write(str(e[0]))
                        pred_log_f.write("\n")
                        pred_log_f.write(str(e[1]))
                        pred_log_f.write("\n")
                        pred_log_f.write(str(e[2]))
                        pred_log_f.write("\n")
                        pred_log_f.write("\n")
            
            if test_loss:
                print("test_loss = {:.5f}".format(np.mean(test_loss)))
                print("test_token_accuracy = {:.5f}".format(np.mean(test_tok_acc)))
                print("test_sentence_accuracy = {:.5f}".format(np.mean(test_seq_acc)))

            if ex_printer is not None:
                print("")
                print("[[Test Examples]]")
                if test_data.Ys is not None:
                    outs = ex_printer.print(Xtest, Ytest, predictions, k=5)
                else:
                    outs = ex_printer.print(Xtest, None, predictions, k=5)
                print("")
    return np.mean(test_tok_acc), np.mean(test_seq_acc), np.mean(test_loss)



def load_model(model_file):
    model_dir = os.path.dirname(model_file)
    model_name = os.path.basename(model_file)

    print("[OK] Start loading graph from {} ...".format(model_file))

    session = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(model_dir, model_name + ".meta"))
    graph = tf.get_default_graph()

    print("[OK] Successfully load the graph.")

    saver.restore(session, os.path.join(model_dir, model_name))
    
    print("[OK] Successfully load variable weights.")

    return graph, session

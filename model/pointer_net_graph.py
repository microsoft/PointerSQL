from __future__ import absolute_import
from __future__ import print_function

from model.pointer_net import *
from model.pointer_net_helper import *

from pprint import pprint

default_var_dict = {
    "train_inputs" : "graph/train_inputs:0",
    "train_outputs": "graph/train_outputs:0",
    "train_input_mask": "graph/train_input_mask:0",
    "train_output_mask": "graph/train_output_mask:0",
    "seq2seq_feed_previous": "graph/seq2seq_feed_previous:0",
    "token_accuracy": "graph/token_accuracy:0",
    "sentence_accuracy": "graph/sentence_accuracy:0",
    "predicted_labels": "graph/predicted_labels:0",
    "total_loss": "graph/total_loss:0",
    "type_masks": "graph/type_masks:0"
}

def build_graph(decoder_type, explicit_pointer, value_based_loss, 
                hyper_param, pnet_vocab, pretrained_enc_embedding=None, 
                multi_encoders=None, old_graph=None, scope=None):
    """ Build a graph """
    # hyper parameter
    embedding_size = hyper_param["embedding_size"]
    #batch_size = hyper_param["batch_size"] # fixed batch size when building graph
    batch_size = None # mutable batch size data
    n_hidden = hyper_param["n_hidden"]
    num_layers = hyper_param["num_layers"]
    learning_rate = hyper_param["learning_rate"]
    dropout_keep_prob = hyper_param["dropout_keep_prob"]
    encoder_merge_method = hyper_param["encoder_merge_method"]

    input_vocab, output_vocab, X_maxlen, Y_maxlen = pnet_vocab.get_all()

    graph = tf.Graph()

    # scope should only be used after graph is defined
    with graph.as_default(), tf.variable_scope(scope or "graph") as scope:
        train_inputs = tf.placeholder(tf.int32, [batch_size, X_maxlen], name="train_inputs")
        # the mask for identifying padding tokens from the sentence
        train_input_mask = tf.placeholder(tf.bool, [batch_size, X_maxlen], name="train_input_mask")
        train_outputs = tf.placeholder(tf.int32, [batch_size, Y_maxlen], name="train_outputs")
        train_output_mask = tf.placeholder(tf.bool, [batch_size, Y_maxlen], name="train_output_mask")
        type_masks = tf.placeholder(tf.bool, [2, batch_size, X_maxlen], name="type_masks")

        batch_size = tf.shape(train_inputs)[0]

        seq2seq_feed_previous = tf.placeholder(tf.bool, name="seq2seq_feed_previous")

        if pretrained_enc_embedding is not None:
            enc_embedding = tf.get_variable("enc_embedding", 
                                            shape=[input_vocab.size, embedding_size],
                                            initializer=tf.constant_initializer(pretrained_enc_embedding),
                                            trainable=False)
        else:
            enc_embedding = tf.get_variable("enc_embedding", 
                                            initializer=tf.random_uniform([input_vocab.size, embedding_size], -1.0,1))

        fw_enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(n_hidden) for _ in range(num_layers)])
        bw_enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(n_hidden) for _ in range(num_layers)])
        dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(n_hidden) for _ in range(num_layers)])

        if dropout_keep_prob < 1:
            t_dropout_keep_prob = tf.cond(seq2seq_feed_previous, lambda: 1., lambda: dropout_keep_prob)
            # the cell used in the model
            fw_enc_cell = tf.contrib.rnn.DropoutWrapper(fw_enc_cell, output_keep_prob=t_dropout_keep_prob)
            bw_enc_cell = tf.contrib.rnn.DropoutWrapper(bw_enc_cell, output_keep_prob=t_dropout_keep_prob)
            dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=t_dropout_keep_prob)

        encoder_inputs = [train_inputs[:,k] for k in range(0, X_maxlen)]
        
        # the first token for decoder inputs is always the <GO> token and the last one from train outputs is not needed
        decoder_inputs = ([tf.fill([batch_size], output_vocab.word_to_index(Vocabulary.GO_TOK))] 
                            + [train_outputs[:,k] for k in range(0, Y_maxlen)][0:-1])

        # embed a raw vector into a 32 value vector
        enc_embeds = [tf.nn.embedding_lookup(enc_embedding, enc_input) for enc_input in encoder_inputs]
        stacked_enc_embeds = tf.transpose(tf.stack(enc_embeds), [1,2,0])

        # embed a raw vector into a 32 value vector
        dec_embedding = tf.get_variable("dec_embedding", initializer=tf.random_uniform([output_vocab.size, embedding_size],-1.0,1))
        # perform lookup based on types of the last decoder cell, note that <GO> symol is always looked up by embedding 
        dec_embeds = [tf.nn.embedding_lookup(dec_embedding, x) if i == 0 or decoder_type[i-1].ty is DecoderType.Projector 
                      else tf.squeeze(tf.matmul(stacked_enc_embeds,
                                                tf.expand_dims(tf.one_hot(x, stacked_enc_embeds.get_shape()[-1]), -1)), -1) 
                      for i, x in enumerate(decoder_inputs)]

        print("Loss function is {}".format(value_based_loss))

        reshaped_input_mask = tf.unstack(tf.transpose(train_input_mask, [1,0]))

        def feed_prev_func(prev, current_decoder_type):
            # the feeding function
            if current_decoder_type.ty is DecoderType.Projector:
                # in this case prev is the output vector repr
                prev_symbol = math_ops.argmax(prev, 1)
                emb_prev = tf.nn.embedding_lookup(dec_embedding, prev_symbol)
            elif current_decoder_type.ty is DecoderType.Pointer:
                # in this case prev is the energy function for pointers 

                logits_n_inputs = tf.concat([tf.expand_dims(prev,-2),
                                             tf.cast(tf.expand_dims(train_inputs,-2), tf.float32)], 
                                            -2)

                if value_based_loss == "sum_vloss":
                    transferred_distrib = tf.map_fn(lambda x: tf.unsorted_segment_sum(x[0], tf.cast(x[1], tf.int32), input_vocab.size), logits_n_inputs)
                elif value_based_loss == "max_vloss" or value_based_loss == "ploss":
                    transferred_distrib = tf.map_fn(lambda x: tf.unsorted_segment_max(x[0], tf.cast(x[1], tf.int32), input_vocab.size), logits_n_inputs)

                emb_prev = tf.nn.embedding_lookup(enc_embedding, tf.argmax(transferred_distrib,-1))
                return emb_prev
            else:
                raise Exception('not a expected type')

            return emb_prev

        outs = pointer_network(enc_embeds, dec_embeds, 
                               fw_enc_cell, bw_enc_cell, dec_cell, 
                               decoder_type, output_vocab.size,
                               encoder_masks=reshaped_input_mask, 
                               feed_prev=seq2seq_feed_previous,
                               loop_function=feed_prev_func,
                               multi_encoders=multi_encoders,
                               encoder_merge_method=encoder_merge_method)


        # split train outs into train outs for projection and train outs for pointer
        # and concretize pointer in the tensor into labels
        pointer_train_outs, proj_train_outs = split_by_type(train_outputs, decoder_type, axis=1)
        pointer_out_mask, proj_out_mask = split_by_type(train_output_mask, decoder_type, axis=1)
        pntr_dec_types = [x for x in decoder_type if x.ty == DecoderType.Pointer]

        def _process_outs(outs, target_type):
            """ reshape outs into shape (batch_size, type_num, X_maxlen) for the purpose of computing """
            cnt = len([i for i in range(len(outs)) if decoder_type[i].ty is target_type])
            if cnt == 0:
                target_outs = tf.reshape([], [batch_size, cnt, X_maxlen])
            else:
                trimed_outs = [outs[i] for i in range(len(outs)) if decoder_type[i].ty is target_type]
                # outputs from the seq2seq model is already logits
                target_outs = trimed_outs
            return target_outs


        proj_outs = tf.transpose(_process_outs(outs, DecoderType.Projector), perm=[1,0,2])

        pointer_outs = _process_outs(outs, DecoderType.Pointer)
        pointer_outs = [pointer_outs[i] * tf.cast(type_masks[pntr_dec_types[i].mask_name], tf.float32) for i in range(len(pntr_dec_types))]
        pointer_outs = tf.transpose(pointer_outs, perm=[1,0,2])
        
        # predictions made by the neural network
        pointer_predictions = tf.nn.softmax(pointer_outs)

        proj_predictions = tf.nn.softmax(proj_outs)

        # labels predicted from the result (sharpen), get encoder symbols from the 
        proj_predicted_labels = tf.cast(tf.argmax(proj_predictions, axis=-1), tf.int32)

        # prepare data for energy transfer
        copied_train_inputs = tf.transpose(tf.stack([train_inputs for x in range(int(pointer_train_outs.get_shape()[-1]))]), [1, 0, 2])
        merged_logits_inputs = tf.concat([tf.expand_dims(pointer_predictions,-2),
                                          tf.cast(tf.expand_dims(copied_train_inputs,-2), tf.float32)], 
                                         -2)

        # transfer distribution over pointer to distribution over encoder symbols
        if value_based_loss == "sum_vloss":
            distrib_over_encoder_symbols = tf.map_fn(lambda y: tf.map_fn(lambda x: tf.unsorted_segment_sum(x[0], tf.cast(x[1], tf.int32), input_vocab.size), y), 
                                                 merged_logits_inputs)
        elif value_based_loss == "max_vloss" or value_based_loss == "ploss":
            distrib_over_encoder_symbols = tf.map_fn(lambda y: tf.map_fn(lambda x: tf.unsorted_segment_max(x[0], tf.cast(x[1], tf.int32), input_vocab.size), y), 
                                                 merged_logits_inputs)

        labels_by_pointers = tf.cast(tf.argmax(distrib_over_encoder_symbols, axis=-1), tf.int32)

        predicted_labels = assemble_by_type(labels_by_pointers, proj_predicted_labels, decoder_type, axis=1)

        if explicit_pointer:
            pointer_predicted_labels = tf.cast(tf.argmax(pointer_predictions, axis=-1), tf.int32)
            # compute accuracy with pointers, since pointers are explicitly provided in the dataset
            token_accuracy, sentence_accuracy = compute_accuracy(pointer_predicted_labels, pointer_train_outs, 
                                                                 proj_predicted_labels, proj_train_outs, 
                                                                 pointer_out_mask, proj_out_mask)
        else:
            # compute accuracy with concrete labels instead of pointers 
            # (computing based on labels is better since we only care about the final result)
            token_accuracy, sentence_accuracy = compute_accuracy(labels_by_pointers, 
                                                                 pointer_to_label(pointer_train_outs, train_inputs), 
                                                                 proj_predicted_labels, proj_train_outs, 
                                                                 pointer_out_mask, proj_out_mask)

        token_accuracy = tf.identity(token_accuracy, name="token_accuracy")
        sentence_accuracy = tf.identity(sentence_accuracy, name="sentence_accuracy")
        predicted_labels = tf.identity(predicted_labels, name="predicted_labels")

        # the loss function
        if value_based_loss == "sum_vloss" or value_based_loss == "max_vloss":
            # loss based on probability of 
            one_hot_train_out = tf.one_hot(pointer_to_label(pointer_train_outs, train_inputs), input_vocab.size, axis=-1)
            # this clip is used to handle nan problem in calculation
            pntr_losses = -tf.reduce_sum(one_hot_train_out * tf.log(tf.clip_by_value(distrib_over_encoder_symbols,1e-10,1.0)), -1)
        elif value_based_loss == "ploss":
            pntr_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pointer_train_outs, logits=pointer_outs)
        else:
            print("Loss function ({}) can not be recognized, exiting...".format(value_based_loss))
            sys.exit(-1)

        # TODO: add a mask to remove losses from padding symbols
        proj_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=proj_train_outs, logits=proj_outs)

        # the loss hould be normalized by the total number of unmasked cells
        total_loss = tf.add(tf.reduce_sum(tf.multiply(pntr_losses, tf.cast(pointer_out_mask, tf.float32))),
                            tf.reduce_sum(tf.multiply(proj_losses, tf.cast(proj_out_mask, tf.float32))),
                            name="total_loss")

        var_dict = {
            "train_inputs" : train_inputs.name,
            "train_outputs": train_outputs.name,
            "train_input_mask": train_input_mask.name,
            "train_output_mask": train_output_mask.name,
            "seq2seq_feed_previous": seq2seq_feed_previous.name,
            "token_accuracy": token_accuracy.name,
            "sentence_accuracy": sentence_accuracy.name,
            "predicted_labels": predicted_labels.name,
            "total_loss": total_loss.name,
            "type_masks": type_masks.name
        }

    return graph, var_dict


from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
import random
import os
from pprint import pprint
import copy

from model import pointer_net_helper as pnet_helper
from model import pointer_net as pnet
from model.pointer_net import DecoderType
from model import pointer_net_graph as pnet_graph
from model.pointer_net_helper import PnetVocab

from model import learn as learn
from model.util import Vocabulary, JmtEmbeddings, GloVeEmbeddings

import app.util

import tensorflow as tf

def to_xydata(dataset, pnet_vocab, explicit_pointer, decoder_type):
    # prepare xy data from a dataset
    return learn.XYData(*pnet_vocab.prepare_pointer_data(dataset, decoder_type, explicit_pointer))

def concat_in1_in2(input1, input2, X1_maxlen, X2_maxlen):
    """ input1 and input2 are both lists of tokens, 
        we concatenate them into a single input list X """
    result = []
    for i in range(len(input1)):
        result.append(input1[i])
    for i in range(len(input1), X1_maxlen):
        result.append(Vocabulary.END_TOK)

    for i in range(len(input2)):
        result.append(input2[i])
    for i in range(len(input2), X2_maxlen):
        result.append(Vocabulary.END_TOK)
    return result


class NL2Prog(object):
    """ Transform NL text to programs """

    def __init__(self, train_file, dev_file, test_file, settings):
        """ prepare memlookup task """
        decoder_regex = settings["decoder_regex"]
        explicit_pointer = settings["explicit_pointer"]
        value_based_loss = settings["value_based_loss"]
        num_data_used = settings["num_data_used"]
        test_only = settings["test_only"]

        _cut = lambda s: s[:min(len(s), num_data_used)] if num_data_used is not None else s

        # prepare training and dev data
        train_dataset = _cut(app.util.read_dataset(train_file, input_num=2))

        print(len(train_dataset))

        if train_file == dev_file == test_file: # evluation mode
                dev_dataset = [e for e in train_dataset]
                test_dataset = [e for e in dev_dataset]
        else:
            if dev_file != train_file:
                dev_dataset = _cut(app.util.read_dataset(dev_file, input_num=2))
            else:
                split_index = int(0.8 * len(train_dataset))
                dev_dataset = train_dataset[split_index + 1:]
                train_dataset = train_dataset[:split_index]

            if test_file != train_file and test_file != dev_file:
                test_dataset =  _cut(app.util.read_dataset(test_file, input_num=2))
            else:
                test_dataset = [e for e in dev_dataset]

        print(len(train_dataset))

        all_dataset = train_dataset + dev_dataset + test_dataset

        if settings["X1_maxlen"] and settings["X2_maxlen"] and settings["Y_maxlen"]:
            X1_maxlen = settings["X1_maxlen"]
            X2_maxlen = settings["X2_maxlen"]
            Y_maxlen = settings["Y_maxlen"]
        else:
            X1_maxlen = max(map(lambda x: len(x["in1"]), all_dataset))  
            X2_maxlen = max(map(lambda x: len(x["in2"]), all_dataset)) 
            Y_maxlen = max(map(lambda x: len(x["out"]), all_dataset))

        # add 1 to enable the language to deal with pointing to the end
        X_maxlen = X1_maxlen + X2_maxlen + 1
        
        print("[XY] X1_maxlen = {}, X2_maxlen = {}, X_maxlen = {}, Y_maxlen = {}".format(X1_maxlen, X2_maxlen, X_maxlen, Y_maxlen))

        col_pntr_mask_fn = lambda X: np.array([True if (x in X[:X1_maxlen] or x == X[-1]) else False for x in X])
        const_pntr_mask_fn = lambda X: np.array([True if i >= X1_maxlen else False for i in range(X_maxlen)])

        true_mask_fn = lambda X: np.array([True for i in range(X_maxlen)])

        abbrv_dict = {
            "w": DecoderType(DecoderType.Projector),
            "p": DecoderType(DecoderType.Pointer, 0, col_pntr_mask_fn), #col_pntr_mask_fn),
            "q": DecoderType(DecoderType.Pointer, 1, const_pntr_mask_fn) 
        }

        self.decoder_type = DecoderType.from_regex(decoder_regex, abbrv_dict=abbrv_dict, maxlen=Y_maxlen)

        for entry in all_dataset:
            entry["in"] = concat_in1_in2(entry["in1"], entry["in2"], X1_maxlen, X2_maxlen)

        # get all used words as vocab
        if test_only:
            input_vocab = Vocabulary.build_from_sentences([entry["in"] for entry in test_dataset])            
            # we have a bug not yet fixed....
            output_vocab = Vocabulary.build_from_sentences([[ entry["out"][i] for i in range(len(entry["out"])) 
                                                              if self.decoder_type[i].ty == DecoderType.Projector ] 
                                                              for entry in all_dataset], use_go_tok=True)
        else:
            input_vocab = Vocabulary.build_from_sentences([entry["in"] for entry in all_dataset])
            output_vocab = Vocabulary.build_from_sentences([[ entry["out"][i] for i in range(len(entry["out"])) 
                                                             if self.decoder_type[i].ty == DecoderType.Projector ] 
                                                            for entry in all_dataset], use_go_tok=True)


        pnet_vocab = PnetVocab(input_vocab, output_vocab, X_maxlen, Y_maxlen, [X1_maxlen, X2_maxlen])

        pprint(pnet_vocab.get_stats())

        # build vocabulary
        self.pnet_vocab = pnet_vocab

        self.ex_printer = pnet_helper.PnetExPrinter(self.decoder_type, self.pnet_vocab)

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.explicit_pointer = explicit_pointer
        self.value_based_loss = value_based_loss

        self.multi_encoders = {
            "merge_method": "sequential",
            "segments": [[i for i in range(0, X1_maxlen)], 
                         [i for i in range(X1_maxlen, X_maxlen)]]    
        }


    def train(self, hyper_param, max_epoch, output_dir=None, pretrained_emb=None):
        print("[OK] Start building graph ...")

        graph, var_dict = pnet_graph.build_graph(self.decoder_type, self.explicit_pointer, 
                                                 self.value_based_loss, hyper_param, self.pnet_vocab,
                                                 pretrained_enc_embedding=pretrained_emb,
                                                 multi_encoders=self.multi_encoders)
        
        print("[OK] Successfully build the graph.")

        train_xydata = to_xydata(self.train_dataset, self.pnet_vocab, self.explicit_pointer, self.decoder_type)
        dev_xydata = to_xydata(self.dev_dataset, self.pnet_vocab, self.explicit_pointer, self.decoder_type)

        session = learn.train_model(graph, var_dict, train_xydata, max_epoch, hyper_param, 
                                   output_dir, test_data=dev_xydata, ex_printer=self.ex_printer)

        print("[OK] Training finished.")

        return graph, session


    def test(self, graph, session, hyper_param, output_dir=None):
        #pprint([n.name for n in graph.get_operations()])
        test_xydata = to_xydata(self.test_dataset, self.pnet_vocab, self.explicit_pointer, self.decoder_type)
        # use the default dict
        var_dict = pnet_graph.default_var_dict
        
        log_file = os.path.join(output_dir, "test_top_1.log") if output_dir else None
        learn.test_model(graph, var_dict, session, test_xydata, hyper_param["batch_size"], 
                         self.ex_printer, log_file=log_file)
        
        print("[OK] Test finished.")

    def test_new(self, graph, session, hyper_param, pretrained_emb, output_dir=None):
        # we need to re build embedding in this case
        with graph.as_default():
            embedding_tensor = graph.get_tensor_by_name("graph/enc_embedding:0")

            original_vocab_size = int(embedding_tensor.get_shape()[0])

            vocab_size = self.pnet_vocab.input_vocab.size
            emb_size = hyper_param["embedding_size"]

            # pad the new embedding as the same size as the original emb tensor
            new_enc_emb = tf.constant(pretrained_emb, name="new_enc_emb", shape=[vocab_size, emb_size])
            if original_vocab_size > vocab_size:
                new_enc_emb = tf.pad(new_enc_emb, [[0, original_vocab_size - vocab_size],[0, 0]])

            print(new_enc_emb)

            re_assign_emb = tf.assign(embedding_tensor, new_enc_emb)
            session.run(re_assign_emb)

        self.test(graph, session, hyper_param, output_dir=output_dir)


    @staticmethod
    def run_wikisql(input_dir, output_dir, config, trained_model=None):
        """ if trained_model is specified, we will perform a model testing """
        train_file = app.util.find_file(input_dir, config["train_file"]) 
        dev_file = app.util.find_file(input_dir, config["dev_file"])
        test_file = app.util.find_file(input_dir, config["test_file"])

        jmt_embedding_file = app.util.find_file(input_dir, config["jmt_embedding_file"])
        glove_embedding_file = app.util.find_file(input_dir, config["glove_embedding_file"])

        hyper_param = config["hyper_param"]
        max_epoch = hyper_param["max_epoch"]
        num_data_used = hyper_param["num_data_used"]
        
        print("|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("[Config]")
        pprint(config)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<|")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<|")

        settings = {
            "decoder_regex": config["decoder_regex"],
            "explicit_pointer": config["explicit_pointer"],
            "value_based_loss": config["value_based_loss"],
            "num_data_used": hyper_param["num_data_used"],
            "test_only": False if trained_model is None else True,

            "X1_maxlen": hyper_param["X1_maxlen"],
            "X2_maxlen": hyper_param["X2_maxlen"],
            "Y_maxlen": hyper_param["Y_maxlen"]
        }

        task = NL2Prog(train_file=train_file,
                       dev_file=dev_file,
                       test_file=test_file,
                       settings=settings)

        pretrained_emb = None

        if jmt_embedding_file and glove_embedding_file:
            input_vocab = task.pnet_vocab.input_vocab
            pretrained_emb = np.zeros([input_vocab.size, hyper_param["embedding_size"]], dtype=np.float32)

            jmt_dim = hyper_param["jmt_embedding_size"]
            glove_dim = hyper_param["glove_embedding_size"]

            if jmt_dim > 0:
                full_jmt_embeddings = JmtEmbeddings(jmt_embedding_file, d_emb=jmt_dim)   
            
            if glove_dim > 0:
                full_glove_embeddings = GloVeEmbeddings(glove_embedding_file, d_emb=glove_dim)         

            for i in input_vocab.index_word:
                w = input_vocab.index_word[i]
                v1 = full_jmt_embeddings.emb(w)

                if jmt_dim > 0:
                    # jmt part
                    for j in range(len(v1)):
                        pretrained_emb[i][j] = v1[j]

                if glove_dim > 0:
                    # glove part
                    v_list = []
                    w = (w.replace("(", "^").replace(")", "^").replace(",", "^")
                          .replace(".", "^").replace("\'", "^").replace("/", "^"))
                    
                    for x in w.split("^"):
                        if x == "":
                            continue
                        v_x = full_glove_embeddings.emb(x)
                        if np.array(v_x).any():
                            v_list.append(v_x)

                    if v_list:
                        v2 = np.mean(v_list, axis=0)
                        for j in range(len(v2)):
                            pretrained_emb[i][j+jmt_dim] = v2[j]

        if trained_model is None:
            # in this case we want to train a new model
            graph, session = task.train(hyper_param=hyper_param,
                                        max_epoch=max_epoch, 
                                        output_dir=output_dir,
                                        pretrained_emb=pretrained_emb)

            task.test(graph, session, hyper_param, output_dir=output_dir)

        else:
            # in this case we want to test a trained model
            graph, session = learn.load_model(trained_model)

            task.test_new(graph, session, hyper_param, pretrained_emb, output_dir=output_dir)

            print("[OK] Start building testing graph ...")
        

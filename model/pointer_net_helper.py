from __future__ import absolute_import
from __future__ import print_function

from model.pointer_net import *
from model.util import Vocabulary

from pprint import pprint


class PnetVocab(object):
  """ assembled vocabulary file used by pointer network """

  def __init__(self, input_vocab, output_vocab, X_maxlen, Y_maxlen, X_seg_maxlen):
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.X_maxlen = X_maxlen
    self.Y_maxlen = Y_maxlen

    # the maximum length for different parts of the query
    self.X_seg_maxlen = X_seg_maxlen

  def get_stats(self):
    return {
      "input_vocab_size": self.input_vocab.size,
      "output_vocab_size": self.output_vocab.size,
      "X_maxlen": self.X_maxlen,
      "Y_maxlen": self.Y_maxlen
    }

  @staticmethod
  def build_from_dataset(dataset, decoder_type, frequency_cap=-1):
    """ build a vocabulary out of the provided dataset """
    input_vocab = Vocabulary.build_from_sentences([entry["in"] for entry in dataset],
                                                  use_unk=False,
                                                  frequency_cap=frequency_cap)
    # the one below only considers words in the projectors
    output_vocab = Vocabulary.build_from_sentences([[v for i, v in enumerate(entry["out"])
                                                     if decoder_type[i].ty == DecoderType.Projector] for entry in
                                                    dataset],
                                                   use_go_tok=True)

    X_maxlen = len(dataset[0]["in"])
    Y_maxlen = len(dataset[0]["out"])
    output_length_fixed = True

    for entry in dataset:
      input_len = len(entry["in"])
      if X_maxlen < input_len:
        X_maxlen = input_len

      output_len = len(entry["out"])
      if Y_maxlen != output_len:
        output_length_fixed = False
      if Y_maxlen < output_len:
        Y_maxlen = output_len

    # if output length every input / output will be enhanced with a end-token
    if not output_length_fixed:
      X_maxlen = X_maxlen + 1
      Y_maxlen = Y_maxlen + 1

    return PnetVocab(input_vocab, output_vocab, X_maxlen, Y_maxlen)

  def get_all(self):
    """ get all specs from the vocab """
    return self.input_vocab, self.output_vocab, self.X_maxlen, self.Y_maxlen

  def prepare_input_data_only(self, dataset):
    """ build a vocabulary out of the provided dataset """
    input_vocab = self.input_vocab
    X_maxlen = self.X_maxlen

    Xs = np.zeros((len(dataset), X_maxlen))
    for i in range(len(dataset)):
      for j in range(len(dataset[i]["in"])):
        Xs[i][j] = input_vocab.word_to_index(dataset[i]["in"][j])
      for j in range(len(dataset[i]["in"]), X_maxlen):
        Xs[i][j] = input_vocab.word_to_index(Vocabulary.END_TOK)

    return Xs

  def prepare_pointer_data(self, dataset, decoder_type, explicit_pointer):
    """ Prepare training input output and mask for the task, decoder type specifies the type of the decoder
            Args:
                dataset: the dataset used for training and testing, each entry is a dict containing entry["in"] and entry["out"]
                decoder_type: types of decoders (pointer/projector)
                explicit_pointer: whether pointers are explicited represented in the output sequences.
                    If not, we need to manually convert values to pointers, by looking up values in the output in the input
            Returns:
                prepare data together with some of their arguments
        """
    input_vocab, output_vocab, X_maxlen, Y_maxlen = self.get_all()

    Xs = np.zeros((len(dataset), X_maxlen))
    Ys = np.zeros((len(dataset), Y_maxlen))
    XMasks = np.zeros((len(dataset), X_maxlen), dtype=bool)
    YMasks = np.zeros((len(dataset), Y_maxlen), dtype=bool)

    mask_functions = {}
    for dec_ty in decoder_type:
      if dec_ty.mask_name is not None and dec_ty.mask_name not in mask_functions:
        mask_functions[dec_ty.mask_name] = dec_ty.mask_fn

    type_masks = np.zeros((len(mask_functions), len(dataset), X_maxlen), dtype=bool)

    for i in range(len(dataset)):

      for j in range(len(dataset[i]["in"])):
        Xs[i][j] = input_vocab.word_to_index(dataset[i]["in"][j])
        # in some cases pads are included in the input sequence,
        # we should mask them explicitly as padding tokens
        if Xs[i][j] != input_vocab.word_to_index(Vocabulary.END_TOK):
          XMasks[i][j] = True
        else:
          XMasks[i][j] = False

      # padding: rest of the input are filled with special token
      for j in range(len(dataset[i]["in"]), X_maxlen):
        Xs[i][j] = input_vocab.word_to_index(Vocabulary.END_TOK)
        XMasks[i][j] = False

      # setup type masks
      for k in mask_functions:
        type_masks[k][i] = mask_functions[k](Xs[i])

      if "out" in dataset[i]:
        # note that input to cell j is the output from cell j-1
        for j in range(0, len(dataset[i]["out"])):
          if decoder_type[j].ty == DecoderType.Projector:
            Ys[i][j] = output_vocab.word_to_index(dataset[i]["out"][j])
          elif decoder_type[j].ty == DecoderType.Pointer:
            if not explicit_pointer:
              # if pointers are not explicitly presented in the sequence, we need to manually lookup pointers
              # print(dataset[i]["in"])
              # print(dataset[i]["out"])
              # print("")
              Ys[i][j] = dataset[i]["in"].index(dataset[i]["out"][j])
            else:
              Ys[i][j] = int(dataset[i]["out"][j])
          YMasks[i][j] = True

        # padding: rest of the input are filled with special token
        for j in range(len(dataset[i]["out"]), Y_maxlen):
          if decoder_type[j].ty == DecoderType.Projector:
            Ys[i][j] = output_vocab.word_to_index(Vocabulary.END_TOK)
          elif decoder_type[j].ty == DecoderType.Pointer:
            Ys[i][j] = len(dataset[i]["in"])  # points to the ending of the input token

          if j == len(dataset[i]["out"]):
            YMasks[i][j] = True
          else:
            YMasks[i][j] = False
      else:
        Ys = None
        YMasks = None

    return Xs, Ys, XMasks, YMasks, type_masks


class PnetExPrinter(object):
  """ a printer class used print examples produced by pointer network """
  def __init__(self, decoder_type, pnet_vocab):
    """ create a printer instance based on decoder_type and pnet_vocab """
    self.decoder_type = decoder_type
    self.pnet_vocab = pnet_vocab

  @staticmethod
  def output_seq_to_sentence(output_vec, output_vocab, decoded_input, decoder_type):
    """ Given a sequence of indices, print its represented sentence (according to decoder type)
        """
    return [output_vocab.index_to_word(int(x))
            if decoder_type[i].ty == DecoderType.Projector
            else decoded_input[int(x)]
            for i, x in enumerate(output_vec)]

  @staticmethod
  def output_seq_to_sentence_by_input_vocab(output_vec, output_vocab, input_vocab, decoder_type):
    """ Given a sequence of indices, print its represented sentence (according to decoder type)
        """
    return [output_vocab.index_to_word(int(x))
            if decoder_type[i].ty == DecoderType.Projector
            else input_vocab.index_to_word(int(x))
            for i, x in enumerate(output_vec)]

  def remove_end_suffix(self, sentence):
    result = []
    for x in sentence:
      if x == Vocabulary.END_TOK:
        result.append(x)
        break
      else:
        result.append(x)
    return result

  def print(self, Xs, Ys, predictions, k=5):
    def _remove_padding(sentence, cut_off=True):
      result = []
      for x in sentence:
        if Vocabulary.END_TOK in x:
          if cut_off:
            result.append(x)
            break
          else:
            result.append("")
            continue
        else:
          result.append(x)
      return result

    outputs = []
    for i in range(min(k, len(Xs))):

      ap = self.pnet_vocab.input_vocab.vec_to_sequence(Xs[i])
      x_out = " ".join(_remove_padding(ap, cut_off=False))

      print("X = ", x_out)

      if Ys is not None:
        # Ys is not always available
        bp = PnetExPrinter.output_seq_to_sentence(Ys[i], self.pnet_vocab.output_vocab, ap, self.decoder_type)
        y_out = " ".join(_remove_padding(bp))
        print("Y = ", y_out)
      else:
        y_out = None

      cp = PnetExPrinter.output_seq_to_sentence_by_input_vocab(predictions[i], self.pnet_vocab.output_vocab,
                                                               self.pnet_vocab.input_vocab, self.decoder_type)
      p_out = " ".join(_remove_padding(cp))

      print("P = ", p_out)
      outputs.append((x_out, y_out, p_out))

    return outputs


def assemble_by_type(pointer_tensor, proj_tensor, decoder_type, axis):
  """ merge decoding result by type, (proj/pointer)
        Args:
            pointer_tensor of shape (batch_size, pointer_len, V)
            proj_tensor of shape (batch_size, proj_len, V)
        Returns:
            combined tensor (batch_size, proj_len + pointer_len, V)
    """
  pointer_len = int(pointer_tensor.get_shape()[axis])
  proj_len = int(proj_tensor.get_shape()[axis])

  gather_indices = []
  pointer_next = 0
  proj_next = 0
  for i in range(pointer_len + proj_len):
    if decoder_type[i].ty == DecoderType.Pointer:
      gather_indices.append(pointer_next)
      pointer_next += 1
    elif decoder_type[i].ty == DecoderType.Projector:
      gather_indices.append(proj_next + pointer_len)
      proj_next += 1

  return tf.gather(tf.concat([pointer_tensor, proj_tensor], axis), gather_indices, axis=axis)


def split_by_type(merged_tensor, decoder_type, axis):
  """ Split train_outs data according to decoder type (proj/pointer)
        Args:
            (batch_size, proj_len + pointer_len, V)
        Returns:
            pointer_tensor of shape (batch_size, pointer_len, V)
            proj_tensor of shape (batch_size, proj_len, V)
    """
  seq_length = merged_tensor.get_shape()[axis]

  pointer_indices = [i for i in range(seq_length) if decoder_type[i].ty is DecoderType.Pointer]
  proj_indices = [i for i in range(seq_length) if decoder_type[i].ty is DecoderType.Projector]

  # casts are used to deal with the case of empty indice list
  pointer_indices = tf.cast(pointer_indices, tf.int32)
  proj_indices = tf.cast(proj_indices, tf.int32)

  return tf.gather(merged_tensor, pointer_indices, axis=axis), tf.gather(merged_tensor, proj_indices, axis=axis)


def pointer_to_label(pointers, inputs):
  """ Given a list of pointers, lookup pointers in given inputs
        Args:
            pointers: pointers, each pointer points a location in the input, of shape (batch_size, pointer_len)
            inputs: inputs sequence that the pointer points to, of shape (batch_size, input_len)
        Returns:
            The lookup result for pointers, i.e., dereference every pointer base on the inputs
    """
  onehot_pntr_repr = tf.cast(tf.one_hot(pointers, inputs.get_shape()[-1]),
                             tf.int32)  # of shape (batch_size, output_len, input_len)
  labels = tf.squeeze(tf.matmul(onehot_pntr_repr, tf.expand_dims(inputs, -1)), -1)
  return labels


def compute_accuracy(pntr_pred_label, pntr_train_label, proj_pred_label, proj_train_label, pntr_mask, proj_mask):
  """ Calculate accuracy with the help of mask
    """
  def _prepare_compute_accuracy(pred_label, train_label, mask):
    reversed_mask = tf.logical_not(mask)
    label_eq = tf.equal(pred_label, train_label)

    # a list that indicates whether a unmasked token is corrected predicted or not,
    # to calculate accuracy, perform a reduce over it
    pre_token_accuracy = tf.boolean_mask(label_eq, mask)
    pre_seq_accuracy = tf.reduce_all(tf.logical_or(label_eq, reversed_mask), -1)

    return pre_token_accuracy, pre_seq_accuracy

  pntr_pre_token_accuracy, pntr_pre_seq_accuracy = _prepare_compute_accuracy(pntr_pred_label, pntr_train_label,
                                                                             pntr_mask)
  proj_pre_token_accuracy, proj_pre_seq_accuracy = _prepare_compute_accuracy(proj_pred_label, proj_train_label,
                                                                             proj_mask)

  token_accuracy = tf.reduce_mean(
    tf.cast(tf.concat([pntr_pre_token_accuracy, proj_pre_token_accuracy], axis=-1), tf.float32))
  seq_accuracy = tf.reduce_mean(tf.cast(tf.logical_and(pntr_pre_seq_accuracy, proj_pre_seq_accuracy), tf.float32))

  return token_accuracy, seq_accuracy
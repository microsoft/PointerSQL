import numpy as np

class Vocabulary(object):

    GO_TOK = "<GO>"
    END_TOK = "<END>"

    UNK_TOK = "<UNK>"

    def __init__(self):
        """ initialize an empty vocab """
        self.vocab = []
        self.size = 0
        self.word_index = {}
        self.index_word = {}
        self.use_unk = False

    def word_to_index(self, word):
        if (self.use_unk and word not in self.word_index):
            print("[Warning] Word {} is not found in the vocabulary, <UNK> is used as a substitute.".format(word))
            return self.word_index[Vocabulary.UNK_TOK]
        return self.word_index[word]

    def index_to_word(self, index):
        return self.index_word[index]

    @staticmethod
    def build_from_words(words, use_go_tok=False, use_unk=False):
        """ build vocabulary from all words we want to include """
        vocab = Vocabulary()

        vocab.vocab = [Vocabulary.END_TOK]
        if use_go_tok:
            vocab.vocab.append(Vocabulary.GO_TOK)
        if use_unk:
            vocab.use_unk = True
            vocab.vocab.append(Vocabulary.UNK_TOK)

        vocab.vocab += sorted(words)
        vocab.size = len(vocab.vocab)

        # build index
        for (i,v) in enumerate(vocab.vocab):
            vocab.word_index[v] = i
            vocab.index_word[i] = v

        return vocab

    @staticmethod
    def build_from_sentences(sentences, use_go_tok=False, use_unk=False, frequency_cap=-1):
        """ Build vocabulary from a sentence list
            Args:
                sentences <list(str)>: the list of sentences, each sentence is a list of wors
                use_go_tok: whether <GO> tok will be included
                use_unk: whether <UNK> symbol will be used
                frequency_cap: words appear less frequent than frequency_cap 
                               will not be included in the vocabulary
            Returns:
                a Vocabulary object build from the sentences
        """
        used_words = {}

        for sentence in sentences:
            for s in sentence:
                if s in used_words:
                    used_words[s] += 1
                else:
                    used_words[s] = 1

        words = [s for s in used_words if used_words[s] > frequency_cap]
        
        return Vocabulary.build_from_words(words, use_go_tok, use_unk)

    def sequence_to_vec(self, sentence):
        return [self.word_to_index(s) for s in sentence]

    def vec_to_sequence(self, indvec):
        return [self.index_word[ind] for ind in indvec]

## the following two embeddings are based on the embedding library by Victor Zhong
## url: https://github.com/vzhong/embeddings

class GloVeEmbeddings(object):

    def __init__(self, emb_file, d_emb=-1, default="zero"):
        self.embeddings = {}
        self.d_emb = d_emb
        self.default = default

        full_emb_size = -1
        with open(emb_file, "r") as f:
            for line in f.readlines():
                l = line.split()
                # the word is in the beginning of the list
                word = l[0]
                if self.d_emb == -1:
                    self.d_emb = len(l) - 1
                if full_emb_size == -1:
                    full_emb_size = len(l) - 1
                # only get d_emb size vector from the embedding
                vec = np.array([float(x) for x in l[-self.d_emb:]], dtype=np.float32)
                self.embeddings[word] = vec
            if self.d_emb != full_emb_size:
                print("[Warning] Full embedding size is {}, larger than specified embedding size {}."
                        .format(full_emb_size, self.d_emb))

    def emb(self, word):
        """ embed a word using the GloVe embedding"""
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[self.default]
        if word in self.embeddings:
            g = self.embeddings[word]
        else:
            g = None
        return [get_default() for i in range(self.d_emb)] if g is None else g


def ngrams(sentence, n):
    """
    Returns:
        list: a list of lists of words corresponding to the ngrams in the sentence.
    """
    return [sentence[i:i + n] for i in range(len(sentence)-n+1)]


class JmtEmbeddings(object):

    def __init__(self, emb_file, d_emb=-1):
        self.embeddings = {}
        self.d_emb = d_emb
        full_emb_size = -1

        with open(emb_file, "r") as f:
            for line in f.readlines():
                l = line.split()
                ngram = l[0]
                if self.d_emb == -1:
                    self.d_emb = len(l) - 1
                if full_emb_size == -1:
                    full_emb_size = len(l) - 1
                vec = np.array([float(x) for x in l[-self.d_emb:]], dtype=np.float32)
                self.embeddings[ngram] = vec

            if self.d_emb != full_emb_size:
                print("[Warning] Full embedding size is {}, larger than specified embedding size {}."
                        .format(full_emb_size, self.d_emb))

    def emb(self, word):
        """ embedding a word using n-gram """
        chars = ['#BEGIN#'] + list(word) + ['#END#']
        embs = np.zeros(self.d_emb, dtype=np.float32)
        match = {}
        for i in [2, 3, 4]:
            grams = ngrams(chars, i)
            for g in grams:
                g = '{}gram-{}'.format(i, ''.join(g))
                if g in self.embeddings:
                    match[g] = np.array(self.embeddings[g], np.float32)
        if match:
            embs = sum(match.values()) / len(match)
        return embs.tolist()


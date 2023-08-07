#!/usr/bin/python
# -*- coding: utf-8 -*-
#from __future__ import print_function

from ctypes import *
import sys#, locale

encodings = {'il2': 'iso-8859-2', 'koi8': 'koi8-r'}
BUFFSIZE = 4096

class Lemmatizer:
    path = "/home/big_maggie/data/project/xmltran_python/liblemm.so"
    vocab = c_char_p("/home/pepa/diakritika/diakritika/liblemm/lgmf_cz.lex".encode('utf-8'))
    encoding = c_char_p()
    num = 3  # tohle ovlivnuje jak se chovat, kdyz je na vstupu znak, ktery neni v il2 - trojka je dobra
    lng = c_char_p()  # char value. e.g. "cz", int value returned by lemma_open_mdr in self.lang
    code = c_char_p()  # ???
    engine = c_int()
    version = c_int()

    def __init__(self, lng, vocab, encoding, path="liblemm.so"):
        self.vsw = c_void_p()
        self.path = path
        try:
            self.lib = cdll.LoadLibrary(path)
        except OSError:
            sys.stderr.write("Error opening liblemm.so. Try setting LD_LIBRARY_PATH.")
            sys.exit(1)
        self.lng = lng
        self.vocab = vocab
        self.encoding = encoding
        self.lib.lemma_set_encoding(self.encoding, self.num)
        if self.lib.lemma_init(byref(self.vsw), self.code) != 0:
            sys.stderr.write("Error initializing the library")
            sys.exit(2)

        self.lib.lemma_version(byref(self.engine), byref(self.version))
        self.lang = self.lib.lemma_open_mdr(bytes(self.lng, encoding='utf-8'), bytes(self.vocab, encoding='utf-8'))
        if self.lang < 0:
            sys.stderr.write("Error opening vocabulary: %s" % self.vocab)
            sys.exit(3)

    def add_vocab(self, lng, vocab):
        self.lang.append(self.lib.lemma_open_mdr(lng, vocab))
        self.vocab.append(vocab)
        self.lng.append(lng)
        if self.lang < 0:
            sys.stderr.write("Error opening vocabulary: %s" % vocab)
            sys.exit(3)

    def getVersion(self):
        return (self.engine.value, self.version.value)

    def lemmatize(self, word, mode=1):
        lemma = create_string_buffer(BUFFSIZE)
        buff = bytes(word.lstrip(), encoding='utf-8')
        bbuff = self.db_utfn2ascc(buff).value
        self.lib.lemma_get_stem.restype = c_int
        lemmCnt = self.lib.lemma_check_wrd(self.lang, bbuff, self.vsw)
        self.lib.lemma_get_stem.restype = POINTER(c_char)
        lemmas = []
        for i in range(lemmCnt):
            l = self.lib.lemma_get_stem(self.lang, i, mode, byref(lemma), BUFFSIZE, self.vsw)
            lemmas.append(self.db_asc2utf(lemma.value).value)  # .decode(encodings[self.encoding]))
        return lemmas

    def allw2(self, word ,mode=0x1FF0):
        buff = create_string_buffer(BUFFSIZE)
        word = word.encode(encodings[self.encoding])
        l = self.lib.lemma_get_allw(self.lang, 0, mode, byref(buff), BUFFSIZE, self.vsw)
        wrd = ""
        words = []
        for i, byte in enumerate(buff.raw):
            if byte == 0:  # zero byte means end of the word
                words.append(wrd)
                if buff.raw[i + 1] == 0: return words  # two consecutive zero bytes mean end of the output
                wrd = ""
            wrd += byte
        return words


    def allw(self,word,mode=0x1FF0):
        wrds=create_string_buffer(BUFFSIZE)
        buff=create_string_buffer(BUFFSIZE)
        try:
            word=word.encode('utf-8')
        except Exception as e:
            #pass
            print("Error (spis pro informaci) while encoding: %s" %e, file=sys.stderr)

        word=self.db_utfn2ascc(word)

        words=[]
        wrd=bytes()
        lemmCnt=self.lib.lemma_check_wrd(self.lang,word,self.vsw)
        sys.stderr.write("expand ")

        if lemmCnt == 0:
            return [word]
        for i in range(lemmCnt):
            l=self.lib.lemma_get_allw(self.lang,i,mode,byref(wrds),BUFFSIZE,self.vsw)
            for i, byte in enumerate(wrds.raw):


                if byte == 0:  # zero byte means end of the word
                    words.append(self.db_asc2utf(wrd).raw.decode())
                    #sys.stderr.write(wrd+'\n')
                    if wrds.raw[i + 1] == 0: break  # two consecutive zero bytes mean end of the output
                    wrd = bytes()
                else:
                    wrd += chr(byte).encode(encodings[self.encoding])
                    #print("new wrd: {}".format( wrd))
            words.append(self.db_asc2utf(wrd).raw.decode())
            wrd=bytes()#next lemma
        return words



    def db_utfn2ascc(self, word):
        utf = word  # .decode('utf-8')
        asc = create_string_buffer(BUFFSIZE)
        self.lib.db_utfn2ascc(byref(asc), sizeof(asc), utf, BUFFSIZE)
        return asc

    # *db_asc2utf(char *utf, int siz, const char *asc);
    def db_asc2utf(self, word):
        asc = word
        utf = create_string_buffer(BUFFSIZE)
        self.lib.db_asc2utf(byref(utf), BUFFSIZE, asc)
        return utf

    def mk_accent(self, word):
        buff = create_string_buffer(BUFFSIZE)
        word = word.encode(encodings[self.encoding])
        # tmp=c_char_p(word)
        self.lib.lemma_mk_accent(self.lang, word, byref(buff), sizeof(buff))
        wrd = ""
        words = []
        for i, byte in enumerate(buff.raw):
            if ord(byte) == 0:  # zero byte means end of the word
                words.append(wrd.decode(encodings[self.encoding]))
                if ord(buff.raw[i + 1]) == 0: return words  # two consecutive zero bytes mean end of the output
                wrd = ""
            wrd += byte

    def get_posp(self, word):
        buff = create_string_buffer(BUFFSIZE)
        try:
            word = word.encode(encodings[self.encoding])
        except:
            pass
        lemmCnt = self.lib.lemma_check_wrd(self.lang, word, self.vsw)
        pos = []
        for i in range(lemmCnt):
            p = self.lib.lemma_get_posp(self.lang, i, self.vsw)
            pos.append(p)
        return pos

    def get_lang(self, buff, threshold, hint):
        buff = bytes(buff.lstrip(), encoding='utf-8')
        bbuff = self.db_utfn2ascc(buff).value

        #print("2")
        #print(bbuff)

        bbuff = bbuff.replace(b' ', b'\0')
        threshold = c_double(threshold)
        self.lib.lemma_get_lang.argtypes = [c_char_p, c_double, c_char_p]
        self.lib.lemma_get_lang.restype = POINTER(c_char)
        lang = self.lib.lemma_get_lang(bbuff, threshold, bytes(hint, encoding='utf-8'))
        lang = cast(lang, c_char_p)
        if lang.value is None:
            return "None"
        return str(lang.value, encoding='utf-8')

    def __del__(self):
        self.lib.lemma_close_mdr(self.lang)
        self.lib.lemma_done(self.vsw)

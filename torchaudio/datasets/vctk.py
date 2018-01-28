from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import shutil
import errno
import torch
import torchaudio
import pickle
import re
import random
import numpy as np
from collections import OrderedDict, Counter

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def make_manifest(dir):
    audios = {}
    dir = os.path.expanduser(dir)
    list_dir = sorted(os.listdir(dir))
    if 'wav16' in list_dir:
        # will read 16kHz version of wavs if available
        target = 'wav16'
    else:
        target = 'wav48'
    #for target in sorted(os.listdir(dir)):
    d = os.path.join(dir, target)
    #if not os.path.isdir(d):
    #    continue
    for root, _, fnames in sorted(os.walk(d)):
        for fname in fnames:
            if is_audio_file(fname):
                spk_id = root.split('/')[-1]
                path = os.path.join(root, fname)
                item = path
                if spk_id not in audios:
                    audios[spk_id] = []
                audios[spk_id].append({'audio':item, 'spk_id':spk_id})
    return audios


def read_aco_file(aco_filename):
    import struct
    with open(aco_filename, 'rb') as aco_f:
        aco_bs = aco_f.read()
        aco_data = struct.unpack('{}f'.format(int(len(aco_bs) / 4)), aco_bs)
    return np.array(aco_data, dtype=np.float32)

def read_audio(fp, downsample=True):
    sig, sr = torchaudio.load(fp)
    if downsample and sr > 16000:
        # 48khz -> 16 khz
        import librosa
        sig = librosa.resample(sig.numpy().squeeze(), 48000, 16000)
        sig = torch.from_numpy(sig)
        #if sig.size(0) % 3 == 0:
        #    sig = sig[::3].contiguous()
        #else:
        #    sig = sig[:-(sig.size(0) % 3):3].contiguous()
        sr = 16000
    return sig, sr

def load_txts(dir):
    """Create a dictionary with all the text of the audio transcriptions."""
    utterences = dict()
    txts = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), "r") as f:
                        fname_no_ext = os.path.basename(fname).rsplit(".", 1)[0]
                        utterences[fname_no_ext] = f.readline()
                        #utterences += f.readlines()
    #utterences = dict([tuple(u.strip().split(" ", 1)) for u in utterences])
    return utterences


class VCTK(data.Dataset):
    """`VCTK <http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_ Dataset.
    `alternate url <http://datashare.is.ed.ac.uk/handle/10283/2651>`

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        dev_mode(bool, optional): if true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions.
        maxlen(int, optional): if specified, wavs are trimmed to maxlen
            (randomly chunked).
    """
    raw_folder = 'vctk/raw'
    processed_folder = 'vctk/processed'
    url = 'http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz'
    dset_path = 'VCTK-Corpus'
    train_size = 0.8
    valid_size = 0.1
    test_size = 0.1

    def __init__(self, root, downsample=True, transform=None,
                 target_transform=None, download=False, dev_mode=False,
                 split='train', maxlen=None, store_chunked=False,
                 max_chunks_file=None, labs_root=None):
        # max_chunks_file: maximum chunks to get from file if store chunked
        self.root = os.path.expanduser(root)
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.chunk_size = 1000
        self.num_samples = 0
        self.max_len = 0
        self.cached_pt = 0
        self.split = split
        self.maxlen = maxlen
        # if there are labs within this root, use them for word spotting
        self.labs_root = labs_root
        self.store_chunked = store_chunked
        self.max_chunks_file = max_chunks_file
        if store_chunked:
            self.processed_folder += '_chunked'
            if labs_root is None:
                raise ValueError('Cannot use chunked data without labs to '
                                 'align words and speech!')
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self._read_info(split)
        self.data, self.labels, self.spk_ids, \
        self.accents, self.genders = torch.load(os.path.join(self.root, 
                                                             self.processed_folder, 
                                                             split,
                                                             "vctk_{:04d}.pt".format(self.cached_pt)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (audio, target, spk_id) where target is a utterance
        """
        #print('cached_pt: ', self.cached_pt)
        #print('chunk_size: ', self.chunk_size)
        #print('index: ', index)
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            del self.data
            del self.labels
            del self.spk_ids
            #print('re-loading cached_pt: ', self.cached_pt)
            self.data, self.labels, \
            self.spk_ids, self.accents, \
            self.genders = torch.load(os.path.join(self.root, 
                                                   self.processed_folder, 
                                                   self.split,
                                                   "vctk_{:04d}.pt".format(self.cached_pt)))
        index = index % self.chunk_size
        #print('data len: ', len(self.data))
        #print('index: ', index)
        audio = self.data[index]
        target = self.labels[index]
        spk_id = self.spk_ids[index]
        gender = self.genders[index]
        accent = self.accents[index]

        if self.maxlen is not None and self.maxlen < audio.size(0):
            # trim with random chunk
            last_t = audio.size(0) - self.maxlen
            beg_i = random.choice(list(range(last_t)))
            audio = audio[beg_i:beg_i + self.maxlen]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if spk_id[0] == 'p':
            # remove prefix to match spk ids in speaker-info.txt
            spk_id = spk_id[1:]

        return audio, target, spk_id, accent, gender

    def __len__(self):
        return self.num_samples

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder,
                                           'train', "vctk_info.txt"))

    def _write_info(self, num_items, split):
        info_path = os.path.join(self.root, self.processed_folder, split,
                                 "vctk_info.txt")
        spk2idx_path = os.path.join(self.root, self.processed_folder, split,
                                    "spk2idx.pkl")
        accent2idx_path = os.path.join(self.root, self.processed_folder, split,
                                    "accent2idx.pkl")
        gender2idx_path = os.path.join(self.root, self.processed_folder, split,
                                    "gender2idx.pkl")
        word2idx_path = os.path.join(self.root, self.processed_folder, split,
                                     "word2idx.pkl")
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("max_len,{}\n".format(self.max_len))
            f.write("num_ids,{}\n".format(self.num_ids))
        with open(spk2idx_path, "wb") as f:
            pickle.dump(self.spk2idx, f)
        with open(accent2idx_path, "wb") as f:
            pickle.dump(self.accent2idx, f)
        with open(gender2idx_path, "wb") as f:
            pickle.dump(self.gender2idx, f)
        with open(word2idx_path, "wb") as f:
            pickle.dump(self.word2idx, f)

    def _read_info(self, split):
        info_path = os.path.join(self.root, self.processed_folder, split,
                                 "vctk_info.txt")
        spk2idx_path = os.path.join(self.root, self.processed_folder,
                                    split,
                                    "spk2idx.pkl")
        accent2idx_path = os.path.join(self.root, self.processed_folder,
                                       split,
                                       "accent2idx.pkl")
        gender2idx_path = os.path.join(self.root, self.processed_folder,
                                       split,
                                       "gender2idx.pkl")
        word2idx_path = os.path.join(self.root, self.processed_folder,
                                       split,
                                       "word2idx.pkl")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.max_len = int(f.readline().split(",")[1])
            self.num_ids = int(f.readline().split(",")[1])
        with open(spk2idx_path, "rb") as f:
            self.spk2idx = pickle.load(f)
        with open(accent2idx_path, "rb") as f:
            self.accent2idx = pickle.load(f)
        with open(gender2idx_path, "rb") as f:
            self.gender2idx = pickle.load(f)
        with open(word2idx_path, "rb") as f:
            self.word2idx = pickle.load(f)

    def data_download(self):
        """Download the VCTK data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        splits = ['train', 'valid', 'test']
        processed_abs_dirs = [os.path.join(self.root, self.processed_folder, \
                                           split) for split in splits]
        dset_abs_path = os.path.join(self.root, self.raw_folder, self.dset_path)
        if self._check_exists():
            return raw_abs_dir, dset_abs_path, processed_abs_dirs, splits

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        try:
            for processed_abs_dir in processed_abs_dirs:
                os.makedirs(processed_abs_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = self.url
        print('Downloading ' + url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.isfile(file_path):
            data = urllib.request.urlopen(url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
        if not os.path.exists(dset_abs_path):
            with tarfile.open(file_path) as zip_f:
                zip_f.extractall(raw_abs_dir)
        else:
            print("Using existing raw folder")
        if not self.dev_mode:
            os.unlink(file_path)
        return raw_abs_dir, dset_abs_path, processed_abs_dirs, splits

    def build_vocabs(self, dset_abs_path):
        # Build the statistics out of training set
        # get num of spk ids
        with open(os.path.join(dset_abs_path,
                               'speaker-info.txt'), 'r') as spk_inf_f:
            split_re = re.compile('\s+')
            # skip line 0 for it is the header
            fields = [split_re.split(l) for i, l in \
                      enumerate(spk_inf_f.readlines()) if i > 0]
            ids = [field[0] for field in fields]
            gender = [field[2] for field in fields]
            accent = [field[3] for field in fields]
            spk2data = {}
            # include speaker p280 for it is not in info file
            ids.append('280')
            gender.append('F')
            genders_unique = list(set(gender))
            accent.append('UNK')
            accents_unique = list(set(accent))
            self.num_ids = len(ids) 
            print('Number of speakers found: ', self.num_ids)
            print('Number of genders found: ', len(genders_unique))
            print('Number of accents found: ', len(accents_unique))
            self.spk2idx = dict((k, i) for i, k in enumerate(ids))
            self.gender2idx = dict((k, i) for i, k in enumerate(gender))
            self.accent2idx = dict((k, i) for i, k in enumerate(accent))
            for i, spk in enumerate(ids):
                spk2data[spk] = {'gender':gender[i],
                                 'accent':accent[i]}
            self.spk2data = spk2data

    def download(self):
        # first, deal with data download/unzip
        raw_abs_dir, dset_abs_path, processed_abs_dirs, \
        splits = self.data_download()


        audios = make_manifest(dset_abs_path)
        utterences = load_txts(dset_abs_path)
        print("Found {} speaker files and {} utterences".format(len(audios), 
                                                                len(utterences)
                                                               )
             )
        self.max_len = 0
        tr_idxes = []
        va_idxes = []
        te_idxes = []
        # prepare splits indexes
        #random.shuffle(list(range(len(audios))))
        for spk_id, manifest in audios.items():
            random.shuffle(manifest)
            #print('Train size: {}'.format(N_train))
            #print('Test size: {}'.format(N_test))
            #print('Valid size: {}'.format(N_valid))
            #print('spk {} manifest len: {}'.format(spk_id, len(manifest)))
            N_train = int(np.ceil(len(manifest) * self.train_size))
            #N_test = int(np.floor(len(manifest) * self.test_size))
            N_valid = int(np.floor(len(manifest) * self.valid_size))
            #print('spk {} N_train {}'.format(spk_id, N_train))
            #assert N_train + N_test + N_valid == len(manifest), '{}+{}+{}' \
            #' != {}'.format(N_train, N_test, N_valid, len(manifest))
            tr_idxes += manifest[:N_train]
            va_idxes += manifest[N_train:N_train + N_valid]
            te_idxes += manifest[N_train + N_valid:]
        split_idxes = [tr_idxes, va_idxes, te_idxes]
        for split, idxes, processed_abs_dir in zip(splits, split_idxes, 
                                                   processed_abs_dirs):
            # process and save as torch files
            print('Processing {} split with {} samples...'.format(split,
                                                                  len(idxes)))
            shutil.copyfile(
                os.path.join(dset_abs_path, "COPYING"),
                os.path.join(processed_abs_dir, "VCTK_COPYING")
            )
            if split == 'train':
                self.build_vocabs(dset_abs_path)
                spk2data = self.spk2data
            files_log = ''
            chunk_n = 0
            curr_chunk = 0
            total_files = 0
            tensors = []
            labels = []
            lengths = []
            spk_ids = []
            genders = []
            accents = []
            word2idx = OrderedDict()
            word2idx['UNK'] = 0
            # make splitter to chunk utterances into words
            prog = re.compile('\s+')
            # make counter to sort dictionary by high -> low freq words
            wordcount = Counter()
                
            for i, f in enumerate(idxes):
                if 'wav48' in f['audio']:
                    txt_dir = os.path.dirname(f['audio']).replace("wav48", "txt")
                else:
                    # wav16 here
                    txt_dir = os.path.dirname(f['audio']).replace("wav16", "txt")
                files_log += '{}'.format(f['audio'])
                if os.path.exists(txt_dir):
                    f_rel_no_ext = os.path.basename(f['audio']).rsplit(".", 1)[0]
                    #print('f_rel_no_ext: ', f_rel_no_ext)
                    sig, sr = read_audio(f['audio'],
                                         downsample=self.downsample)
                    if split == 'train':
                        # capture word vocabulary
                        words = prog.split(utterences[f_rel_no_ext])
                        for word in filter(None, words):
                            wordcount.update([word])

                    if self.store_chunked:
                        # save chunked version of audio in hops of 1s and 1s
                        # chunks
                        T = sig.size(0)
                        L = sr
                        self.max_len = sr
                        chunk_count = 0
                        # pre-make seq of chunks
                        chunk_begs = []
                        chunk_ends = []
                        for beg_i in range(0, T, int(sr)):
                            if T - beg_i < sr:
                                L = T - beg_i
                            if L < sr * 0.3:
                                # skip less than 300ms chunk
                                continue
                            chunk_begs.append(beg_i)
                            chunk_ends.append(beg_i + L)
                        if len(chunk_begs) == 0:
                            # skip this file, no chunks available
                            print('Skipping audio {} for only has {} '
                                  'samples'.format(f['audio'], T))
                            continue
                        if self.max_chunks_file is not None:
                            import scipy.stats as ss
                            # select randomly the chunks to keep
                            idxes = np.array(list(range(len(chunk_begs))))
                            # limit num of chunks depending on available chunks
                            Nc = min(self.max_chunks_file, len(chunk_begs))
                            npdf = ss.norm.pdf(idxes, loc = len(idxes) / 2, 
                                               scale = len(idxes) / 10)
                            npdf = npdf / npdf.sum()
                            ch_idxes = np.random.choice(idxes, Nc, p=npdf)
                        else:
                            # select all indexes
                            ch_idxes = np.array(list(range(len(chunk_begs))))

                        for ch_idx in ch_idxes:
                            beg_i = chunk_begs[ch_idx]
                            end_i = chunk_ends[ch_idx]
                            files_log += '\n[{}:{}]\n'.format(beg_i, end_i)
                            tensors.append(sig[beg_i:end_i])
                            lengths.append(L)
                            labels.append(utterences[f_rel_no_ext])
                            spkid = f['spk_id'][1:]
                            spk_ids.append(spkid)
                            genders.append(spk2data[spkid]['gender'])
                            accents.append(spk2data[spkid]['accent'])
                            chunk_n += 1
                            total_files += 1
                            if chunk_n  % self.chunk_size == 0:
                                # closed a chunk, reset chunk_n
                                # sort sigs/spkid/labels: longest -> shortest
                                tensors, labels,\
                                spk_ids, accents, \
                                genders = zip(*[(b, c, d, e, f) for (a, b, c,
                                                                     d, e, f) in \
                                               sorted(zip(lengths, tensors, labels,
                                                          spk_ids, accents,
                                                          genders),
                                                      key=lambda x: x[0], reverse=True)])
                                data = (tensors, labels, spk_ids, accents,
                                        genders)
                                torch.save(
                                    data,
                                    os.path.join(self.root,
                                                 self.processed_folder,
                                                 split,
                                                 'vctk_{:04d}.pt'.format(curr_chunk)
                                                )
                                )
                                curr_chunk += 1
                                chunk_n = 0
                                tensors = []
                                labels = []
                                lengths = []
                                spk_ids = []
                                accents = []
                                genders = []
                            chunk_count += 1
                            if self.max_chunks_file is not None and \
                               chunk_count >= self.max_chunks_file:
                                # break the loop of chunks in this file
                                break
                        files_log = files_log[:-1]
                    else:
                        tensors.append(sig)
                        lengths.append(sig.size(0))
                        labels.append(utterences[f_rel_no_ext])
                        spkid = f['spk_id'][1:]
                        spk_ids.append(spkid)
                        accents.append(spk2data[spkid]['accent'])
                        genders.append(spk2data[spkid]['gender'])
                        self.max_len = sig.size(0) if sig.size(0) > self.max_len else self.max_len
                        chunk_n += 1
                        total_files += 1
                        if chunk_n  % self.chunk_size == 0:
                            # closed a chunk, reset chunk_n
                            # sort sigs/spkid/labels: longest -> shortest
                            tensors, labels,\
                            spk_ids, accents, \
                            genders = zip(*[(b, c, d, e, f) for (a, b, c, d, e,
                                                                f) in \
                                           sorted(zip(lengths, tensors, labels,
                                                      spk_ids, accents,
                                                      genders),
                                                   key=lambda x: x[0], reverse=True)])
                            data = (tensors, labels, spk_ids, accents, genders)
                            torch.save(
                                data,
                                os.path.join(self.root,
                                             self.processed_folder,
                                             split,
                                             'vctk_{:04d}.pt'.format(curr_chunk)
                                            )
                            )
                            curr_chunk += 1
                            chunk_n = 0
                            tensors = []
                            labels = []
                            lengths = []
                            spk_ids = []
                            accents = []
                            genders = []
                files_log += '\n'
            if chunk_n > 0:
                # something still in buffer for last chunk
                tensors, labels,\
                spk_ids, accents, \
                genders = zip(*[(b, c, d, e, f) for (a, b, c, d, e, f) in \
                                sorted(zip(lengths, tensors, labels,
                                           spk_ids, accents, genders),
                                       key=lambda x: x[0], reverse=True)])
                data = (tensors, labels, spk_ids, accents, genders)
                torch.save(
                    data,
                    os.path.join(self.root,
                                 self.processed_folder,
                                 split,
                                 'vctk_{:04d}.pt'.format(curr_chunk)
                                )
                )
                curr_chunk += 1

            with open(os.path.join(self.root,
                                   self.processed_folder,
                                   split, '{}_wavs.guia'.format(split)),
                      'w') as guia_f:
                guia_f.write(files_log)
            # finish the vocabulary building
            for ii, ww in enumerate(wordcount.most_common(len(wordcount))):
                word = ww[0]
                word2idx[word] = ii + 1
            self.word2idx = word2idx
            print('Built word vocabulary of size: ', len(word2idx))
            self._write_info(total_files, split)
            if not self.dev_mode:
                shutil.rmtree(raw_abs_dir, ignore_errors=True)

            print('Done!')


class AcoVCTK(VCTK):
    """ Acoustic features VCTK (with Ahocoder) """
    def __init__(self, root, download=False, dev_mode=False,
                 transform=None, target_transform=None,
                 split='train', maxlen=None, hop_length=80):
        super().__init__(root, downsample=True, transform=transform,
                         target_transform=target_transform, download=download,
                         dev_mode=dev_mode, split=split)
        self.maxlen = maxlen
        self.hop_length = hop_length
        self.prepare_data()

    def prepare_data(self):
        # first, deal with data download/unzip
        raw_abs_dir, dset_abs_path, processed_abs_dirs, \
        splits = self.data_download()
        print('raw_abs_dir: ', raw_abs_dir)
        print('dset_abs_path: ', dset_abs_path)
        print('processed_abs_dirs: ', processed_abs_dirs)
        print('splits: ', splits)
        if not os.path.exists(os.path.join(dset_abs_path, 'aco')):
            raise ValueError('Pre-computed aco features not available!')
        # audios and acos will be loaded
        # first check which audios exsit
        audios = make_manifest(dset_abs_path)
        print("Found {} speakers".format(len(audios)))
        self.max_len = 0
        tr_idxes = []
        va_idxes = []
        te_idxes = []
        # prepare splits indexes
        for spk_id, manifest in audios.items():
            random.shuffle(manifest)
            N_train = int(np.ceil(len(manifest) * self.train_size))
            N_valid = int(np.floor(len(manifest) * self.valid_size))
            tr_idxes += manifest[:N_train]
            va_idxes += manifest[N_train:N_train + N_valid]
            te_idxes += manifest[N_train + N_valid:]
        split_idxes = [tr_idxes, va_idxes, te_idxes]
        self.build_vocabs(dset_abs_path)
        self.num_samples = len(tr_idxes) + len(va_idxes) + len(te_idxes)
        #print('te_idxes: ', te_idxes)
        if self.split == 'train':
            self.curr_split = tr_idxes
        elif self.split == 'valid':
            self.curr_split = va_idxes
        elif self.split == 'test':
            self.curr_split = te_idxes
        else:
            raise ValueError('Unrecognized split: ', self.split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (audio, aco)
        """
        sample = self.curr_split[index]
        audio = sample['audio']
        spk_id = sample['spk_id']
        print('selecting audio: ', audio)
        if 'wav48' in audio:
            aco_dir = os.path.dirname(audio).replace("wav48", "aco")
        else:
            # wav16 here
            aco_dir = os.path.dirname(audio).replace("wav16", "aco")
        parent_path = aco_dir
        basename = os.path.splitext(os.path.basename(audio))[0]
        aco_file = os.path.join(parent_path, 
                                basename)
        cc_file = aco_file + '.cc'
        lf0_file = aco_file + '.lf0'
        fv_file = aco_file + '.fv'
        print('Loading cc file: ', cc_file)
        if not os.path.exists(aco_file + '.cc'):
            raise FileNotFoundError('File {} not found!'.format(cc_file))
        
        # load the audio signal
        sig, sr = torchaudio.load(audio)
        print('Loaded signal with {} samples'.format(sig.size()))
        print('Sampling rate: ', sr)
		# read cc file
        cc_sig = read_aco_file(cc_file).reshape((-1, 40))
        fv_sig = read_aco_file(fv_file).reshape((-1, 1))
        lf0_sig = read_aco_file(lf0_file).reshape((-1, 1))
        print('cc sig shape: ', cc_sig.shape)
        print('fv sig shape: ', fv_sig.shape)
        print('lf0 sig shape: ', lf0_sig.shape)

        # pad wav to achieve same num of samples to have N aco winds
        tot_len = int(fv_sig.shape[0] * 80)
        diff = tot_len - sig.size(0)
        if diff > 0:
            sig = torch.cat((sig, torch.zeros(diff, 1)), dim=0)
        print('Post pad sig size: ', sig.size())
        print('Post pad cc size: ', cc_sig.shape[0])

        if self.maxlen is not None and self.maxlen < sig.size(0):
            # trim with random chunk
            last_t = sig.size(0) - self.maxlen
            beg_i = random.choice(list(range(last_t)))
            end_i = beg_i + self.maxlen
            sig = sig[beg_i:end_i]
            # select proper acoustic window
            print('Maxlen from {} to {} in time'.format(beg_i, end_i))
            # divide time index by stride to obtain window
            win_beg_i = beg_i // 80
            win_end_i = end_i // 80
            print('Maxlen from CEPS {} to {} tim time'.format(win_beg_i,
                                                              win_end_i))
            cc_sig = cc_sig[win_beg_i:win_end_i, :]
            fv_sig = fv_sig[win_beg_i:win_end_i, :]
            lf0_sig = lf0_sig[win_beg_i:win_end_i, :]

        target = torch.cat((torch.FloatTensor(cc_sig),
                            torch.FloatTensor(fv_sig),
                            torch.FloatTensor(lf0_sig)), dim=1)

        if self.transform is not None:
            sig = self.transform(sig)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if spk_id[0] == 'p':
            # remove prefix to match spk ids in speaker-info.txt
            spk_id = spk_id[1:]

        return sig, target, spk_id

    def __len__(self):
        return len(self.curr_split)

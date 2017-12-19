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

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def make_manifest(dir):
    audios = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_audio_file(fname):
                    spk_id = root.split('/')[-1]
                    path = os.path.join(root, fname)
                    item = path
                    audios.append({'audio':item, 'spk_id':spk_id})
    return audios

def read_audio(fp, downsample=True):
    sig, sr = torchaudio.load(fp)
    if downsample:
        # 48khz -> 16 khz
        if sig.size(0) % 3 == 0:
            sig = sig[::3].contiguous()
        else:
            sig = sig[:-(sig.size(0) % 3):3].contiguous()
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
                 split='train', maxlen=None):
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

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self._read_info(split)
        self.data, self.labels, self.spk_ids = torch.load(os.path.join(self.root, 
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
            #print('re-loading cached_pt: ', self.cached_pt)
            self.data, self.labels, \
            self.spk_ids = torch.load(os.path.join(self.root, 
                                                   self.processed_folder, 
                                                   self.split,
                                                   "vctk_{:04d}.pt".format(self.cached_pt)))
        index = index % self.chunk_size
        #print('data len: ', len(self.data))
        audio = self.data[index]
        target = self.labels[index]
        spk_id = self.spk_ids[index]

        if self.maxlen is not None:
            # trim with random chunk
            if self.maxlen < audio.size(0):
                last_t = audio.size(0) - self.maxlen
                beg_i = random.choice(list(range(last_t)))
                #print('Selecting chunk of len {} at {}'.format(self.maxlen,
                #                                               beg_i))
                audio = audio[beg_i:beg_i + self.maxlen]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if spk_id[0] == 'p':
            # remove prefix to match spk ids in speaker-info.txt
            spk_id = spk_id[1:]

        return audio, target, spk_id

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
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("max_len,{}\n".format(self.max_len))
            f.write("num_ids,{}\n".format(self.num_ids))
        with open(spk2idx_path, "wb") as f:
            pickle.dump(self.spk2idx, f)

    def _read_info(self, split):
        info_path = os.path.join(self.root, self.processed_folder, split,
                                 "vctk_info.txt")
        spk2idx_path = os.path.join(self.root, self.processed_folder,
                                    split,
                                    "spk2idx.pkl")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.max_len = int(f.readline().split(",")[1])
            self.num_ids = int(f.readline().split(",")[1])
        with open(spk2idx_path, "rb") as f:
            self.spk2idx = pickle.load(f)

    def download(self):
        """Download the VCTK data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        splits = ['train', 'valid', 'test']
        processed_abs_dirs = [os.path.join(self.root, self.processed_folder, \
                                           split) for split in splits]
        dset_abs_path = os.path.join(self.root, self.raw_folder, self.dset_path)

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

        audios = make_manifest(dset_abs_path)
        utterences = load_txts(dset_abs_path)
        print("Found {} audio files and {} utterences".format(len(audios), 
                                                              len(utterences)))
        self.max_len = 0
        # prepare splits indexes
        random.shuffle(list(range(len(audios))))
        N_train = int(np.ceil(len(audios) * self.train_size))
        N_test = int(np.floor(len(audios) * self.test_size))
        N_valid = int(np.ceil(len(audios) * self.valid_size))
        print('Train size: {}'.format(N_train))
        print('Test size: {}'.format(N_test))
        print('Valid size: {}'.format(N_valid))
        assert N_train + N_test + N_valid == len(audios)
        split_idxes = [audios[:N_train], 
                       audios[N_train:N_train + N_valid],
                       audios[-N_test:]]
        for split, idxes, processed_abs_dir in zip(splits, split_idxes, 
                                                   processed_abs_dirs):
            # process and save as torch files
            print('Processing {}...'.format(split))
            shutil.copyfile(
                os.path.join(dset_abs_path, "COPYING"),
                os.path.join(processed_abs_dir, "VCTK_COPYING")
            )
            if split == 'train':
                # Build the statistics out of training set
                # get num of spk ids
                with open(os.path.join(dset_abs_path,
                                       'speaker-info.txt'), 'r') as spk_inf_f:
                    split_re = re.compile('\s+')
                    # skip line 0 for it is the header
                    ids = [split_re.split(l)[0] for i, l in \
                           enumerate(spk_inf_f.readlines()) if i > 0]
                    # include speaker p280 for it is not in info file
                    ids.append('280')
                    self.num_ids = len(ids) 
                    print('Number of speakers found: ', self.num_ids)
                    self.spk2idx = dict((k, i) for i, k in enumerate(ids))
            for n in range(len(idxes) // self.chunk_size + 1):
                tensors = []
                labels = []
                lengths = []
                spk_ids = []
                st_idx = n * self.chunk_size
                end_idx = st_idx + self.chunk_size
                for i, f in enumerate(idxes[st_idx:end_idx]):
                    txt_dir = os.path.dirname(f['audio']).replace("wav48", "txt")
                    if os.path.exists(txt_dir):
                        f_rel_no_ext = os.path.basename(f['audio']).rsplit(".", 1)[0]
                        sig = read_audio(f['audio'], downsample=self.downsample)[0]
                        tensors.append(sig)
                        lengths.append(sig.size(0))
                        labels.append(utterences[f_rel_no_ext])
                        spk_ids.append(f['spk_id'])
                        self.max_len = sig.size(0) if sig.size(0) > self.max_len else self.max_len
                # sort sigs/labels: longest -> shortest
                tensors, labels, \
                spk_ids = zip(*[(b, c, d) for (a,b,c, d) in sorted(zip(lengths,
                                                                       tensors,
                                                                       labels,
                                                                       spk_ids), key=lambda x: x[0], reverse=True)])
                data = (tensors, labels, spk_ids)
                torch.save(
                    data,
                    os.path.join(
                        self.root,
                        self.processed_folder,
                        split,
                        "vctk_{:04d}.pt".format(n)
                    )
                )
            #self._write_info((n*self.chunk_size)+i+1, split)
            self._write_info(len(idxes), split)
            if not self.dev_mode:
                shutil.rmtree(raw_abs_dir, ignore_errors=True)

            print('Done!')

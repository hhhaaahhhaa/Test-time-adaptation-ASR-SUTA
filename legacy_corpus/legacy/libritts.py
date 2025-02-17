from torch.utils.data import Dataset

from dlhlp_lib.parsers.raw_parsers import LibriTTSRawParser, LibriTTSInstance


class LibriTTSDataset(Dataset):
    def __init__(self, split, bucket_size, path, ascending=False):
        self.src_parser = LibriTTSRawParser(path)
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        split = ['test-other']
        # List all wave files
        file_list = []
        text = []
        for s in split:
            for instance in getattr(self.src_parser, s.replace('-', '_'), []):
                file_list.append(instance.wav_path)
                text.append(instance.text)

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)

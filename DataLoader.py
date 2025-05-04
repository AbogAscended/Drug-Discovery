import random, torch, bisect
from multiprocessing import get_context
from torch.utils.data import Dataset, Sampler, DataLoader, get_worker_info


def worker_init_fn(worker_ids):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.file_handles = [
        open(path, 'rb') for path in dataset.filepaths
    ]


class Data:
    def __init__(self, filepaths, encoder, n_gram, batch_size, num_workers, num_epochs):
        self.filepaths = filepaths
        self.encoder = encoder
        self.n_gram = n_gram
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.ds = FileDataset(filepaths, self.encoder, n_gram=n_gram)
        self.full_sampler = FileBatchSampler(
            counts=self.ds.counts,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            sample_ratio=1.0
        )


    def get_loaders(self):
        all_batches = list(self.full_sampler)
        random.shuffle(all_batches)

        val_frac = 0.20
        n_val = int(len(all_batches) * val_frac)
        val_batches = all_batches[:n_val]
        train_batches = all_batches[n_val:]
        train_sampler = ListBatchSampler(train_batches)
        val_sampler = ListBatchSampler(val_batches)

        train_loader = DataLoader(
            self.ds,
            batch_sampler=train_sampler,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=get_context("spawn"),
            persistent_workers=True
        )

        val_loader = DataLoader(
            self.ds,
            batch_sampler=val_sampler,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=get_context("spawn"),
            persistent_workers=True
        )

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(0.05 * total_steps)

        return train_loader, val_loader, total_steps, warmup_steps


class ListBatchSampler(Sampler):
    def __init__(self, batch_list):
        self.batch_list = batch_list

    def __iter__(self):
        yield from self.batch_list

    def __len__(self):
        return len(self.batch_list)

class FileDataset(Dataset):
    def __init__(self, filepaths, encoder, n_gram):
        self.filepaths = filepaths
        self.encoder = encoder
        self.n_gram = n_gram

        self.counts = []
        self.offsets = []
        total = 0
        for path in filepaths:
            offs = []
            with open(path, 'rb') as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offs.append(pos)
            total += len(offs)
            self.counts.append(total)
            self.offsets.append(offs)

        self.file_handles = None

    def __len__(self):
        return self.counts[-1]

    def __getitem__(self, idx):
        if self.file_handles is None:
            raise RuntimeError("file_handles not initialized – did you forget worker_init_fn?")

        file_idx = bisect.bisect_right(self.counts, idx)
        prev = 0 if file_idx == 0 else self.counts[file_idx-1]
        line_idx = idx - prev

        fh = self.file_handles[file_idx]
        fh.seek(self.offsets[file_idx][line_idx])
        seq = fh.readline().decode('utf-8').strip()

        seq_enc = self.encoder.encode_sequence(seq)  # (L, D)
        L, D = seq_enc.shape
        n = self.n_gram

        windows = [seq_enc[i : i + n] for i in range(L - n)]
        targets = [seq_enc[i + n].view(1, D) for i in range(L - n)]

        return torch.stack(windows), torch.cat(targets, dim=0)



class FileBatchSampler(Sampler):
    def __init__(self, counts, batch_size, shuffle=True, drop_last=True, sample_ratio: float = 1.0):
        self.counts = counts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sample_ratio = sample_ratio

        self.batches = []
        prev = 0
        for cum in counts:
            idxs = list(range(prev, cum))
            if shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), batch_size):
                batch = idxs[i : i + batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)
            prev = cum

        if shuffle:
            random.shuffle(self.batches)
        if not (0 < sample_ratio <= 1):
            raise ValueError("sample_ratio must be in (0,1]")
        if sample_ratio < 1.0:
            keep_n = int(len(self.batches) * sample_ratio)
            self.batches = random.sample(self.batches, keep_n)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)



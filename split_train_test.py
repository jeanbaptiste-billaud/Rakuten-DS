from rakuten_vision.dataset import train_test_dataset_to_lmdb_with_partition
from torch.multiprocessing import cpu_count

if __name__ == '__main__':
    train_test_dataset_to_lmdb_with_partition(batch_size=cpu_count())
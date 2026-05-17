import argparse
import torch

from dataset import get_pcvr_data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--schema', required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    args = p.parse_args()

    train_loader, valid_loader, train_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=args.schema,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        buffer_batches=1,
        shuffle_train=False,
    )

    it = iter(train_loader)
    batch = next(it)
    ed = batch.get('engineered_dense_feats')
    if ed is None:
        print('engineered_dense_feats not found in batch')
        return
    if isinstance(ed, torch.Tensor):
        arr = ed.cpu().numpy()
        print('engineered_dense_feats.shape=', arr.shape)
        print('nan count=', np.isnan(arr).sum())
        print('inf count=', np.isinf(arr).sum())
        print('mean,min,max =', float(arr.mean()), float(arr.min()), float(arr.max()))
    else:
        print('engineered_dense_feats type:', type(ed))


if __name__ == '__main__':
    import numpy as np
    main()

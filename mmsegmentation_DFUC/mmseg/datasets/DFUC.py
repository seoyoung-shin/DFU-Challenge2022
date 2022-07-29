# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DFUCDataset(CustomDataset):
    """DFUC dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'target')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(DFUCDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)

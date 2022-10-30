
def get_dataloaders(cfg, train_loader_cfg):
    from mmdet.datasets import build_dataset, build_dataloader
    from mmcv.utils import get_git_hash
    from mmdet import __version__

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    # prepare data loaders
    dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    return data_loaders

@RUNNERS.register_module()
class CustomRunner(EpochBasedRunner):
    def run(self,
        cfg,
        # data_loaders: List[DataLoader],
        workflow: List[Tuple[str, int]],
        train_loader_cfg: dict,
        max_epochs: Optional[int] = None,
        **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        data_loaders = get_dataloaders(cfg, train_loader_cfg)
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                            get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                            self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                            self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            data_loaders = get_dataloaders(cfg, train_loader_cfg)

            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    if len(data_loaders[i]) > 0:
                        self.logger.info(f"Training with dataloader size {data_loaders[i].dataset.dataset.data_infos.__len__()}")
                        epoch_runner(data_loaders[i], **kwargs)
                    else:
                        self.logger.warning("Dataloader has length 0")

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

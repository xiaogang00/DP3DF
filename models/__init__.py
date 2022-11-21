import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration

    if model == 'video_base6_4':
        from .Video_base_model6_4 import VideoBaseModel as M
    elif model == 'video_base6_5':
        from .Video_base_model6_5 import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

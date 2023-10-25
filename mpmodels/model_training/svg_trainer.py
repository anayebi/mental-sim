from mpmodels.model_training.fitvid_trainer import FitVidTrainer


class SVGTrainer(FitVidTrainer):
    def __init__(self, config):
        super(SVGTrainer, self).__init__(config)
        self.grad_clip_max_norm = None
        self.action_conditioned = False

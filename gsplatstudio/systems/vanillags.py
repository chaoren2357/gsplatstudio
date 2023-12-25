import gsplatstudio

@gsplatstudio.register("vanilla-gsplat")
class vanillaGS:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        self.trainer.train()
    def load(self, logger, data, trainer):
        self.logger = logger
        self.data = data
        self.trainer = trainer
from tqdm import tqdm

class ProgressBar:
    def __init__(self, total_iters, first_iter = 0):
        self.total_iters = total_iters
        self.bar = tqdm(range(first_iter, total_iters), desc="Training progress")

    def update(self,iteration,**kwargs):
        if iteration % 10 == 0:
            ema_loss_for_log = kwargs.get("ema_loss_for_log")
            self.bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            self.bar.update(10)
        if iteration == self.total_iters:
            self.bar.close()

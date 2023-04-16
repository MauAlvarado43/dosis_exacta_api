
class BestWeightsTracker:
    
    def __init__(self, filepath):
        self._filepath = filepath
        self._best_weights = None
        self._best_mAP = 0

    def on_epoch_end(self, model, mAP):
        if mAP > self._best_mAP:
            self._best_mAP = mAP
            self._best_weights = model.get_weights()

    def restore_and_save_best_weights(self, model):
        if self._best_weights is not None:
            model.set_weights(self._best_weights)
            model.save_weights(filepath = self._filepath, overwrite = True, save_format = "h5")
            print("Saved best model weights (Mean Average Precision = %1.2f%%) to '%s'" % (self._best_mAP, self._filepath))
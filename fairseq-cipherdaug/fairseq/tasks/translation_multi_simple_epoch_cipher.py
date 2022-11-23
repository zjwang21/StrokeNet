import torch
import logging
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch_eval import TranslationMultiSimpleEpochEvalTask


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch_cipher")
class TranslationMultiSimpleEpochCipherTask(TranslationMultiSimpleEpochEvalTask):
    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochEvalTask.add_args(parser)
        parser.add_argument("--reg-alpha", default=0, type=int)

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.criterion_reg_alpha = getattr(args, "reg_alpha", 0)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Overriding from base class to support sending in *num_updates* to criterion

        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, num_updates=update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

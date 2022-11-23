import math
from dataclasses import dataclass, field

import torch
import logging
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
)

logger = logging.getLogger(__name__)


@dataclass
class LabelSmoothedCrossEntropyCriterionJSConfig(LabelSmoothedCrossEntropyCriterionConfig):
    js_alpha: int = field(
        default=1,
        metadata={"help": "alpha hyperparameter for JS loss for CipherDAug"},
    )
    js_warmup: int = field(
        default=1,
        metadata={"help": "WarmUp model with regular x-ent for this many updates before computing JS loss"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_js", dataclass=LabelSmoothedCrossEntropyCriterionJSConfig)
class LabelSmoothedCrossEntropyJSCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        js_alpha=0,
        js_warmup=1,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.js_alpha = js_alpha
        self.js_warmup = js_warmup
        logger.info("Alpha for JS Loss set to {} .".format(js_alpha))
        logger.info("JS Loss will start after {} updates.".format(js_warmup))

    def compute_kl_loss(self, model, net_output, prime_net_output, pad_mask=None, reduce=True):
        # mean ouptut probs for the 2 forward passes
        # mean_net_output = (net_output[0] + prime_net_output[0]) / 2
        # mean_probs = model.get_normalized_probs((mean_net_output,), log_probs=False)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        prime_lprobs = model.get_normalized_probs(prime_net_output, log_probs=True)

        probs = model.get_normalized_probs(net_output, log_probs=False)
        prime_probs = model.get_normalized_probs(prime_net_output, log_probs=False)

        # p, q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
        # p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)

        # og
        # p_loss = torch.nn.functional.kl_div(lprobs, mean_probs, reduction="none")
        # q_loss = torch.nn.functional.kl_div(prime_lprobs, mean_probs, reduction="none")

        p_loss = torch.nn.functional.kl_div(lprobs, prime_probs, reduction="none")
        q_loss = torch.nn.functional.kl_div(prime_lprobs, probs, reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def forward(self, model, sample, reduce=True, num_updates=None):

        if ("prime" not in sample) or (num_updates is not None and num_updates < self.js_warmup):
            return super().forward(model, sample, reduce=reduce)

        sample_input = sample["net_input"]
        prime_sample = sample["prime"]["net_input"]

        prime_sample_input = {
            "src_tokens": prime_sample["src_tokens"],
            "src_lengths": prime_sample["src_lengths"],
            "prev_output_tokens": sample_input["prev_output_tokens"],
        }

        # original outputs
        net_output = model(**sample_input)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # prime outputs
        prime_net_output = model(**prime_sample_input)
        prime_lprobs = model.get_normalized_probs(prime_net_output, log_probs=True)
        prime_lprobs = prime_lprobs.view(-1, prime_lprobs.size(-1))

        # # mean ouptut probs for the 2 forward passes
        # mean_net_output = (net_output[0] + prime_net_output[0]) / 2
        # mean_lprobs = model.get_normalized_probs(net_output, log_probs=False)

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        # target = torch.cat([target, target.clone()], dim=0)

        # x-ent loss for original input
        og_loss, og_nll_loss = label_smoothed_nll_loss(
            lprobs,
            target.view(-1, 1),
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # x-ent loss for prime input
        prime_loss, prime_nll_loss = label_smoothed_nll_loss(
            prime_lprobs,
            target.view(-1, 1),
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        js_loss = self.compute_kl_loss(model, net_output, prime_net_output, pad_mask=pad_mask)
        # js_loss = torch.zeros(1).to(og_loss.device)
        loss = og_loss + prime_loss + self.js_alpha * js_loss

        ntokens = sample["ntokens"]
        nsentences = sample["target"].size(0) * 2
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        sample_size = sample_size * 2

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(og_nll_loss.data) if reduce else og_nll_loss.data,
            "js_loss": utils.item(js_loss.data) if reduce else js_loss.data,
            "prime_nll_loss": utils.item(prime_nll_loss.data) if reduce else prime_nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        # don't log for valid
        if sample_size == 2 * ntokens:
            js_loss = utils.item(sum(log.get("js_loss", 0) for log in logging_outputs))
            metrics.log_scalar(
                "js_loss",
                js_loss / sample_size,
                sample_size,
                round=3,
            )

            prime_nll_loss = utils.item(sum(log.get("prime_nll_loss", 0) for log in logging_outputs))
            metrics.log_scalar(
                "prime_nll_loss",
                prime_nll_loss / ntokens / math.log(2),
                ntokens,
                round=3,
            )

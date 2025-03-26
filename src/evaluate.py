from lm_eval.models.huggingface import HFLM
import lm_eval

def evaluate_task(model, tokenizer, batch_size: int, tasks: list[str], limit=None):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        )["results"]
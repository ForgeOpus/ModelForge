"""Shared question/context parsing and constrained answer-span decoding."""
import json


def parse_question(text):
    try:
        data = json.loads(text)
        question, context = data["question"], data["context"]
        if not isinstance(question, str) or not question.strip() or not isinstance(context, str) or not context.strip():
            raise ValueError
        return question, context
    except (ValueError, TypeError, KeyError):
        raise ValueError('Send JSON with nonempty "question" and "context" strings.') from None


def decode_answer(result, encoding, context):
    offsets = encoding["offset_mapping"][0].tolist()
    sequence = encoding.sequence_ids(0)
    starts = result.start_logits[0].detach().cpu().tolist()
    ends = result.end_logits[0].detach().cpu().tolist()
    candidates = [(starts[i] + ends[j], i, j)
                  for i in range(len(sequence)) if sequence[i] == 1 and offsets[i][1] > offsets[i][0]
                  for j in range(i, min(i + 30, len(sequence)))
                  if sequence[j] == 1 and offsets[j][1] > offsets[j][0]]
    if not candidates:
        return "(No answer span)"
    _, start, end = max(candidates)
    return context[offsets[start][0]:offsets[end][1]]

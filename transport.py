"""Read OpenAI-compatible JSON or SSE without storing reasoning text."""
import json
import time


def bounded_chunks(response, deadline_seconds):
    """read1 returns after a socket read; readline may wait forever on a trickle."""
    deadline = time.monotonic() + deadline_seconds
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Streaming request exceeded total deadline.')
        # Bound the next blocking read as well as time between received chunks.
        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        if sock is not None:
            sock.settimeout(min(remaining, 180))
        chunk = response.read1(8192)
        if time.monotonic() > deadline:
            raise TimeoutError('Streaming request exceeded total deadline.')
        if not chunk:
            return
        total += len(chunk)
        if total > 2_000_000:
            raise ValueError('Provider response exceeds 2 MB limit.')
        yield chunk


def bounded_lines(response, deadline_seconds):
    pending = b''
    for chunk in bounded_chunks(response, deadline_seconds):
        pending += chunk
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            yield line
    if pending:
        yield pending


def read_completion(response, streaming=False, deadline_seconds=300):
    if not streaming:
        return json.load(response)
    # Some compatible providers ignore stream=True and return ordinary JSON.
    headers = getattr(response, 'headers', {})
    if 'application/json' in headers.get('Content-Type', '').lower():
        return json.loads(b''.join(bounded_chunks(response, deadline_seconds)))
    content, usage, finish_reason = [], {}, None
    byte_count, reasoning_characters, completed = 0, 0, False
    for raw in bounded_lines(response, deadline_seconds):
        byte_count += len(raw)
        if byte_count > 2_000_000:
            raise ValueError('Provider response exceeds 2 MB limit.')
        line = raw.decode('utf-8').strip()
        if not line or line.startswith((':', 'event:', 'id:', 'retry:')):
            continue
        if not line.startswith('data:'):
            raise ValueError('Expected SSE data events; provider returned an unsupported response.')
        data = line[5:].strip()
        if data == '[DONE]':
            completed = True
            break
        event = json.loads(data)
        if event.get('error'):
            raise RuntimeError('Provider returned an error inside the event stream.')
        if isinstance(event.get('usage'), dict):
            usage = event['usage']
        for choice in event.get('choices', []):
            if choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta', {})
            text = delta.get('content')
            if text:
                if not isinstance(text, str):
                    raise ValueError('Expected text output in streaming response.')
                content.append(text)
            reasoning = delta.get('reasoning_content')
            if isinstance(reasoning, str):
                reasoning_characters += len(reasoning)
            if choice.get('finish_reason') is not None:
                finish_reason = choice['finish_reason']
    if not completed and finish_reason is None:
        raise ConnectionError('Provider stream ended before a completion marker; partial answer discarded.')
    return {'choices': [{'message': {'content': ''.join(content)}, 'finish_reason': finish_reason}],
            'usage': usage, 'stream_diagnostics': {'reasoning_characters': reasoning_characters,
                                                  'completed': completed or finish_reason is not None}}

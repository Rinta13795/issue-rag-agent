"""Record actual provider usage; missing cache reports are not zero hits."""

def record_usage(session, response):
    usage = getattr(response, 'usage_metadata', None) or {}
    metadata = getattr(response, 'response_metadata', None) or {}
    raw = metadata.get('token_usage') or metadata.get('usage') or {}
    inputs = usage.get('input_tokens', raw.get('prompt_tokens'))
    outputs = usage.get('output_tokens', raw.get('completion_tokens'))
    if isinstance(inputs, int):
        session.prompt_tokens = (session.prompt_tokens or 0) + inputs
    if isinstance(outputs, int):
        session.completion_tokens = (session.completion_tokens or 0) + outputs
    cached = (usage.get('input_token_details') or {}).get('cache_read')
    if cached is None:
        cached = raw.get('prompt_cache_hit_tokens', (raw.get('prompt_tokens_details') or {}).get('cached_tokens'))
    if isinstance(cached, int) and isinstance(inputs, int) and 0 <= cached <= inputs:
        session.cached_input_tokens = (session.cached_input_tokens or 0) + cached
        session.cache_reported_input_tokens += inputs

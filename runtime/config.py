import json
from pathlib import Path
from runtime.types import RunBudget


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding='utf-8'))
    lock = json.loads((path.parent/'user_simulator.lock.json').read_text())
    if config['user'] != lock:
        raise ValueError('User simulator differs from M0 freeze')
    for actor in ('agent','user'):
        route = config[actor]
        if route['provider'] != 'openai' or not route['api_base'].startswith('https://'):
            raise ValueError('M1 requires the configured OpenAI-compatible HTTPS route')
    for key in ('trials','concurrency','request_timeout_seconds','max_output_tokens'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    variants = {'baseline':(False,False), 'payment_guard':(True,False),
                'fact_guard':(False,True), 'all_guards':(True,True)}
    if config['trials'] != 3 or config['variant'] not in variants:
        raise ValueError('M1 requires three trials and an authorized harness variant')
    harness=config.get('harness',{})
    if (bool(harness.get('payment_fact_check',False)),bool(harness.get('retail_fact_check',False))) != variants[config['variant']]:
        raise ValueError('Variant and response-check flags differ')
    if harness.get('max_text_retries',2) != 2:
        raise ValueError('User fixed the maximum response rewrites at two')
    if not 0 < float(config['cost_cap_cny']) <= 9.5:
        raise ValueError('Development batch equivalent cost cap must be <=9.5 CNY')
    if not 0 <= config.get('max_retries',0) <= 2:
        raise ValueError('max_retries must be 0..2')
    if config.get('max_retries',0) and config.get('retry_wait_seconds',0)<60:
        raise ValueError('Online retries require at least 60 seconds wait')
    if config.get('min_request_interval_seconds',0)<0 or config.get('unknown_request_reserve_cny',0)<0:
        raise ValueError('Pacing interval and scheduling reserve must be nonnegative')
    budget = RunBudget(**config['budget'])
    if any(value <= 0 for value in config['budget'].values()):
        raise ValueError('Run budgets must be positive')
    return config, budget

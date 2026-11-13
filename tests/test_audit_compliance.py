from scripts.audit_compliance import scan_trace


def test_authentication_requires_success_before_personal_access():
    call = lambda id, name: {'id': id, 'function': {'name': name, 'arguments': '{}'}}
    messages = [
        {'role': 'assistant', 'tool_calls': [call('a', 'find_user_id_by_email')]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'Error: user not found'},
        {'role': 'assistant', 'tool_calls': [call('b', 'get_order_details')]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': '{}'},
        {'role': 'assistant', 'tool_calls': [call('c', 'find_user_id_by_email')]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'user_1'},
        {'role': 'assistant', 'tool_calls': [call('d', 'get_user_details')]},
        {'role': 'tool', 'tool_call_id': 'd', 'content': '{}'}]
    report = scan_trace(messages, [m for m in messages if m['role']=='assistant'], {'user_1'})
    assert report['preauth_data_access'] == [{'message_index': 3, 'tool': 'get_order_details'}]
    assert report['auth_events'] == [{'message_index': 6, 'user_id': 'user_1'}]


def test_structure_uses_original_response_before_reference_truncation():
    calls = [{'id': str(i), 'function': {'name': 'get_order_details', 'arguments': '{}'}} for i in (1,2)]
    raw = {'role': 'assistant', 'content': 'Checking now', 'tool_calls': calls}
    saved = {'role': 'assistant', 'content': 'Checking now', 'tool_calls': calls[:1]}
    result = scan_trace([saved], [raw], set())
    assert result['text_and_tool'][0]['message_index'] == 1
    assert result['multiple_tool_calls'][0]['raw_tool_count'] == 2
    assert result['multiple_tool_calls'][0]['saved_tool_count'] == 1

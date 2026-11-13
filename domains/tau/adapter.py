"""Project-side adapter; hidden task and grading evidence never enter messages."""
from copy import deepcopy
from runtime.types import ToolSpec, ToolResult, ToolStatus
from tau_bench.types import Action

WRITE_IDS = {
    'cancel_pending_order': ['order_id'],
    'exchange_delivered_order_items': ['order_id', 'item_ids', 'new_item_ids', 'payment_method_id'],
    'return_delivered_order_items': ['order_id', 'item_ids', 'payment_method_id'],
    'modify_pending_order_address': ['order_id'],
    'modify_pending_order_items': ['order_id', 'item_ids', 'new_item_ids', 'payment_method_id'],
    'modify_pending_order_payment': ['order_id', 'payment_method_id'],
    'modify_user_address': ['user_id'],
}


class TauAdapter:
    mode = 'dialogue'

    def __init__(self, env, task_index):
        self.env, self.task_index = env, task_index
        self.done, self.reward = False, None
        self.evidence = None
        self.reward_info = None
        self._calculate = env.calculate_reward
        env.calculate_reward = self._score

    def list_tools(self):
        return [ToolSpec(f['name'], f['description'], f['parameters'],
                         f['name'] in WRITE_IDS, WRITE_IDS.get(f['name'], []))
                for entry in self.env.tools_info for f in [entry['function']]]

    def reset(self):
        self.done, self.reward, self.evidence = False, None, None
        return self.env.reset(task_index=self.task_index).observation

    def snapshot(self):
        initial = self.env.data_load_func()
        changes = {table: {key: deepcopy(row) for key, row in rows.items()
                          if row != initial[table].get(key)}
                   for table, rows in self.env.data.items()}
        self.evidence = {'data_hash': self.env.get_data_hash(), 'changes': changes,
                         'actions': [a.model_dump() for a in self.env.actions]}

    def _score(self):
        # Fixed tau commit resets data and replays gold here. Snapshot first.
        self.snapshot()
        result = self._calculate()
        self.reward_info = result.model_dump()
        return result

    def _step(self, action):
        result = self.env.step(action)
        self.done = result.done
        if result.done:
            self.reward = result.reward
        return result.observation

    def execute(self, call):
        observation = self._step(Action(name=call.name, kwargs=call.arguments))
        error = observation.startswith(('Error', 'Unknown action'))
        return ToolResult(ToolStatus.FATAL_ERROR if error else ToolStatus.OK, observation)

    def respond(self, text):
        return self._step(Action(name='respond', kwargs={'content': text}))

    def final_reward(self):
        return self.reward

    def close(self):
        if self.evidence is None:
            self.snapshot()
        self.env.calculate_reward = self._calculate

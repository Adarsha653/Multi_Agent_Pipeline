import time
import json
import os
from datetime import datetime

os.makedirs('/tmp/logs', exist_ok=True)

class PipelineLogger:
    def __init__(self, query: str):
        self.query = query
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.events = []
        self.timers = {}
        self.log_file = f'/tmp/logs/session_{self.session_id}.json'
        print(f'Logger initialized — session: {self.session_id}')

    def start_agent(self, agent_name: str):
        self.timers[agent_name] = time.time()
        self._log('agent_start', agent_name)

    def end_agent(self, agent_name: str, details: dict = {}):
        start_time = self.timers.get(agent_name)
        elapsed = round(time.time() - start_time, 2) if start_time else 0.0
        self._log('agent_end', agent_name, {**details, 'duration_seconds': elapsed})
        print(f'   [{agent_name}] completed in {elapsed}s')

    def log_error(self, agent_name: str, error: str):
        self._log('error', agent_name, {'error': error})
        print(f'   [{agent_name}] ERROR: {error}')

    def _log(self, event_type: str, agent_name: str, details: dict = {}):
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'agent': agent_name,
            'details': details
        }
        self.events.append(event)
        self._save()

    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'query': self.query,
                'events': self.events
            }, f, indent=2)

    def summary(self):
        print('\n' + '='*50)
        print('PIPELINE SUMMARY')
        print('='*50)
        for e in self.events:
            if e['event_type'] == 'agent_end':
                d = e['details'].get('duration_seconds', '?')
                print(f"  {e['agent']:<25} {d}s")
        print(f'  Log saved to: {self.log_file}')
        print('='*50)

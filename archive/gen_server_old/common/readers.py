import os
import sys

from gen_server.job_queue.pulsar import make_topic

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../protobuf')))

from gen_server.job_queue.interface import JobQueue
from gen_server.settings import settings

from protobuf import UserHistory, JobSnapshot


class JobReader:
    def __init__(self, queue: JobQueue):
        self.queue = queue

    def history(self, user_id: str, limit: int = 50):
        jobs = []
        topic = f"persistent://{settings.pulsar.tenant}/{settings.pulsar.user_history_namespace}/{user_id}"
        reader = self.queue.reader(topic, is_read_compacted=True)

        while reader.has_message_available():
            if len(jobs) >= limit:
                break

            message = reader.read_next()
            data = UserHistory.Job.FromString(message.data())
            jobs.append(data)

        return UserHistory(jobs=jobs)

    def stream_ids(self, job_ids: list[str]):
        for job_id in job_ids:
            topic = make_topic(settings.pulsar.job_snapshot_namespace, job_id)
            reader = self.queue.reader(topic)

            print(reader.has_message_available())

            if reader.has_message_available():
                message = reader.read_next()
                yield JobSnapshot.FromString(message.data())

import threading


# import pulsar

# client = pulsar.Client('pulsar://localhost:6650')
#
# producer = client.create_producer('my-topic')
# producer.send()


class BaseQueue:
    def __init__(self):
        self.items = []
        self.last_seek = 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if len(self.items) == 0:
            raise Exception("Queue is empty")
        return self.items.pop(0)

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def peek(self):
        if len(self.items) == 0:
            raise Exception("Queue is empty")
        return self.items[0]

    def seek(self):
        if len(self.items) == 0:
            raise Exception("Queue is empty")
        item = self.items[self.last_seek]
        self.last_seek += 1

        return item


class InMemoryQueue:
    queue = {}
    producers = {}

    def __init__(self):
        self.lock = threading.Lock()

    def create_producer(self, topic: str):
        producer = InMemoryQueueProducer(self, topic)
        if topic not in self.producers:
            self.producers[topic] = producer

        if topic not in self.queue:
            self.queue[topic] = BaseQueue()

        return producer

    def create_consumer(self, topic: str):
        return InMemoryQueueConsumer(self, topic)


class InMemoryQueueProducer:
    topic: str
    queue: InMemoryQueue

    def __init__(self, queue: InMemoryQueue, topic):
        self.queue = queue
        self.topic = topic

    def send(self, data: bytes):
        with self.queue.lock:
            self.queue.queue[self.topic].enqueue({'topic': self.topic, 'data': data})

    def close(self):
        pass


class InMemoryQueueConsumer:
    topic: str
    queue: InMemoryQueue

    def __init__(self, queue: InMemoryQueue, topic):
        self.queue = queue
        self.topic = topic

    def receive(self):
        with self.queue.lock:
            try:
                return self.queue.queue[self.topic].seek()
            except Exception as e:
                return None

    def close(self):
        pass

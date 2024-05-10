from cqueue.in_memory import InMemoryQueue

if __name__ == "__main__":
    queue = InMemoryQueue()
    producer = queue.create_producer("test-topic")

    producer.send(b"Hello, World! 1")
    producer.send(b"Hello, World! 2")
    producer.send(b"Hello, World! 3")
    producer.send(b"Hello, World! 4")

from cqueue.in_memory import InMemoryQueue

if __name__ == "__main__":

    queue = InMemoryQueue()
  
    consumer = queue.create_consumer("test-topic")
    while True:
        message = consumer.receive()
        if message is None:
            break
        print(f"Received message: {message}")

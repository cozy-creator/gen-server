from pulsar import Client, AuthenticationToken

from gen_server.settings import settings

_authentication = AuthenticationToken(settings.pulsar.token)
client = Client(settings.pulsar.service_url, authentication=_authentication, operation_timeout_seconds=30)


async def receive_messages(consumer):
    while True:
        try:
            msg = await consumer.receive()
            yield msg
        except Exception as err:
            if not consumer.is_connected():
                break
            else:
                print(f"Error receiving message: {err}")


def create_producer(topic: str):
    producer = client.create_producer(topic)
    return producer


def add_topic_message(topic: str, *, message: bytes, properties: dict, namespace: str):
    producer = create_producer(f"persistent://{settings.pulsar.tenant}/{namespace}/{topic}")
    producer.send(message, properties)
    producer.close()

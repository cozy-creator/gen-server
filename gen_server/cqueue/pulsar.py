import pulsar
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


def create_reader(topic: str):
    reader = client.create_reader(topic)
    return reader


def create_subscription(topic: str, message_listener):
    subscription = client.subscribe(topic, message_listener)
    return subscription


def add_topic_message(topic: str, *, message: bytes, properties: dict, namespace: str) -> pulsar.MessageId:
    producer = create_producer(f"persistent://{settings.pulsar.tenant}/{namespace}/{topic}")
    message_id = producer.send(content=message, properties=properties)
    producer.close()

    return message_id

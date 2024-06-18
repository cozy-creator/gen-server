from typing import Optional

import pulsar
import requests
from pulsar import Client, AuthenticationToken, MessageId

from gen_server.job_queue.interface import JobQueue
from gen_server.settings import settings

_authentication = AuthenticationToken(settings.pulsar.auth_token)
client = Client(settings.pulsar.service_url, authentication=_authentication, operation_timeout_seconds=30)


class Pulsar(JobQueue):
    def __init__(self, config):
        self.config = config
        authentication = AuthenticationToken(config.auth_token)
        self.client = Client(config.service_url, authentication=authentication, operation_timeout_seconds=30)

    def publish(self, topic: str, **kwargs):
        producer = self.client.create_producer(topic)

        message_id = producer.send(
            content=kwargs.get('message'),
            properties=kwargs.get('properties'),
            partition_key=kwargs.get("partition_key")
        )
        producer.close()

        return message_id

    def subscribe(self, topic: str, **kwargs):
        subscription_name = kwargs.get('subscription_name')
        if subscription_name is None:
            raise ValueError("Subscription name is required")

        consumer_type = kwargs.get('consumer_type')
        if consumer_type is None:
            raise ValueError("Consumer type is required")

        subscription = self.client.subscribe(
            topic,
            consumer_type=consumer_type,
            subscription_name=subscription_name,
            message_listener=kwargs.get('listener'),
            consumer_name=kwargs.get('consumer_name'),
            dead_letter_policy=kwargs.get('dead_letter_policy'),
            receiver_queue_size=kwargs.get('receiver_queue_size'),
        )

        return subscription

    def unsubscribe(self, topic: str, **kwargs):
        subscription_name = kwargs.get('subscription_name')
        if subscription_name is None:
            raise ValueError("Subscription name is required")

        return self.client.subscribe(topic, subscription_name=subscription_name)

    def reader(self, topic: str, is_read_compacted: Optional[bool], **kwargs):
        start_message_id = kwargs.get('start_message_id')
        if start_message_id is None:
            start_message_id = MessageId.earliest

        reader = self.client.create_reader(
            topic,
            start_message_id=start_message_id,
            is_read_compacted=is_read_compacted
        )
        return reader

    def set_retention(self, namespace, retention_size_in_mb, retention_time_in_minutes):
        if not isinstance(self.config.tenant, str) or not isinstance(namespace, str):
            raise ValueError('Tenant and namespace must be strings')
        if not isinstance(retention_size_in_mb, int) or not isinstance(retention_time_in_minutes, int):
            raise ValueError('Retention size and time must be integers')
        if retention_size_in_mb < 0 or retention_time_in_minutes < 0:
            raise ValueError('Retention size and time must be non-negative')
        if not isinstance(self.config.auth_token, str):
            raise ValueError('Auth token must be a string')

        endpoint_url = f'{settings.pulsar.rest_url}/admin/v2/namespaces/{self.config.tenant}/{namespace}/retention'
        body = {
            'retentionSizeInMB': retention_size_in_mb,
            'retentionTimeInMinutes': retention_time_in_minutes
        }

        headers = {
            'Authorization': f'Bearer {self.config.auth_token}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(endpoint_url, json=body, headers=headers)
        except requests.exceptions.RequestException as e:
            raise Exception(f'Network error: {e}')

        if response.status_code != 200:
            raise Exception(f'Error setting retention policy: {response.text}')


async def receive_messages(consumer):
    while True:
        try:
            msg: pulsar.Message = await consumer.receive()
            yield msg
        except Exception as err:
            if not consumer.is_connected():
                break
            else:
                print(f"Error receiving message: {err}")


def create_producer(topic: str):
    producer = client.create_producer(topic)
    return producer


def add_topic_message(topic: str, *, message: bytes, properties: dict, namespace: str) -> pulsar.MessageId:
    producer = create_producer(f"persistent://{settings.pulsar.tenant}/{namespace}/{topic}")
    message_id = producer.send(content=message, properties=properties)
    producer.close()

    return message_id


def make_topic(namespace: str, name: str, is_persistent: bool = True):
    return make_topic_with_tenant(settings.pulsar.tenant, namespace, name, is_persistent)


def make_topic_with_tenant(tenant: str, namespace: str, name: str, is_persistent: bool = True):
    if is_persistent:
        return f"persistent://{tenant}/{namespace}/{name}"
    return f"non-persistent://{tenant}/{namespace}/{name}"

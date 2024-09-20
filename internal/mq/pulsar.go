package mq

import (
	"context"
	"sync"

	"github.com/apache/pulsar-client-go/pulsar"
)

type PulsarMQ struct {
	client    pulsar.Client
	producers sync.Map
	consumers sync.Map
}

func NewPulsarMQ(url string) (*PulsarMQ, error) {
	client, err := newPulsarClient(url)
	if err != nil {
		return nil, err
	}

	return &PulsarMQ{
		client: client,
	}, nil
}

func (mq *PulsarMQ) Publish(ctx context.Context, topic string, message []byte) error {
	producer, err := mq.getProducer(topic)
	if err != nil {
		return err
	}

	producerMsg := &pulsar.ProducerMessage{Payload: message}
	_, err = (*producer).Send(ctx, producerMsg)
	return err
}

func (mq *PulsarMQ) Receive(ctx context.Context, topic string) ([]byte, error) {
	consumer, err := mq.getConsumer(topic)
	if err != nil {
		return nil, err
	}

	msg, err := (*consumer).Receive(ctx)
	if err != nil {
		return nil, err
	}

	return msg.Payload(), nil
}

func (mq *PulsarMQ) CloseTopic(topic string) error {
	if producer, ok := mq.producers.Load(topic); ok {
		(*producer.(*pulsar.Producer)).Close()
		mq.producers.Delete(topic)
	}

	if consumer, ok := mq.consumers.Load(topic); ok {
		(*consumer.(*pulsar.Consumer)).Close()
		mq.consumers.Delete(topic)
	}

	return nil
}

func (mq *PulsarMQ) Close() error {
	mq.client.Close()
	return nil
}

func (mq *PulsarMQ) getProducer(topic string) (*pulsar.Producer, error) {
	producer, err := newProducer(mq.client, topic)
	if err != nil {
		return nil, err
	}

	value, _ := mq.producers.LoadOrStore(topic, producer)
	return value.(*pulsar.Producer), nil
}

func (mq *PulsarMQ) getConsumer(topic string) (*pulsar.Consumer, error) {
	consumer, err := newConsumer(mq.client, topic)
	if err != nil {
		return nil, err
	}

	value, _ := mq.consumers.LoadOrStore(topic, consumer)
	return value.(*pulsar.Consumer), nil
}

func newPulsarClient(url string) (pulsar.Client, error) {
	options := pulsar.ClientOptions{
		URL: url,
	}

	client, err := pulsar.NewClient(options)
	defer client.Close()

	return client, err
}

func newProducer(client pulsar.Client, topic string) (*pulsar.Producer, error) {
	options := pulsar.ProducerOptions{
		Topic: topic,
	}

	producer, err := client.CreateProducer(options)
	defer producer.Close()

	return &producer, err
}

func newConsumer(client pulsar.Client, topic string) (*pulsar.Consumer, error) {
	options := pulsar.ConsumerOptions{
		Topic: topic,
	}

	consumer, err := client.Subscribe(options)
	defer consumer.Close()

	return &consumer, err
}

package mq

import (
	"context"
	"fmt"
	"log"
	"runtime/debug"
	"strings"
	"sync"

	"github.com/apache/pulsar-client-go/pulsar"
	"github.com/cozy-creator/gen-server/internal/config"
)

type PulsarMQ struct {
	client    pulsar.Client
	producers sync.Map
	consumers sync.Map
}

func NewPulsarMQ(config *config.PulsarConfig) (*PulsarMQ, error) {
	client, err := newPulsarClient(config)
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

func (mq *PulsarMQ) Receive(ctx context.Context, topic string) (interface{}, error) {
	fmt.Println("Receiving message from topic: ", topic)
	consumer, err := mq.getConsumer(topic)
	if err != nil {
		fmt.Println("Error getting consumer: ", err)
		return nil, err
	}

	return (*consumer).Receive(ctx)
}

func (mq *PulsarMQ) GetMessageData(message interface{}) ([]byte, error) {
	msg := message.(pulsar.Message)
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

func (mq *PulsarMQ) Ack(topic string, message interface{}) error {
	consumer, err := mq.getConsumer(topic)
	fmt.Println("Acking message from topic: ", topic)
	if err != nil {
		return err
	}

	if err := (*consumer).Ack(message.(pulsar.Message)); err != nil {
		fmt.Println("Error acking message: ", err)
		return err
	}

	return nil
}

func (mq *PulsarMQ) getProducer(topic string) (*pulsar.Producer, error) {
	value, ok := mq.producers.Load(topic)
	if ok {
		return value.(*pulsar.Producer), nil
	}

	producer, err := newProducer(mq.client, topic)
	if err != nil {
		return nil, err
	}

	mq.producers.Store(topic, producer)
	return producer, nil
}

func (mq *PulsarMQ) getConsumer(topic string) (*pulsar.Consumer, error) {
	value, ok := mq.consumers.Load(topic)
	if ok {
		return value.(*pulsar.Consumer), nil
	}

	consumer, err := newConsumer(mq.client, topic)
	if err != nil {
		return nil, err
	}

	mq.consumers.Store(topic, consumer)
	return consumer, nil
}

func newPulsarClient(config *config.PulsarConfig) (pulsar.Client, error) {
	options := pulsar.ClientOptions{
		URL: config.URL,
		// OperationTimeout: config.OperationTimeout,
		// ConnectionTimeout: time.Duration(config.ConnectionTimeout),
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
	return &producer, err
}

func newConsumer(client pulsar.Client, topic string) (*pulsar.Consumer, error) {
	options := pulsar.ConsumerOptions{
		Topic:            topic,
		Type:             pulsar.Exclusive,
		SubscriptionName: strings.ReplaceAll(topic, "/", "-"),
	}

	fmt.Printf("hdhd.%s\n", topic)
	consumer, err := client.Subscribe(options)
	if err != nil {
		log.Printf("Stack trace: %s\n", debug.Stack()) // Print the stack trace
		fmt.Println("Error creating xxxx: ", err)
	}

	return &consumer, err
}

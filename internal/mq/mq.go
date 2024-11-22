package mq

import (
	"context"
	"errors"

	"github.com/cozy-creator/gen-server/internal/config"
)

var (
	ErrTopicNotExists = errors.New("topic does not exist")
	ErrQueueFull      = errors.New("queue is full")
	ErrQueueClosed    = errors.New("queue closed")
	ErrTopicClosed    = errors.New("topic closed")
)

const (
	MQTypeInMemory = "inmemory"
	MQTypePulsar   = "pulsar"
)

type MQ interface {
	Publish(ctx context.Context, topic string, message []byte) error
	Receive(ctx context.Context, topic string) (interface{}, error)
	GetMessageData(message interface{}) ([]byte, error)
	Ack(topic string, message interface{}) error
	CloseTopic(topic string) error
	Close() error
}

func NewMQ(cfg *config.Config) (MQ, error) {
	if cfg != nil && cfg.Pulsar != nil {
		return NewPulsarMQ(cfg.Pulsar)
	} else {
		return NewInMemoryMQ(10)
	}
}

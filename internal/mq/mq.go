package mq

import (
	"context"
	"errors"
	"fmt"

	"github.com/cozy-creator/gen-server/internal/config"
)

var (
	ErrTopicNotExists = errors.New("topic does not exist")
	ErrQueueFull      = errors.New("queue is full")
	ErrQueueClosed    = errors.New("queue closed")
	ErrTopicClosed    = errors.New("topic closed")
	ErrNoMessage      = errors.New("no message")
)

const (
	MQTypeInMemory = "inmemory"
	MQTypePulsar   = "pulsar"
)

type MQ interface {
	Publish(ctx context.Context, topic string, message []byte) error
	Receive(ctx context.Context, topic string) ([]byte, error)
	CloseTopic(topic string) error
	Close() error
}

func NewMQ(cfg *config.Config) (MQ, error) {
	switch cfg.MQType {
	case MQTypeInMemory:
		return NewInMemoryMQ(10)
	case MQTypePulsar:
		return NewPulsarMQ(cfg.Pulsar.URL)
	default:
		return nil, fmt.Errorf("unknown MQ type: %s", cfg.MQType)
	}
}

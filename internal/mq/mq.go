package mq

import (
	"context"
	"errors"
)

var (
	ErrTopicNotExists = errors.New("topic does not exist")
	ErrQueueFull      = errors.New("queue is full")
	ErrQueueClosed    = errors.New("queue closed")
	ErrTopicClosed    = errors.New("topic closed")
	ErrNoMessage      = errors.New("no message")
	ErrTimeout        = errors.New("timeout")
)

type MQueue interface {
	Publish(ctx context.Context, topic string, message []byte) error
	Receive(ctx context.Context, topic string) ([]byte, error)
	Ack(ctx context.Context, topic string, messageID *string) error
	CloseTopic(topic string) error
	Close() error
}

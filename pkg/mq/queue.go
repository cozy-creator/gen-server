package mq

import (
	"context"
)

type MessageQueue interface {
	Publish(ctx context.Context, topic string, message []byte) error
	Receive(ctx context.Context, topic string) ([]byte, error)
	Ack(ctx context.Context, topic string, messageID *string) error
	Close() error
}

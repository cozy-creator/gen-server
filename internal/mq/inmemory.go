package mq

import (
	"context"
	"sync"
)

type InMemoryMQ struct {
	maxSize int
	topics  sync.Map
	closeCh chan struct{}
}

func NewInMemoryMQ(maxSize int) (*InMemoryMQ, error) {
	return &InMemoryMQ{
		maxSize: maxSize,
		closeCh: make(chan struct{}),
	}, nil
}

func (q *InMemoryMQ) Publish(ctx context.Context, topic string, message []byte) error {
	value, _ := q.topics.LoadOrStore(topic, make(chan []byte, q.maxSize))
	ch := value.(chan []byte)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-q.closeCh:
		return ErrQueueClosed
	case ch <- message:
		return nil
	default:
		return ErrQueueFull
	}
}

func (q *InMemoryMQ) Receive(ctx context.Context, topic string) ([]byte, error) {
	value, _ := q.topics.LoadOrStore(topic, make(chan []byte, q.maxSize))
	ch := value.(chan []byte)

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-q.closeCh:
			return nil, ErrQueueClosed
		case data, ok := <-ch:
			if !ok {
				q.topics.Delete(topic)
				return nil, ErrTopicClosed
			}
			return data, nil
		}
	}
}

func (q *InMemoryMQ) CloseTopic(topic string) error {
	value, ok := q.topics.Load(topic)
	ch := value.(chan []byte)
	if !ok {
		return ErrTopicNotExists
	}

	close(ch)
	return nil
}

func (q *InMemoryMQ) Close() error {
	close(q.closeCh)
	return nil
}

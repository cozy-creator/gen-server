package equeue

import (
	"context"
	"sync"
	"time"
)

type InMemoryQueue struct {
	topics  sync.Map
	closeCh chan struct{}
	maxSize int
}

var inMemoryQueue *InMemoryQueue

func GetDefaultInMemoryQueue() *InMemoryQueue {
	if inMemoryQueue == nil {
		inMemoryQueue = NewInMemoryQueue(1)
	}
	return inMemoryQueue
}

func NewInMemoryQueue(maxSize int) *InMemoryQueue {
	return &InMemoryQueue{
		closeCh: make(chan struct{}),
		maxSize: maxSize,
	}
}

func (q *InMemoryQueue) Publish(ctx context.Context, topic string, message []byte) error {
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

func (q *InMemoryQueue) Receive(ctx context.Context, topic string) ([]byte, error) {
	value, _ := q.topics.LoadOrStore(topic, make(chan []byte, q.maxSize))
	ch := value.(chan []byte)
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-q.closeCh:
		return nil, ErrQueueClosed
	case data, ok := <-ch:
		if !ok {
			return nil, ErrTopicClosed
		}
		return data, nil
	default:
		return nil, ErrNoMessage
	}
}

func (q *InMemoryQueue) CloseTopic(topic string) error {
	value, ok := q.topics.Load(topic)
	ch := value.(chan []byte)
	if !ok {
		return ErrTopicNotExists
	}

	close(ch)
	q.topics.Delete(topic)
	return nil
}

func (q *InMemoryQueue) Ack(ctx context.Context, topic string, messageID *string) error {
	_, ok := q.topics.Load(topic)
	if !ok {
		return ErrTopicNotExists
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-q.closeCh:
		return ErrQueueClosed
	case <-time.After(time.Second * 5):
		return ErrTimeout
	default:
		return nil
	}
}
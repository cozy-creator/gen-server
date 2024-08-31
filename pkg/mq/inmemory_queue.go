package mq

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type InMemoryQueue struct {
	queues  map[string]chan []byte
	closeCh chan struct{}
	mu      sync.Mutex
	maxSize int
	size    int
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
		queues:  make(map[string]chan []byte),
		closeCh: make(chan struct{}),
		maxSize: maxSize,
	}
}

func (q *InMemoryQueue) Publish(ctx context.Context, topic string, message []byte) error {
	q.mu.Lock()
	if _, ok := q.queues[topic]; !ok {
		q.queues[topic] = make(chan []byte, q.maxSize)
	}
	q.mu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-q.closeCh:
		return fmt.Errorf("queue closed")
	default:
		q.mu.Lock()
		if q.size != 0 && q.size >= q.maxSize {
			return fmt.Errorf("cannot publish message, queue is full")
		}

		q.size++
		q.mu.Unlock()
		q.queues[topic] <- message
		return nil
	}
}

func (q *InMemoryQueue) Receive(ctx context.Context, topic string) ([]byte, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-q.closeCh:
		return nil, fmt.Errorf("queue closed")
	case data := <-q.queues[topic]:
		return data, nil
	default:
		return nil, fmt.Errorf("no message available")
	}
}

func (q *InMemoryQueue) Close() error {
	q.mu.Lock()
	defer q.mu.Unlock()

	close(q.closeCh)
	for _, queue := range q.queues {
		close(queue)
	}
	return nil
}

func (q *InMemoryQueue) Ack(ctx context.Context, topic string, messageID *string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-q.closeCh:
		return fmt.Errorf("queue closed")
	case <-time.After(time.Second * 5):
		return fmt.Errorf("timeout")
	default:
		q.mu.Lock()
		defer q.mu.Unlock()

		if _, ok := q.queues[topic]; !ok {
			return fmt.Errorf("topic not found")
		}

		q.size -= 1

		return nil
	}
}

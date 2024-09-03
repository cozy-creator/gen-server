package mq

import (
	"context"
	"fmt"
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
		return fmt.Errorf("queue closed")
	case ch <- message:
		return nil
	default:
		return fmt.Errorf("cannot publish message, queue is full")
	}
}

func (q *InMemoryQueue) Receive(ctx context.Context, topic string) ([]byte, error) {
	value, _ := q.topics.LoadOrStore(topic, make(chan []byte, q.maxSize))
	ch := value.(chan []byte)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-q.closeCh:
		return nil, fmt.Errorf("queue closed")
	case data, ok := <-ch:
		if !ok {
			return nil, fmt.Errorf("topic closed")
		}
		return data, nil
	default:
		return nil, nil
	}
}

func (q *InMemoryQueue) CloseTopic(topic string) error {
	value, ok := q.topics.Load(topic)
	if !ok {
		return fmt.Errorf("topic not found")
	}

	ch := value.(chan []byte)

	// Ensure no new messages can be sent to this channel
	q.topics.Delete(topic)

	// Close the channel to signal to receivers that no more messages will be sent
	close(ch)
	fmt.Println("Closing topic", topic)
	return nil
}

func (q *InMemoryQueue) Ack(ctx context.Context, topic string, messageID *string) error {
	_, ok := q.topics.Load(topic)
	if !ok {
		return fmt.Errorf("topic not found")
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-q.closeCh:
		return fmt.Errorf("queue closed")
	case <-time.After(time.Second * 5):
		return fmt.Errorf("timeout")
	default:
		return nil
	}
}

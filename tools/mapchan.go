package tools

import "errors"

type MapChan[T any] struct {
	data map[string]chan T
}

var bytesChan *MapChan[[]byte]

func DefaultBytesMap() *MapChan[[]byte] {
	if bytesChan == nil {
		bytesChan = &MapChan[[]byte]{data: make(map[string]chan []byte)}
	}

	return bytesChan
}

func (c *MapChan[T]) Get(key string) chan T {
	if _, ok := c.data[key]; !ok {
		panic(errors.New("key not found"))
	}

	return c.data[key]
}

func (c *MapChan[T]) Set(key string, value chan T) {
	c.data[key] = value
}

func (c *MapChan[T]) Delete(key string) {
	if _, ok := c.data[key]; !ok {
		panic(errors.New("key not found"))
	}

	close(c.data[key])
	delete(c.data, key)
}

func (c *MapChan[T]) Has(key string) bool {
	_, ok := c.data[key]
	return ok
}

func (c *MapChan[T]) Clear() {
	for key := range c.data {
		close(c.data[key])
	}

	c.data = make(map[string]chan T)
}

func (c *MapChan[T]) Len() int {
	return len(c.data)
}

func (c *MapChan[T]) Send(key string, value T) {
	if c.data[key] == nil {
		c.data[key] = make(chan T)
	}

	c.data[key] <- value
}

func (c *MapChan[T]) Receive(key string) (*T, bool) {
	if c.data[key] == nil {
		return nil, false
	}

	value, ok := <-c.data[key]
	return &value, ok
}

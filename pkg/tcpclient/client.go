package tcpclient

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"go.uber.org/zap"
)

var (
	ErrConnectionClosed = errors.New("connection is closed")
	ErrTimeout          = errors.New("operation timed out")
)

type TCPClient struct {
	address     string
	timeout     time.Duration
	maxRetries  int
	connections chan net.Conn
	tlsConfig   *tls.Config
	logger      *zap.Logger
	mu          sync.Mutex
}

type TCPClientOption func(*TCPClient)

func WithTLS(config *tls.Config) TCPClientOption {
	return func(c *TCPClient) {
		c.tlsConfig = config
	}
}

func WithLogger(logger *zap.Logger) TCPClientOption {
	return func(c *TCPClient) {
		c.logger = logger
	}
}

func NewTCPClient(address string, timeout time.Duration, poolSize int, opts ...TCPClientOption) (*TCPClient, error) {
	client := &TCPClient{
		address:     address,
		timeout:     timeout,
		maxRetries:  3,
		connections: make(chan net.Conn, poolSize),
		logger:      zap.NewNop(),
	}

	for _, opt := range opts {
		opt(client)
	}

	for i := 0; i < poolSize; i++ {
		conn, err := client.dial()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize connection pool: %w", err)
		}
		client.connections <- conn
	}

	return client, nil
}

func (c *TCPClient) dial() (net.Conn, error) {
	dialer := &net.Dialer{Timeout: c.timeout}
	if c.tlsConfig != nil {
		return tls.DialWithDialer(dialer, "tcp", c.address, c.tlsConfig)
	}
	return dialer.Dial("tcp", c.address)
}

func (c *TCPClient) getConnection() (net.Conn, error) {
	select {
	case conn := <-c.connections:
		return conn, nil
	case <-time.After(c.timeout):
		return nil, ErrTimeout
	}
}

func (c *TCPClient) releaseConnection(conn net.Conn) {
	c.connections <- conn
}

func (c *TCPClient) Send(ctx context.Context, data string) error {
	var err error
	for i := 0; i < c.maxRetries; i++ {
		if err = c.send(ctx, data); err == nil {
			return nil
		}
		c.logger.Warn("Failed to send data, retrying", zap.Error(err), zap.Int("attempt", i+1))
	}
	return fmt.Errorf("failed to send data after %d attempts: %w", c.maxRetries, err)
}

func (c *TCPClient) send(ctx context.Context, data string) error {
	conn, err := c.getConnection()
	if err != nil {
		return err
	}
	defer c.releaseConnection(conn)

	writer := bufio.NewWriter(conn)
	_, err = writer.WriteString(data + "\n")
	if err != nil {
		return fmt.Errorf("failed to send data: %w", err)
	}

	err = writer.Flush()
	if err != nil {
		return fmt.Errorf("failed to flush data: %w", err)
	}

	return nil
}

func (c *TCPClient) Receive(ctx context.Context) (string, error) {
	var err error
	var response string
	for i := 0; i < c.maxRetries; i++ {
		if response, err = c.receive(ctx); err == nil {
			return response, nil
		}
		c.logger.Warn("Failed to receive data, retrying", zap.Error(err), zap.Int("attempt", i+1))
	}
	return "", fmt.Errorf("failed to receive data after %d attempts: %w", c.maxRetries, err)
}

func (c *TCPClient) receive(ctx context.Context) (string, error) {
	conn, err := c.getConnection()
	if err != nil {
		return "", err
	}
	defer c.releaseConnection(conn)

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
		reader := bufio.NewReader(conn)
		response, err := reader.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("failed to receive data: %w", err)
		}

		return response, nil
	}
}

func (c *TCPClient) ReceiveFullBytes(ctx context.Context, n int) ([]byte, error) {
	var err error
	var response []byte
	for i := 0; i < c.maxRetries; i++ {
		if response, err = c.receiveFullBytes(ctx, n); err == nil {
			return response, nil
		}
		c.logger.Warn("Failed to receive full bytes, retrying", zap.Error(err), zap.Int("attempt", i+1))
	}
	return nil, fmt.Errorf("failed to receive full bytes after %d attempts: %w", c.maxRetries, err)
}

func (c *TCPClient) receiveFullBytes(ctx context.Context, n int) ([]byte, error) {
	conn, err := c.getConnection()
	if err != nil {
		return nil, err
	}
	defer c.releaseConnection(conn)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		bytes := make([]byte, n)
		_, err = io.ReadFull(conn, bytes)
		if err != nil {
			return nil, fmt.Errorf("failed to receive data: %w", err)
		}

		return bytes, nil
	}
}

func (c *TCPClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	close(c.connections)
	for conn := range c.connections {
		if err := conn.Close(); err != nil {
			c.logger.Error("Failed to close connection", zap.Error(err))
		}
	}

	return nil
}

func (c *TCPClient) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	err := c.Send(ctx, "PING")
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	response, err := c.Receive(ctx)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	if response != "PONG" {
		return fmt.Errorf("unexpected health check response: %s", response)
	}

	return nil
}

package services

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"time"
)

type TCPClient struct {
	conn    net.Conn
	reader  *bufio.Reader
	writer  *bufio.Writer
	timeout time.Duration
	closed  bool
}

func NewTCPClient(address string, timeout time.Duration) (*TCPClient, error) {
	conn, err := net.DialTimeout("tcp", address, timeout)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to server: %w", err)
	}

	client := &TCPClient{
		conn:    conn,
		reader:  bufio.NewReader(conn),
		writer:  bufio.NewWriter(conn),
		closed:  false,
		timeout: timeout,
	}

	return client, nil
}

// Send sends data to the server.
func (c *TCPClient) Send(data string) error {
	if c.closed {
		return errors.New("connection is closed")
	}

	_, err := c.writer.WriteString(data + "\n")
	if err != nil {
		return fmt.Errorf("failed to send data: %w", err)
	}

	err = c.writer.Flush()
	if err != nil {
		return fmt.Errorf("failed to flush data: %w", err)
	}

	return nil
}

func (c *TCPClient) Receive() (string, error) {
	if c.closed {
		return "", errors.New("connection is closed")
	}

	c.conn.SetReadDeadline(time.Now().Add(c.timeout))

	response, err := c.reader.ReadString('\n')
	if err != nil {
		return "", fmt.Errorf("failed to receive data: %w", err)
	}

	response = strings.TrimSpace(response)

	return response, nil
}

func (c *TCPClient) ReceiveFullBytes(n int) ([]byte, error) {
	if c.closed {
		return nil, errors.New("connection is closed")
	}

	bytes := make([]byte, n)
	response, err := io.ReadFull(c.reader, bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to receive data: %w", err)
	}

	return bytes[:response], nil
}

func (c *TCPClient) Close() error {
	if c.closed {
		return nil
	}

	err := c.conn.Close()
	if err != nil {
		return fmt.Errorf("failed to close connection: %w", err)
	}

	c.closed = true
	return nil
}

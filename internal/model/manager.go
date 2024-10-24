package model

import (
    "context"
    "encoding/binary"
    "encoding/json"
    "fmt"

    "github.com/cozy-creator/gen-server/internal/mq"
    "github.com/cozy-creator/gen-server/pkg/tcpclient"
)

type ModelManager struct {
    tcpClient *tcpclient.TCPClient
    mq        mq.MQ
}

func NewModelManager(tcpClient *tcpclient.TCPClient, mq mq.MQ) *ModelManager {
    return &ModelManager{
        tcpClient: tcpClient,
        mq:        mq,
    }
}

func (m *ModelManager) TCPClient() *tcpclient.TCPClient {
    return m.tcpClient
}

func (m *ModelManager) LoadModel(ctx context.Context, modelID string) error {
    req := ModelRequest{
        Command: LoadModel,
        ModelID: modelID,
    }
    
    data, err := json.Marshal(req)
    if err != nil {
        return fmt.Errorf("failed to marshal request: %w", err)
    }

    // Send model command header to distinguish from generation requests
    header := []byte("MODEL:")
    if err := m.tcpClient.Send(ctx, string(append(header, data...))); err != nil {
        return fmt.Errorf("failed to send request: %w", err)
    }

    // Wait for response
    resp, err := m.tcpClient.ReceiveFullBytes(ctx, 4) // size header
    if err != nil {
        return fmt.Errorf("failed to receive response size: %w", err)
    }

    size := binary.BigEndian.Uint32(resp)
    data, err = m.tcpClient.ReceiveFullBytes(ctx, int(size))
    if err != nil {
        return fmt.Errorf("failed to receive response: %w", err)
    }

    var modelResp ModelResponse
    if err := json.Unmarshal(data, &modelResp); err != nil {
        return fmt.Errorf("failed to unmarshal response: %w", err)
    }

    if !modelResp.Success {
        return fmt.Errorf("failed to load model: %s", modelResp.Error)
    }

    return nil
}

func (m *ModelManager) UnloadModel(ctx context.Context, modelID string) error {
    req := ModelRequest{
        Command: UnloadModel,
        ModelID: modelID,
    }
    
    data, err := json.Marshal(req)
    if err != nil {
        return fmt.Errorf("failed to marshal request: %w", err)
    }

    header := []byte("MODEL:")
    if err := m.tcpClient.Send(ctx, string(append(header, data...))); err != nil {
        return fmt.Errorf("failed to send request: %w", err)
    }

    resp, err := m.tcpClient.ReceiveFullBytes(ctx, 4)
    if err != nil {
        return fmt.Errorf("failed to receive response size: %w", err)
    }

    size := binary.BigEndian.Uint32(resp)
    data, err = m.tcpClient.ReceiveFullBytes(ctx, int(size))
    if err != nil {
        return fmt.Errorf("failed to receive response: %w", err)
    }

    var modelResp ModelResponse
    if err := json.Unmarshal(data, &modelResp); err != nil {
        return fmt.Errorf("failed to unmarshal response: %w", err)
    }

    if !modelResp.Success {
        return fmt.Errorf("failed to unload model: %s", modelResp.Error)
    }

    return nil
}

func (m *ModelManager) CheckModel(ctx context.Context, modelID string) (ModelLocation, error) {
    req := ModelRequest{
        Command: CheckModel,
        ModelID: modelID,
    }
    
    data, err := json.Marshal(req)
    if err != nil {
        return None, fmt.Errorf("failed to marshal request: %w", err)
    }

    header := []byte("MODEL:")
    if err := m.tcpClient.Send(ctx, string(append(header, data...))); err != nil {
        return None, fmt.Errorf("failed to send request: %w", err)
    }

    resp, err := m.tcpClient.ReceiveFullBytes(ctx, 4)
    if err != nil {
        return None, fmt.Errorf("failed to receive response size: %w", err)
    }

    size := binary.BigEndian.Uint32(resp)
    data, err = m.tcpClient.ReceiveFullBytes(ctx, int(size))
    if err != nil {
        return None, fmt.Errorf("failed to receive response: %w", err)
    }

    var modelResp ModelResponse
    if err := json.Unmarshal(data, &modelResp); err != nil {
        return None, fmt.Errorf("failed to unmarshal response: %w", err)
    }

    if !modelResp.Success {
        return None, fmt.Errorf("failed to check model: %s", modelResp.Error)
    }

    return modelResp.Location, nil
}
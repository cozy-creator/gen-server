package models

import (
	"encoding/json"

	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type EventType = string

const (
	StatusEventType EventType = "status"
	ErrorEventType  EventType = "error"
	OutputEventType EventType = "output"
)

type Event struct {
	bun.BaseModel `bun:"table:events,alias:e"`

	ID        uuid.UUID       `bun:",pk"`
	Type      EventType       `bun:",notnull"`
	Data      json.RawMessage `bun:",type:jsonb,notnull"`
	CreatedAt bun.NullTime    `bun:",nullzero,notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime    `bun:",nullzero,notnull"`
}

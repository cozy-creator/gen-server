package models

import (
	"encoding/json"

	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type JobStatus string

const (
	JobStatusFailed    JobStatus = "FAILED"
	JobStatusQueued    JobStatus = "IN_QUEUE"
	JobStatusCompleted JobStatus = "COMPLETED"
	JobStatusRunning   JobStatus = "IN_PROGRESS"
)

type Job struct {
	bun.BaseModel `bun:"table:jobs"`

	ID          uuid.UUID       `bun:",pk"`
	Status      JobStatus       `bun:",notnull"`
	Input       json.RawMessage `bun:",type:jsonb,notnull"`
	CompletedAt bun.NullTime    `bun:",nullzero,notnull"`
	CreatedAt   bun.NullTime    `bun:",nullzero,notnull,default:current_timestamp"`
}

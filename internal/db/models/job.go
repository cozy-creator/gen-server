package models

import (
	"encoding/json"

	"github.com/google/uuid"
	"github.com/uptrace/bun"

	"github.com/cozy-creator/gen-server/internal/types"
)

type Job struct {
	bun.BaseModel `bun:"table:jobs"`

	ID          uuid.UUID            `bun:",type:uuid,pk"`
	Status      types.JobStatus      `bun:",notnull"`
	Input       json.RawMessage 	`bun:"type:jsonb,notnull"`
	Images      []Image              `bun:"rel:has-many,join:id=job_id"`
	Events      []Event              `bun:"rel:has-many,join:id=job_id"`
	Metric      *JobMetric           `bun:"rel:has-one,join:id=job_id"`
	CompletedAt bun.NullTime         `bun:",nullzero"`
	UpdatedAt   bun.NullTime         `bun:",nullzero,notnull,default:current_timestamp"`
	CreatedAt   bun.NullTime         `bun:",nullzero,notnull,default:current_timestamp"`
}

func NewJob(id uuid.UUID, input json.RawMessage) *Job {
	return &Job{
		ID:     id,
		Input:  input,
		Status: types.StatusInQueue,
	}
}
package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type JobStatus string

const (
	JobStatusFailed    JobStatus = "FAILED"
	JobStatusQueued    JobStatus = "IN_QUEUE"
	JobStatusCompleted JobStatus = "COMPLETED"
	JobStatusProgress  JobStatus = "IN_PROGRESS"
)

type Job struct {
	bun.BaseModel `bun:"table:jobs"`

	ID          uuid.UUID    `bun:",type:uuid,pk"`
	Status      JobStatus    `bun:",notnull"`
	Input       []byte       `bun:",notnull"`
	Images      []Image      `bun:"rel:has-many,join:id=job_id"`
	Events      []Event      `bun:"rel:has-many,join:id=job_id"`
	Metric      *JobMetric   `bun:"rel:has-one,join:id=job_id"`
	CompletedAt bun.NullTime `bun:",nullzero"`
	UpdatedAt   bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
	CreatedAt   bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
}

func NewJob(id uuid.UUID, input []byte) *Job {
	return &Job{
		ID:     id,
		Input:  input,
		Status: JobStatusQueued,
	}
}

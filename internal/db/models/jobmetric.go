package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type JobMetric struct {
	bun.BaseModel `bun:"table:job_metrics"`

	ID            uuid.UUID    `bun:",type:uuid,pk"`
	JobID         uuid.UUID    `bun:",type:uuid,notnull"`
	InferenceTime float64      `bun:",notnull"`
	Job           *Job         `bun:"rel:belongs-to,join:job_id=id"`
	UpdatedAt     bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
	CreatedAt     bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
}

func NewJobMetric(jobID uuid.UUID) *JobMetric {
	return &JobMetric{
		JobID: jobID,
		ID:    uuid.Must(uuid.NewRandom()),
	}
}

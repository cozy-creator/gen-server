package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type JobMetric struct {
	bun.BaseModel `bun:"table:job_metrics"`

	ID            uuid.UUID    `bun:",pk"`
	JobID         uuid.UUID    `bun:",notnull"`
	InferenceTime float64      `bun:",notnull"`
	UpdatedAt     bun.NullTime `bun:",nullzero,notnull"`
	CreatedAt     bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
}

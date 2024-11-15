package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type Image struct {
	bun.BaseModel `bun:"table:images"`

	ID        uuid.UUID    `bun:",type:uuid,pk"`
	Url       string       `bun:",notnull"`
	MimeType  string       `bun:",notnull"`
	JobID     uuid.UUID    `bun:",type:uuid,notnull"`
	Job       *Job         `bun:"rel:belongs-to,join:job_id=id"`
	UpdatedAt bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
	CreatedAt bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
}

func NewImage(url string, jobID uuid.UUID, mimeType string) *Image {
	return &Image{
		Url:      url,
		JobID:    jobID,
		MimeType: mimeType,
		ID:       uuid.Must(uuid.NewRandom()),
	}
}

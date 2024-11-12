package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type Image struct {
	bun.BaseModel `bun:"table:images"`

	ID        uuid.UUID    `bun:",pk"`
	Url       string       `bun:",notnull"`
	JobID     uuid.UUID    `bun:",notnull"`
	UpdatedAt bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
	CreatedAt bun.NullTime `bun:",nullzero,notnull,default:current_timestamp"`
}

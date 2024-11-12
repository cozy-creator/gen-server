package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type APIKey struct {
	bun.BaseModel `bun:"table:api_keys"`

	ID        uuid.UUID    `bun:",pk,type:uuid"`
	KeyHash   string       `bun:",notnull"`
	KeyMask   string       `bun:",notnull"`
	IsRevoked bool         `bun:",notnull,default:false"`
	CreatedAt bun.NullTime `bun:",notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime `bun:",notnull,nullzero,default:current_timestamp"`
}

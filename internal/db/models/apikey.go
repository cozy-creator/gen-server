package models

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type APIKey struct {
	bun.BaseModel `bun:"table:api_keys"`

	ID        uuid.UUID    `bun:",type:uuid,pk"`
	KeyHash   string       `bun:",notnull"`
	KeyMask   string       `bun:",notnull"`
	IsRevoked bool         `bun:",notnull,default:false"`
	CreatedAt bun.NullTime `bun:",notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime `bun:",notnull,nullzero,default:current_timestamp"`
}

func NewAPIKey(keyHash, keyMask string) *APIKey {
	return &APIKey{
		KeyHash: keyHash,
		KeyMask: keyMask,
		ID:      uuid.Must(uuid.NewRandom()),
	}
}

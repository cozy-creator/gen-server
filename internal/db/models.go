// Code generated by sqlc. DO NOT EDIT.
// versions:
//   sqlc v1.27.0

package db

import (
	"database/sql"
)

type ApiKey struct {
	ID        int64
	KeyHash   string
	IsRevoked bool
	CreatedAt sql.NullTime
}
